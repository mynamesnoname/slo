"""LLM visual-inspection agent for DESI spectra.

Takes the deterministic feature report (from features.py) plus the redrock
pipeline fit, and produces a VI-style result:

  vi_z          : best redshift
  vi_spectype   : QSO | GALAXY
  vi_quality    : 0-4 following the DESI VI grading scheme
  evidence      : list of concrete spectral features used
  flags         : R (bad redshift fit), C (bad spectype fit), S (bad spectrum)
  reasoning     : auditable free-text explanation

The agent is a plain LLM call with a carefully designed prompt; outputs are
parsed from JSON and validated.
"""

import json
import os
import re

from features import LINE_CATALOG, lines_at_z, near_sky

# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------

GRADING_RUBRIC = """\
VI_QUALITY grading (official DESI scheme):
  [4] Confident: two or more secure, mutually consistent features (e.g. two
      strong lines at the same redshift, or one strong line + a strong
      continuum break / forest).
  [3] Probable: at least one secure feature plus a consistent continuum shape,
      or many weak features that agree.
  [2] Possible: one strong emission feature, but you are not sure what it is.
  [1] Unlikely: one or some unidentified features.
  [0] Nothing there: no signal (noise-dominated spectrum).

Optional flags:
  R : the pipeline (redrock) redshift is wrong
  C : the pipeline spectype is wrong
  S : the spectrum itself is bad (e.g. sky/cosmic residuals, no usable flux)
"""


def build_report(feats, redrock=None, top_n=4, target_prior=None):
    """Build the text report handed to the LLM."""
    L = []
    if target_prior:
        L.append("TARGET SELECTION PRIOR (informational only, CAN be wrong): "
                 f"this object was targeted as a {target_prior}. "
                 "Use it as a hint (e.g. ELG targets usually show [OII] 3727 "
                 "emission; QSO targets usually show broad lines or power-law "
                 "continuum), but verify with the actual spectral lines.")
        L.append("")
    L.append("SPECTRUM GLOBAL PROPERTIES")
    L.append(f"- TARGETID: {feats['targetid']}")
    L.append(f"- Wavelength coverage: {feats['wmin']:.0f}-{feats['wmax']:.0f} A; "
             f"usable pixels fraction: {feats['ivar_fraction']:.2f}")
    L.append(f"- Global S/N (sum over pixels, SNR_sum-like): {feats['global_snr']:.1f}")
    L.append(f"- D4000 break index: {feats['d4000']:.3f}  "
             f"(>1.4 = strong old-galaxy break; ~1.0 = featureless/power-law continuum)")
    L.append(f"- Continuum slope alpha (d lnF/d ln lambda): {feats['slope']:+.2f} "
             f"(strongly negative = very blue, e.g. QSO power law; "
             f"positive = red, e.g. old galaxy)")
    if redrock is not None:
        L.append("")
        L.append("PIPELINE FIT (redrock, treated as a second opinion you may override)")
        L.append(f"- z={redrock.get('Z', float('nan')):.4f} "
                 f"spectype={redrock.get('SPECTYPE', '?')} "
                 f"subtype={redrock.get('SUBTYPE', '?')} "
                 f"chi2={redrock.get('CHI2', 0):.0f} "
                 f"deltachi2={redrock.get('DELTACHI2', 0):.0f} "
                 f"zwarn={redrock.get('ZWARN', 0)}")
        rr_z = redrock.get("Z")
        if rr_z is not None and not (isinstance(rr_z, float) and rr_z != rr_z):
            L.append("- Measured line S/N AT the redrock redshift "
                     "(positive=emission, negative=absorption):")
            tbl = lines_at_z(feats, float(rr_z), sn_thresh=2.0)
            if not tbl:
                L.append("    no catalog line with |S/N| >= 2 at the redrock z "
                         "(fit is unsupported by line evidence)")
            for l in tbl:
                sky = " [near sky line!]" if l.get("sky") else ""
                L.append(f"    {l['name']:<8} rest={l['rest']:7.2f} -> obs={l['obs']:7.1f} A "
                         f"S/N narrow={l['sn_narrow']:+6.1f} broad={l['sn_broad']:+6.1f} "
                         f"contrast={l.get('contrast', 0):.1f} "
                         f"[{l['kind']}, {l['fwhm_hint']}]{sky}")
    L.append("")
    L.append("CANDIDATE REDSHIFTS from a deterministic weighted line-S/N scan")
    L.append("(S/N is the integrated flux excess above the local continuum "
             "divided by the noise; positive = emission, negative = absorption).")
    cands = feats.get("candidates", [])
    if not cands:
        L.append("- No candidate redshift with any line S/N above threshold. "
                 "Spectrum is likely featureless/noise.")
    best_nsec = max([c.get("n_secure", 0) for c in cands], default=0)
    nsec2 = len([c for c in cands if c.get("n_secure", 0) >= 2])
    L.append("")
    L.append(f"NOISE/CONFIDENCE CONTEXT: the strongest candidate has "
             f"{best_nsec} secure line(s); {nsec2} candidate(s) have >=2 secure lines. "
             f"Global S/N sum is {feats['global_snr']:.0f}.")
    if best_nsec == 0:
        L.append("DETERMINISTIC QUALITY HINT: no secure feature anywhere -> "
                 "the spectrum is likely noise-dominated; grade q=0 or q=1.")
    elif best_nsec == 1:
        L.append("DETERMINISTIC QUALITY HINT: only one candidate with a single "
                 "secure feature -> at most q=2 unless the continuum shape "
                 "strongly supports an identification (then q=3).")
    else:
        L.append("DETERMINISTIC QUALITY HINT: at least one candidate has >=2 "
                 "mutually consistent secure lines -> q=3 or q=4 is possible "
                 "if the lines are credible.")
    for i, c in enumerate(cands[:top_n]):
        nlines = len([l for l in c["lines"] if abs(l[1]) > 2.5 or abs(l[2]) > 2.5])
        nsec = c.get("n_secure", 0)
        L.append(f"- Candidate #{i + 1}: z={c['z']:.4f} (scan score {c['score']:.0f}, "
                 f"{nlines} line(s) |S/N|>2.5, {nsec} secure (|S/N|>=3))")
        for name, snn, snb, kind, *rest in c["lines"]:
            contrast = rest[0] if rest else 0.0
            rw = LINE_CATALOG[name][0]
            obs = rw * (1 + c["z"])
            sky = " [near sky line!]" if near_sky(obs) else ""
            L.append(f"    {name:<8} rest={rw:7.2f} -> obs={obs:7.1f} A  "
                     f"S/N narrow={snn:+6.1f} broad={snb:+6.1f} "
                     f"contrast={contrast:.1f}  [{kind}]{sky}")
    if "lya_forest_at_best" in feats:
        t, fr = feats["lya_forest_at_best"]
        L.append("")
        L.append(f"Ly-alpha forest check at best candidate z: mean transmission "
                 f"{t:.2f}, patchiness {fr:.2f} "
                 f"(transmission <0.9 with patchiness suggests real Ly-a forest = QSO)")
    L.append("")
    L.append("LINE CATALOG REFERENCE (rest-frame wavelengths, A)")
    for name, (rw, _, kind) in LINE_CATALOG.items():
        L.append(f"  {name:<8} {rw:7.2f}  {kind}")
    return "\n".join(L)


SYSTEM_PROMPT = """\
You are an expert spectroscopic visual-inspection (VI) scientist for the DESI \
survey. A human expert looks at a spectrum, identifies emission/absorption \
lines, judges their consistency at a trial redshift, and assigns a \
redshift, a spectral classification, and a confidence grade.

You are given:
  (1) quantitative, deterministic measurements of a DESI spectrum
      (continuum shape, line S/N at candidate redshifts), and
  (2) the pipeline (redrock) fit as a second opinion.

Your job: decide the VI result exactly like the expert, and justify it.

Rules:
- Decide the redshift from the SET of lines that are mutually consistent
  (same z). One strong line alone is weak evidence and may be a misidentified
  line. ALIAS WARNING: a strong low-z galaxy line can masquerade as a
  higher-z QSO line at a different rest wavelength, e.g. H-alpha 6563 at
  z~0.06 appears at ~6963 A and can be read as "MgII 2796 at z=1.49";
  [OIII]5007 at z~0.53 appears at ~7677 A and can be read as "CIII] 1909 at
  z=3.02"; H-alpha/[NII] at z~0.30 appears at ~8500-8540 A and can be read as
  "CIV at z=4.5". Always ask: is there a lower-z interpretation of the SAME
  observed features with MORE consistent lines (e.g. H-alpha + [NII] + [SII]
  + [OIII] + H-beta)? Prefer it.
- Classify: QSO if you see broad emission lines (Ly-a, CIV, CIII], MgII) and/or
  a Ly-alpha forest / power-law blue continuum. AGN AT LOW REDSHIFT: strongly
  broad Balmer lines (H-alpha/H-beta with FWHM much larger than narrow galaxy
  lines, i.e. broad S/N >> narrow S/N) or [NeV] 3426 emission are AGN
  signatures -> classify QSO even at low z. GALAXY if you see narrow
  emission ([OII], H-beta, [OIII], H-alpha) or Balmer/4000-A-break absorption
  features of a galaxy. NOTE: in passively-evolving galaxies the Balmer lines
  (H-beta, H-gamma, H-delta) and Ca H&K appear in ABSORPTION (negative S/N),
  which is a normal galaxy signature, not an error.
- Use redrock's z/spectype as a hint but verify it against the line evidence
  (a table of line S/N at the redrock z is given above): if those lines are
  real and mutually consistent, redrock is likely right. A large redrock
  deltachi2 means a confident fit; a tiny deltachi2 (<~10) means the fit is
  not much better than alternatives -> do NOT trust redrock's z in that case.
  Flag R and give your own z only if the lines clearly disagree.
- Grade quality strictly with the official rubric below, and respect the
  DETERMINISTIC QUALITY HINT: never give q=4 unless you have two or more
  credible mutually consistent features, and never claim a confident
  classification on an isolated |S/N| 3-4 feature.
- NOISE CALIBRATION: in a noisy spectrum (few secure features anywhere, small
  redrock deltachi2), isolated features of |S/N| 3-4 occur by chance and are
  NOT reliable. Require |S/N| >= 5 for a single feature to count as secure, or
  two features at the same z with |S/N| >= 3.5. When genuinely unsure, give the
  LOWER quality grade (q=0 or q=1) and do not force an identification: grading
  a noise spectrum as q=0 with the best-guess z is a valid, honest answer.
- A line with |S/N| >= 3 is nominally "secure"; 2 <= |S/N| < 3 is "weak"
  (but see the noise calibration above). CONTRAST: each line also reports
  "contrast" = line peak / continuum level. High-contrast lines (>=3) are
  robust even when the formal S/N is modest (they may sit in noisy pixels);
  low-contrast (<2) "features" with S/N 3-5 are often just noise accumulation
  and must not anchor a redshift by themselves. Beware: sky-line residuals and
  noise peaks can masquerade as lines; multiple consistent lines are far more
  trustworthy than one. A line marked "[near sky line!]" lies within ~2.5 A of
  a bright OH/atomic sky emission line: treat it as SUSPECT but not
  automatically fake - if it is strong (|S/N| >= 5) and/or has an independent
  partner line at the same z, it is probably real; if it is weak (|S/N| < 5)
  and isolated, do not build your classification on it.

{grading}
Output ONLY a JSON object with these fields:
{{
  "vi_z": <float, your best redshift>,
  "vi_spectype": "QSO" or "GALAXY",
  "vi_quality": <int 0-4>,
  "evidence": [<string per secure feature, e.g. "CIV 1549 broad emission at z=2.31 (S/N 38)">],
  "flags": [<"R" and/or "C" and/or "S", or empty list>],
  "reasoning": <2-5 sentences explaining how you reached the conclusion>
}}
Do not include anything outside the JSON object.
""".format(grading=GRADING_RUBRIC)


def _extract_json(text):
    """Robustly extract a JSON object from an LLM response."""
    text = text.strip()
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # try to strip markdown fences
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


def fallback_vi(feats, redrock=None):
    """Deterministic fallback used when the LLM call fails.

    Uses the best candidate's z (or redrock z if it is better supported),
    classifies from the line mix, and grades quality from the number of
    secure features following the official rubric.
    """
    cands = feats.get("candidates", [])
    chosen = None
    if cands:
        chosen = cands[0]
    rr_z = None
    if redrock is not None:
        rr_z = redrock.get("Z")
    # decide between best candidate and redrock by line support at each z
    def support(z):
        try:
            tbl = lines_at_z(feats, z, sn_thresh=3.0)
            return len(tbl)
        except Exception:  # noqa: BLE001
            return 0
    if chosen is not None and rr_z is not None and \
            not (isinstance(rr_z, float) and rr_z != rr_z):
        if support(rr_z) > support(chosen["z"]) + 1:
            z = float(rr_z)
            used_rr = True
        else:
            z = chosen["z"]
            used_rr = False
    elif chosen is not None:
        z = chosen["z"]; used_rr = False
    elif rr_z is not None and not (isinstance(rr_z, float) and rr_z != rr_z):
        z = float(rr_z); used_rr = True
    else:
        z = float("nan"); used_rr = False
    tbl = lines_at_z(feats, z, sn_thresh=2.0) if z == z else []
    n_sec = len([l for l in tbl if abs(l["sn_narrow"]) >= 3 or abs(l["sn_broad"]) >= 3])
    # classification from line mix
    qso_names = {"LyA", "SiIV", "CIV", "HeII", "CIII]", "MgII", "NeV"}
    gal_names = {"OII", "Hdelta", "Hgamma", "Hbeta", "OIII4959", "OIII5007",
                 "NII", "Halpha", "SII6716", "SII6731", "NeIII", "CaK", "CaH",
                 "MgI", "NaD"}
    qso_score = sum(1 for l in tbl if l["name"] in qso_names and
                    (abs(l["sn_narrow"]) >= 3 or abs(l["sn_broad"]) >= 3))
    gal_score = sum(1 for l in tbl if l["name"] in gal_names and
                    (abs(l["sn_narrow"]) >= 3 or abs(l["sn_broad"]) >= 3))
    if qso_score > gal_score:
        stype = "QSO"
    elif gal_score > qso_score:
        stype = "GALAXY"
    else:
        rr_type = (redrock or {}).get("SPECTYPE", "")
        stype = "QSO" if rr_type == "QSO" else "GALAXY"
    if n_sec >= 2:
        q = 3 if len(tbl) < 3 else 4
    elif n_sec == 1:
        q = 2
    else:
        q = 0
    return {
        "targetid": feats.get("targetid"),
        "vi_z": float(z), "vi_spectype": stype, "vi_quality": q,
        "evidence": [f"fallback: {l['name']} at z={z:.4f} S/N {l['sn_narrow']:.1f}"
                     for l in tbl if abs(l["sn_narrow"]) >= 3 or abs(l["sn_broad"]) >= 3],
        "flags": (["R"] if used_rr else []) ,
        "reasoning": f"deterministic fallback (LLM failed): used "
                     f"{'redrock' if used_rr else 'best scan candidate'} z={z:.4f}, "
                     f"n_secure={n_sec}, line mix -> {stype}",
        "line_table_at_z": tbl,
        "fallback": True,
    }


def calibrate_quality(res, feats):
    """Post-hoc check: enforce the official rubric with the deterministic
    line table at the chosen z. Returns (new_quality, capped)."""
    q = res.get("vi_quality", -1)
    z = res.get("vi_z")
    if q < 3 or z is None or (isinstance(z, float) and z != z):
        return q, False
    tbl = res.get("line_table_at_z") or []
    n_sec = len([l for l in tbl if abs(l.get("sn_narrow", 0)) >= 3
                 or abs(l.get("sn_broad", 0)) >= 3])
    if q >= 4 and n_sec < 2:
        return 3 if n_sec >= 1 else 2, True
    if q == 3 and n_sec < 1:
        return 2, True
    return q, False


class VIAgent:
    def __init__(self, env_path=None, model=None, temperature=0.0):
        if env_path is None:
            env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        env = {}
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
        from openai import OpenAI
        self.client = OpenAI(api_key=env["API_KEY"], base_url=env["BASE_URL"])
        self.model = model or env.get("LLM_MODEL", "deepseek-v4-flash")
        self.temperature = temperature

    def inspect(self, feats, redrock=None, target_prior=None, max_retries=3):
        """Run the VI agent on one spectrum; returns dict with result + audit."""
        report = build_report(feats, redrock, target_prior=target_prior)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": report},
        ]
        audit = {"report": report, "attempts": []}
        last = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=32000,
                )
                raw = resp.choices[0].message.content or ""
                reason = getattr(resp.choices[0].message, "reasoning_content", None) or ""
                audit["attempts"].append(raw)
                audit["last_reasoning"] = reason
                parsed = _extract_json(raw)
                if parsed is None:
                    last = "unparseable"
                    continue
                parsed["vi_z"] = float(parsed.get("vi_z", float("nan")))
                parsed["vi_spectype"] = str(parsed.get("vi_spectype", "?")).upper()
                parsed["vi_quality"] = int(round(float(parsed.get("vi_quality", -1))))
                parsed.setdefault("evidence", [])
                parsed.setdefault("flags", [])
                parsed.setdefault("reasoning", "")
                # attach the deterministic line table at the chosen z
                # (isolated: a failure here must not discard the LLM result)
                try:
                    parsed["line_table_at_z"] = lines_at_z(
                        feats, parsed["vi_z"], sn_thresh=1.5)
                except Exception:  # noqa: BLE001
                    parsed["line_table_at_z"] = []
                q_cap, capped = calibrate_quality(parsed, feats)
                parsed["vi_quality"] = q_cap
                parsed["quality_capped"] = capped
                parsed["targetid"] = feats["targetid"]
                parsed["_audit"] = audit
                return parsed
            except Exception as e:  # noqa: BLE001
                last = f"error: {e}"
                audit["attempts"].append(last)
        fb = fallback_vi(feats, redrock)
        fb["_audit"] = audit
        fb["reasoning"] += f" | last LLM error: {last}"
        return fb


if __name__ == "__main__":
    import sys
    from features import extract_features
    path = sys.argv[1]
    feats = extract_features(path)
    redrock = None
    if len(sys.argv) > 2 and sys.argv[2] != "-":
        import pickle
        with open(sys.argv[2], "rb") as fh:
            redrock = pickle.load(fh)
    agent = VIAgent()
    res = agent.inspect(feats, redrock)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("_")},
                     indent=2, ensure_ascii=False))
