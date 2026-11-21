"""The visual-inspection agent.

Pipeline per spectrum:
1. Load + preprocess the spectrum; extract features (smoothing, continuum,
   feature detection, per-arm SNR).
2. Incorporate redrock per-class results (GALAXY / QSO / STAR best z + chi2).
3. Retrieve knowledge-base sections relevant to the candidate hypotheses.
4. Build a "line check table" at candidate redshifts and present a digest
   to the LLM, which returns a structured verdict (spectype, z, quality)
   with an explicit evidence trail.
5. Deterministic post-processing: refine the redshift using the secure
   lines claimed by the LLM (line-anchored z) and validate the final
   classification against KB rules (fatal-problem checks).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .continuum import continuum_metrics
from .features import SpectrumFeatures
from .kb import KnowledgeBase
from .lines import LINE_TABLE
from .llm import LLMClient
from .spectrum import Spectrum

SYSTEM_PROMPT = """You are an expert spectroscopist performing visual inspection (VI) of DESI spectra. \
You grade each spectrum like a human expert: decide the object type, the redshift, and a VI quality \
score, and provide an explicit evidence trail showing how you reached the conclusion.

Grading scheme (VI_QUALITY, same spirit as DESI VI):
- 4.0  Confident classification, two or more secure features
- 3.0  Probable classification, at least one secure feature plus continuum, or many weak features
- 2.0  Possible classification, one strong feature but unsure what it is
- 1.0  Unlikely classification, one or some unidentified features
- 0.0  Nothing there - no signal
You may report half points (e.g. 3.5) if between categories.

Final classification must map to ONLY: "QSO", "GALAXY" (includes ELG/LRG/BGS), "STAR", or "UNKNOWN".

The "sig" values in the tables are the significance of the deviation from the \
local continuum in units of noise at the line-center pixel of a Gaussian-smoothed spectrum \
(sigma=15 A). Two significance measures are shown as "sig/ivar": the first is a local-contrast \
measure (feature vs the scatter of its own neighborhood, robust to continuum shape), the second \
is the propagated-ivar measure (feature vs the formal noise). |sig|>=4 is a robust detection; \
3-4 marginal; <3 weak. Negative sig = absorption, positive = emission. A feature that is weak in \
local-contrast but stronger in ivar is likely real but embedded in correlated noise. In \
particular, |sig_ivar| >= 10 proves a genuine flux excess exists at that wavelength even if the \
local-contrast sig is <4 (broad continuum curvature can inflate the local noise estimate); use \
sig_ivar for DETECTION and the local-contrast sig for judging how cleanly the feature stands out.

Contamination-zone policy (from the KB): lines in flagged zones (OH-zone >7800 A, \
arm-junction, blue-edge-noise, skyline-*) must be treated as suspect and corroborated by \
other lines before being trusted. HOWEVER: an OH-zone line with very high significance \
(|sig| >= 8) is almost certainly real (skyline subtraction residuals are typically <5 sigma); \
treat |sig|>=8 as real even in the OH zone, and only require corroboration for 4-8 sigma \
OH-zone lines. A resolved [O II] doublet (two components ~3 A*(1+z) apart, both with \
|sig|>=3) is strong confirmation that the feature is real [O II], even in the OH zone.

Quality grading note: the scheme explicitly allows q=3 for "many weak features" — a coherent \
pattern of several 3-4 sigma lines (e.g. Ca K, Ca H, G-band, H delta, H gamma all present and \
consistent at one redshift) supports a probable classification even if no single line reaches \
4 sigma. Do not grade such spectra at q<2 simply because individual lines are sub-4-sigma. \
Conversely, an LRG/BGS with a coherent pattern of 3+ stellar absorption lines at one redshift \
plus a 4000 A break should be graded at least 3.5 (likely 4.0), matching how experts grade \
red galaxies - the coherent multi-line pattern IS the secure evidence.

The full knowledge base (KB) is provided below. It encodes the domain rules (line lists, doublet \
spacings, ionization consistency, composite profiles, classification diagnostics). You MUST apply \
these rules in your analysis. Explicitly cite which lines you use as secure evidence and which KB \
rules you applied (in kb_rules_applied). Be skeptical: a misidentified line at the wrong redshift \
is the most common error.

===== KNOWLEDGE BASE =====

{KB}"""


def _build_system_prompt(kb) -> str:
    kb_text = "\n\n".join(f"### [{s.provenance}]\n{s.text}" for s in kb.sections)
    return SYSTEM_PROMPT.replace("{KB}", kb_text)


class Verdict:
    def __init__(self, targetid, filename, spectype, z, z_err, quality, evidence,
                 reasoning, candidates, refined_z, refined_z_err, n_secure_lines,
                 fatal_issues, raw=None):
        self.targetid = targetid
        self.filename = filename
        self.spectype = spectype
        self.z = z
        self.z_err = z_err
        self.quality = quality
        self.evidence = evidence  # list of dicts
        self.reasoning = reasoning
        self.candidates = candidates
        self.refined_z = refined_z
        self.refined_z_err = refined_z_err
        self.n_secure_lines = n_secure_lines
        self.fatal_issues = fatal_issues
        self.raw = raw or {}

    def to_dict(self):
        return {
            "targetid": self.targetid,
            "filename": self.filename,
            "spectype": self.spectype,
            "z": self.z,
            "z_err": self.z_err,
            "quality": self.quality,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "candidates": self.candidates,
            "refined_z": self.refined_z,
            "refined_z_err": self.refined_z_err,
            "n_secure_lines": self.n_secure_lines,
            "fatal_issues": self.fatal_issues,
        }


def _fmt_line_check(rows, max_rows=None):
    """Format a line-check table compactly."""
    out = []
    header = f"{'line':<12} {'rest':>7} {'type':>4} {'w_obs':>9} {'sig/ivar':>14} {'cont':>8} {'dev':>8} {'flags'}"
    out.append(header)
    n = 0
    for r in rows:
        extra = ""
        if r.get("doublet"):
            d = r["doublet"]
            extra = f"  [OII]dbl a={d['a_sig']:+.1f} b={d['b_sig']:+.1f}"
        if r["in_range"] and r["meas"] is not None:
            m = r["meas"]
            flags = ",".join(r["flags"]) if r["flags"] else "-"
            out.append(
                f"{r['name']:<12} {r['rest']:7.1f} {r['type']:>4} {r['w_obs']:9.1f} "
                f"{m['sig']:+7.1f}/{m['sig_ivar']:+6.1f} {m['cont']:8.2f} {m['dev']:+8.2f} {flags}{extra}"
            )
            n += 1
        else:
            flags = ",".join(r["flags"]) if r["flags"] else "out-of-range"
            out.append(
                f"{r['name']:<12} {r['rest']:7.1f} {r['type']:>4} {r['w_obs']:9.1f} "
                f"{'--':>7} {'--':>8} {'--':>8} {flags}{extra}"
            )
    return "\n".join(out)


class VIAgent:
    def __init__(self, kb: KnowledgeBase | None = None, llm: LLMClient | None = None,
                 max_candidate_z: int = 3, verbose: bool = False):
        self.kb = kb or KnowledgeBase()
        self.llm = llm or LLMClient()
        self.max_candidate_z = max_candidate_z
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    def analyze(self, filename: str, redrock_per_class: dict | None = None,
                redrock_best: dict | None = None,
                zscan_candidates: list | None = None) -> Verdict:
        with Spectrum(filename) as spec:
            feats = SpectrumFeatures(spec)
            return self._analyze_features(spec, feats, redrock_per_class,
                                          redrock_best, zscan_candidates)

    def _analyze_features(self, spec, feats, redrock_per_class, redrock_best,
                          zscan_candidates=None) -> Verdict:
        targetid = spec.targetid

        # --- candidate redshifts -------------------------------------------------
        candidates = {}
        if redrock_per_class:
            for cls, info in redrock_per_class.items():
                candidates[f"redrock-{cls}"] = {"z": info["z"], "chi2": info["chi2"]}
        if redrock_best:
            candidates["redrock-best"] = {
                "z": redrock_best["z"],
                "chi2": redrock_best["chi2"],
                "spectype": redrock_best["spectype"],
            }
        if zscan_candidates:
            for i, c in enumerate(zscan_candidates):
                candidates[f"zscan-{c.get('class','?')}-{i}"] = {
                    "z": c["z"], "chi2": c["chi2"], "spectype": c.get("class")
                }
        # dedupe, keep lowest chi2 per unique z
        deduped = {}
        for k, v in candidates.items():
            zkey = round(v["z"], 4)
            if zkey not in deduped or v["chi2"] < deduped[zkey]["chi2"]:
                deduped[zkey] = dict(v, label=k)
        candidates = sorted(deduped.values(), key=lambda c: c["chi2"])
        candidates = candidates[: self.max_candidate_z]

        if not candidates:
            # fall back to a coarse scan: just use z=0 and a few trial values
            candidates = [{"z": 0.0, "chi2": np.inf, "label": "fallback"}]

        # --- line check tables at candidate z ------------------------------------
        check_tables = []
        for c in candidates:
            rows = feats.check_lines_at_z(c["z"])
            check_tables.append((c, rows))

        # --- continuum metrics at best candidate ---------------------------------
        best_z = candidates[0]["z"] if candidates else 0.0
        cont = continuum_metrics(feats.wave, feats.flux, best_z)

        # --- KB retrieval ----------------------------------------------------------
        query = self._kb_query(candidates, feats)
        kb_sections = self.kb.retrieve(query, k=5)
        retrieved_titles = [s.provenance for s, sc in kb_sections]
        self._last_retrieved = retrieved_titles
        kb_hint = (
            "Retrieved KB sections most relevant to this case (for your audit trail):\n"
            + "\n".join(f"- {t}" for t in retrieved_titles)
        )

        # --- build digest -----------------------------------------------------------
        digest = self._build_digest(spec, feats, candidates, check_tables, cont)

        user_prompt = f"""## Spectrum

{digest}

## Knowledge base sections relevant to this case

{kb_hint}
(The full KB with all rules is in the system prompt.)

## Task

Inspect the spectrum and produce your verdict as a JSON object with EXACTLY these keys:

{{
  "spectype": "QSO" | "GALAXY" | "STAR" | "UNKNOWN",
  "redshift": <float, your best redshift estimate>,
  "redshift_err": <float>,
  "vi_quality": <float 0-4>,
  "secure_lines": [
    {{"line": "<name>", "kind": "em"|"abs", "w_obs": <float>, "rest": <float>,
      "sig": <float>, "z_from_line": <float, = w_obs/rest - 1>, "note": "<why this is secure>"}}
  ],
  "rejected_lines": [
    {{"line": "<name>", "kind": "em"|"abs", "w_obs": <float>, "rest": <float>, "sig": <float>,
      "reason": "<why rejected, e.g. zone flag, wrong spacing, missing partner>"}}
  ],
  "kb_rules_applied": ["<rule 1>", "<rule 2>", ...],
  "reasoning": "<concise narrative of how you reached the conclusion, max ~150 words>",
  "fatal_problems": ["<any fatal-problem check that failed, or []>"]
}}

Rules:
- secure_lines must contain only lines you are confident in; prefer cross-checking at least 2 lines.
- redshift should be consistent with your secure lines (they are the evidence).
- For galaxies, absorption (Ca K/H, G-band, etc.) and the 4000 A break are the key evidence; \
for QSO, broad lines (Ly alpha, C IV, C III], Mg II) and/or narrow high-ionization lines ([Ne V]).
- If nothing is detectable, spectype="UNKNOWN", vi_quality=0.
- Report vi_quality per the grading scheme in the system prompt.
- Be CONCISE: max 6 secure_lines, max 6 rejected_lines, short notes (one sentence),
  reasoning max 120 words. Do not repeat the tables back. Respond with a COMPLETE,
  well-formed JSON object.
- Use EXACTLY the key names given above. Do NOT invent other keys such as VI_SPECTYPE,
  VI_Z, VI_QUALITY, TARGETID, or evidence_trail - the schema is fixed."""

        system = _build_system_prompt(self.kb)
        raw = self.llm.complete_json(system, user_prompt)

        verdict = self._postprocess(raw, spec, feats, candidates, check_tables)
        return verdict

    # ------------------------------------------------------------------ #
    def _kb_query(self, candidates, feats) -> str:
        """Build a retrieval query from the candidate hypotheses."""
        parts = []
        for c in candidates[:3]:
            parts.append(f"redshift z={c['z']:.3f}")
            st = str(c.get("spectype", "")).upper()
            if "QSO" in st:
                parts.append("QSO broad lines Mg II C IV C III] classification")
            elif "GALAXY" in st or "STAR" in st:
                parts.append("galaxy absorption Ca K/H 4000 break classification")
        # add type-agnostic diagnostics terms
        parts.append("line identification doublet spacing ionization")
        return " ".join(parts)

    def _build_digest(self, spec, feats, candidates, check_tables, cont) -> str:
        lines = []
        lines.append(f"TARGETID: {spec.targetid}")
        lines.append(f"file: {os.path.basename(spec.filename)}")
        lines.append(
            f"Per-arm median SNR: B={spec.snr_arm['B']:.2f}  R={spec.snr_arm['R']:.2f}  "
            f"Z={spec.snr_arm['Z']:.2f}   (sum={spec.snr_sum:.1f})"
        )
        lines.append("")
        lines.append("redrock (pipeline) candidates:")
        for c in candidates:
            lbl = c.get("label", "?")
            st = c.get("spectype", "")
            lines.append(f"  [{lbl}] z={c['z']:.4f}  chi2={c['chi2']:.1f}  spectype={st}")
        lines.append("")
        lines.append("Continuum metrics at best candidate z=%.4f:" % (candidates[0]["z"] if candidates else 0.0))
        lines.append(
            f"  Dn4000={cont['Dn4000']:.3f}  BalmerBreak={cont['BalmerBreak']:.3f}  "
            f"slope(4500-6000)={cont['slope_4500_6000']:.4f}/kA"
        )
        lines.append("")
        lines.append(feats.feature_summary())
        lines.append("")
        for c, rows in check_tables:
            lines.append(f"--- Line check table at z={c['z']:.4f} (label {c.get('label','?')}) ---")
            lines.append(_fmt_line_check(rows))
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def _postprocess(self, raw, spec, feats, candidates, check_tables) -> Verdict:
        """Validate the LLM verdict, refine z from secure lines, check KB fatal rules."""
        spectype = str(raw.get("spectype", "UNKNOWN")).upper().strip()
        # normalize subtype labels to top-level categories (KB rule)
        if spectype in ("ELG", "LRG", "BGS", "GALAXY", "GALAXIES"):
            spectype = "GALAXY"
        elif spectype in ("QSO", "QUASAR", "AGN"):
            spectype = "QSO"
        elif spectype in ("STAR", "STARS"):
            spectype = "STAR"
        else:
            spectype = "UNKNOWN"
        z_llm = float(raw.get("redshift", 0.0) or 0.0)
        z_err_llm = float(raw.get("redshift_err", 0.0) or 0.0)
        quality = float(raw.get("vi_quality", 0.0) or 0.0)
        quality = max(0.0, min(4.0, quality))
        secure = raw.get("secure_lines", []) or []
        rejected = raw.get("rejected_lines", []) or []
        reasoning = str(raw.get("reasoning", ""))
        kb_rules = raw.get("kb_rules_applied", []) or []
        fatal = raw.get("fatal_problems", []) or []

        # --- refine z from secure lines ------------------------------------------
        z_list, w_list = [], []
        evidence = []
        for s in secure:
            try:
                rest = float(s.get("rest", 0))
                w_obs = float(s.get("w_obs", 0))
                sig = float(s.get("sig", 0))
            except (TypeError, ValueError):
                continue
            if rest <= 0:
                continue
            z_line = w_obs / rest - 1.0
            # exclude absurd values
            if not (-0.1 < z_line < 7.0):
                continue
            z_list.append(z_line)
            w_list.append(abs(sig))
            evidence.append({
                "line": s.get("line", "?"),
                "kind": s.get("kind", "em"),
                "w_obs": w_obs,
                "rest": rest,
                "sig": sig,
                "z_from_line": z_line,
                "note": s.get("note", ""),
                "used": True,
            })
        for s in rejected:
            evidence.append({
                "line": s.get("line", "?"),
                "kind": s.get("kind", "em"),
                "w_obs": s.get("w_obs"),
                "rest": s.get("rest"),
                "sig": s.get("sig"),
                "z_from_line": None,
                "note": s.get("reason", ""),
                "used": False,
            })

        refined_z, refined_z_err, n_secure = None, None, 0
        if len(z_list) >= 1:
            z_arr = np.array(z_list)
            w_arr = np.array(w_list)
            refined_z = float(np.average(z_arr, weights=w_arr))
            # scatter estimate
            if len(z_arr) >= 2:
                refined_z_err = float(np.std(z_arr))
            else:
                refined_z_err = float(z_err_llm if z_err_llm > 0 else 0.005)
            n_secure = len(z_arr)

        # choose final z: prefer the line-anchored (refined) z, since secure
        # lines are the primary evidence; fall back to LLM z only if no lines.
        final_z = z_llm
        final_z_err = z_err_llm
        if refined_z is not None:
            final_z = refined_z
            final_z_err = refined_z_err
            if abs(refined_z - z_llm) > max(0.02, 0.5 * (1 + z_llm) * 0.01):
                fatal.append(
                    f"LLM z ({z_llm:.4f}) vs secure-line z ({refined_z:.4f}) "
                    f"disagree by {abs(refined_z - z_llm):.4f}; using line-anchored z"
                )

        # --- KB fatal-problem checks (deterministic) --------------------------------
        fatal_issues = self._kb_fatal_checks(spec, feats, spectype, final_z, evidence)
        fatal_issues += [str(f) for f in fatal]

        # --- deterministic quality recalibration (enforce grading-scheme consistency)
        quality = self._recalibrate_quality(quality, n_secure, feats, evidence, final_z)
        candidates_out = [
            {"label": c.get("label", "?"), "z": c["z"], "chi2": c.get("chi2", np.inf),
             "spectype": c.get("spectype", "")}
            for c in candidates
        ]
        return Verdict(
            targetid=spec.targetid,
            filename=os.path.basename(spec.filename),
            spectype=spectype,
            z=final_z,
            z_err=final_z_err,
            quality=quality,
            evidence=evidence,
            reasoning=reasoning,
            candidates=candidates_out,
            refined_z=refined_z,
            refined_z_err=refined_z_err,
            n_secure_lines=n_secure,
            fatal_issues=fatal_issues,
            raw={"kb_rules_applied": kb_rules,
                 "kb_retrieved": getattr(self, "_last_retrieved", [])},
        )

    # ------------------------------------------------------------------ #
    def _recalibrate_quality(self, quality, n_secure, feats, evidence, z):
        """Deterministic quality guard.

        Currently only handles the clearest failure mode: the LLM claimed a
        high quality while there is NO significant feature anywhere in the
        spectrum (nothing there -> q<=1). Heavier recalibration was tested
        and hurt aggregate MAE, so it is intentionally conservative.
        """
        det_max_sig = 0.0
        for f in feats.features[:8]:
            det_max_sig = max(det_max_sig, abs(f["sig"]))
        if det_max_sig < 2.5 and quality >= 2.0:
            return 1.0
        return quality

    def _kb_fatal_checks(self, spec, feats, spectype, z, evidence):
        """Deterministic fatal-problem checks grounded in KB rules."""
        issues = []
        used_lines = [e["line"] for e in evidence if e.get("used")]
        used = set(used_lines)

        if spectype == "GALAXY":
            # Ca K/H must appear together; Ca K deeper than Ca H
            has_k = "Ca K_abs" in used
            has_h = "Ca H_abs" in used
            if has_k != has_h:
                issues.append("KB: Ca K/H missing partner (one present without the other)")
            # [Ne V] check: a strong [Ne V] implies AGN-level ionization (Type 2 QSO)
            nev = feats.measure_line(3426.0 * (1 + z))
            if nev is not None and nev["dev"] > 0:
                if abs(nev["sig"]) >= 4.0 or abs(nev.get("sig_ivar", 0)) >= 15.0:
                    issues.append(
                        "KB: strong [Ne V] 3426 detected at z=%.4f (sig_ivar=%.1f) - requires "
                        "AGN ionization; classifying as GALAXY may be wrong (Type 2 QSO)"
                        % (z, nev.get("sig_ivar", 0))
                    )
        if spectype == "QSO":
            # check [Ne V] if claimed as the only secure feature -> needs care
            # check broad-line requirement is handled in the LLM reasoning
            pass
        # [O III] without [O II] (if both in range) - ionization consistency
        if "[O III]b" in used and "[O II]" not in used:
            # check whether [O II] would be in range
            for ln in LINE_TABLE:
                if ln["name"] == "[O II]":
                    w_obs = ln["rest"] * (1 + z)
                    if feats.wave[0] + 30 < w_obs < feats.wave[-1] - 30:
                        issues.append("KB: [O III] claimed without [O II] in range (ionization inconsistency)")
        return issues


def run_agent_on_file(filename, redrock_per_class=None, redrock_best=None,
                      kb=None, llm=None, verbose=False):
    agent = VIAgent(kb=kb, llm=llm, verbose=verbose)
    return agent.analyze(filename, redrock_per_class, redrock_best)
