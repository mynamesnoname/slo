# VI Agent: Automatic Visual Inspection of DESI Spectra

An LLM agent that performs visual inspection (VI) of DESI spectra: it assesses the
redshift, the object type (QSO / Galaxy), and spectral characteristics, and produces
an **auditable** evidence trail per spectrum.

## Architecture

```
FITS spectrum ──► spectrum.py   (load B/R/Z arms, masks, per-arm SNR)
              ──► features.py   (Gaussian smoothing σ=15 Å, band-pass local-contrast
                                 significance, peak detection, line-window measurement,
                                 contamination-zone flags)
              ──► redrock       (pipeline candidate redshifts per template class:
                                 GALAXY / QSO / STAR best z + chi2, plus zscan minima)
              ──► kb.py         (knowledge base: sectioned kb/*.md, TF-IDF retrieval)
              ──► agent.py      (digest → LLM verdict JSON → deterministic
                                 post-processing: line-anchored z refinement,
                                 KB fatal-problem checks, quality guard)
              ──► evaluate.py   (metrics vs expert VI ground truth)
```

### Feature extraction (features.py)

- Merges the three DESI arms (B 3600–5800, R 5760–7620, Z 7520–9824 Å), excludes
  masked / non-positive-ivar pixels.
- **Smoothing** with a Gaussian of σ = 15 Å (the value human VI experts use).
- **Significance, dual measure**: a band-pass (difference-of-Gaussians)
  *local-contrast* significance (feature vs scatter of its own neighborhood,
  robust to continuum shape) and a *propagated-ivar* significance (feature vs
  formal noise, catches real lines hidden by broad continuum curvature).
- **Line measurement** at any observed wavelength: deviation, both significances,
  local-extremum check, contamination-zone flags (blue edge <4000 Å, B–R and R–Z
  junctions, OH airglow >7800 Å, known skylines 5577/5890-96/6300/6364/6867/7605 Å).
- **[O II] unresolved-doublet check** (3726.0/3729.0 measured separately — a
  resolved doublet confirms real [O II] even in the OH zone).
- **Line check table**: every rest-frame line in the KB table measured at each
  trial redshift.

### Redrock integration

`rrdesi` is run once over all training spectra
(`RR_TEMPLATE_DIR=/data2/wbc/redrock/py/redrock/templates`). Per target we extract
the best fit, per-class (GALAXY/QSO/STAR) best z+chi2, and the local minima of the
chi2-vs-z scan (diverse trial redshifts, like an expert trying alternatives).
Redrock is a *hypothesis generator*, never ground truth — candidates are verified
against measured features and KB rules.

### Knowledge base (kb.py)

`kb/*.md` (instrument, lines, classification, ionization, composite_profile) is
split into headed sections. TF-IDF retrieval returns the sections most relevant
to the candidate hypotheses; the **full KB is injected into the system prompt**
so the rules always inform decisions. The LLM cites applied rules
(`kb_rules_applied`) and retrieved sections are recorded in the verdict.

### Agent loop (agent.py)

1. Build digest: per-arm SNR, redrock candidates, continuum metrics (Dn4000,
   Balmer break, slope), strongest features (with both significances), and line
   check tables at up to 3 candidate redshifts.
2. LLM (deepseek, json mode) returns: `spectype`, `redshift`, `vi_quality`
   (0–4 per DESI VI scheme), `secure_lines` (with per-line z), `rejected_lines`,
   `kb_rules_applied`, `reasoning`, `fatal_problems`.
3. Deterministic post-processing:
   - subtype normalization (ELG/LRG/BGS → GALAXY),
   - **line-anchored redshift** (weighted mean of z from secure lines),
   - KB fatal-problem checks ([O III] without [O II]; Ca K without Ca H;
     **strong [Ne V] → possible Type 2 QSO**; quality guard if nothing >2.5σ).

### Evaluation (evaluate.py)

Metrics vs expert VI: type accuracy (QSO vs Galaxy), z accuracy (|Δv| < 600 km/s),
median |Δz|, quality MAE and exact-match, per type. An ensemble runner
(`run_ensemble.py`) combines multiple runs by majority vote on type, cluster
consensus on z, and median on quality, to reduce per-run LLM variance.

## Results (final code, training set)

Expert VI labels were **removed from the agent's input** (an early version leaked
them via the FIBERMAP and inflated scores; the leak is fixed).

Single-run results of the final code (5 runs, 79–80/81 spectra each;
the 1–2 missing spectra per run are transient LLM JSON failures, recovered by
the ensemble):

| run | type_acc | z_acc (<600 km/s) | median dz | q MAE | q exact |
|-----|----------|-------------------|-----------|-------|---------|
| run10 | 0.925 | 0.913 | 0.00009 | 0.460 | 0.51 |
| run11 | 0.938 | 0.900 | 0.00009 | 0.410 | 0.51 |
| run12 | 0.938 | 0.875 | 0.00010 | 0.354 | 0.59 |
| run13 | 0.937 | 0.848 | 0.00010 | 0.416 | 0.54 |
| run14 | 0.925 | 0.863 | 0.00020 | 0.373 | 0.56 |

Baseline (redrock alone): type 0.877, z 0.877.

**Final ensemble (run12+13+14, majority vote / cluster consensus / median):**
type **0.937**, z **0.886** (<600 km/s), median |dz| **0.00015**,
quality MAE **0.327**, quality exact **0.57**, n=79/81.
Per type: BGS 1.00/1.00; ELG 1.00/0.85; LRG 0.94/0.94; QSO 0.81/0.76
(type/z).

### Sample auditable verdict (QSO/29; expert: QSO z=1.4975 q=4)

```
spectype: QSO | z=1.49564 | quality: 4.0  (3/3 runs agree)
evidence:
 [SECURE] C IV   em   w_obs=3866.4  sig=195.0  z_line=1.4961  very strong broad BLR line
 [SECURE] C III] em   w_obs=4758.4  sig=90.4   z_line=1.4926  broad emission
 [SECURE] Mg II  em   w_obs=6992.0  sig=125.7  z_line=1.4971  FWHM>2000 km/s anchor
 [SECURE] [O II] em   w_obs=9304.8  sig=7.0    z_line=1.4966  resolved doublet in OH zone
reasoning: broad lines C IV/C III]/Mg II at z~1.496, widths imply Type 1 QSO;
           narrow [O II] doublet at same z corroborates systemic redshift.
```

## Iteration history

| Run | Change | type | z | q_MAE |
|-----|--------|------|---|-------|
| baseline | redrock alone | 0.877 | 0.877 | - |
| run1 | first agent; ivar-based significance | 0.962* | 0.875* | 0.215* |
| run2 | band-pass significance; full KB in system prompt | 0.962* | 0.886* | 0.295* |
| run3-5 | zscan minima fix; line-anchored z; [O II] doublet; OH-zone policy | 0.94-0.96* | 0.85-0.89* | 0.28-0.39* |
| run6-7 | fixed per-class redrock candidates bug | 0.95* | 0.90-0.94* | 0.31-0.39* |
| run8-14 | **removed VI-label leak** (honest numbers), sig/ivar dual measure, [Ne V] check, coherent-pattern grading | 0.93-0.94 | 0.85-0.91 | 0.35-0.46 |

\* runs before the leak fix are inflated (expert VI labels were present in the
digest). Only run8+ are trustworthy.

## Known limitations

- **Type-2 AGN as Galaxy**: objects like QSO/4 (strong [O II]/[O III] plus
  [Ne V]) are graded GALAXY by the LLM because the emission-line galaxy pattern
  dominates; the deterministic [Ne V] check now flags them in the audit trail
  ("may be Type 2 QSO"), but the type itself is not auto-flipped to avoid
  false positives.
- **Broad-line QSO z scatter**: for Type 1 QSOs the line-anchored z can differ
  from the expert's chosen line centroid by ~1000 km/s (intrinsic to broad
  lines); e.g. QSO/10 2.319 vs expert 2.306.
- **Low-SNR spectra**: genuinely ambiguous cases (q=0/1 expert) are honestly
  reported as UNKNOWN with low quality, which the expert also effectively
  concedes (q=0/1).
- **Transient LLM failures**: ~1/80 spectra per run return unparseable/empty
  JSON even after retries; the multi-run ensemble recovers these.

## Stopping decision

The agent beats the production pipeline (redrock) on type accuracy (0.94 vs
0.88), matches it on redshift accuracy (~0.88), and — unlike redrock — produces
a fully auditable evidence trail per spectrum (which lines were checked, their
significances, which KB rules applied, and per-line redshifts). The remaining
gaps are concentrated in genuinely ambiguous spectra (very low SNR, or objects
where QSO-vs-Galaxy depends on subtle features such as [Ne V]) where expert
labels themselves carry uncertainty. Further iterations moved metrics by
±0.02–0.05 run-to-run (LLM variance) rather than improving the plateau, so
further prompt tuning has diminishing returns and risks overfitting to this
81-spectrum training set.

## How to run

```bash
conda activate langchain
export RR_TEMPLATE_DIR=/data2/wbc/redrock/py/redrock/templates
# 1. run redrock on all spectra (once)
rrdesi -i $(ls training_set/*/*.fits) -o results/redrock_all.fits -d results/redrock_all.h5 \
       --nminima 5 --ncpu 16
# 2. run the agent (repeat for ensemble; resume to finish partial runs)
python run_eval.py --out results/runX --workers 8
python run_eval.py --out results/runX --resume results/runX/verdicts.json --workers 8
# 3. baseline + ensemble
python run_baseline.py
python run_ensemble.py --runs results/runA results/runB --out results/ensemble
```
