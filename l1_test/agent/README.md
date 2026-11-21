# LLM Visual-Inspection Agent for DESI Spectra

This directory contains an LLM-based visual-inspection (VI) agent for DESI
spectra, developed for the task in `../AGENTS.md`. It inspects each spectrum,
produces a VI-style result (redshift, spectral type, quality grade, and
supporting evidence), and keeps every decision auditable.

## Architecture

The agent is a two-stage pipeline:

1. **Deterministic feature extraction** (`features.py`, no ML / no randomness):

   - Loads a DESI spectrum FITS (B+R+Z cameras), cleans bad pixels, estimates
     a robust continuum, and Gaussian-smooths the flux.
   - Measures integrated line S/N (narrow +-3 A and broad +-40 A rest windows)
     for a 22-line rest-frame catalog (Ly-a, SiIV, CIV, HeII, CIII], MgII,
     [NeV], [OII], Balmer, [OIII], [NII], H-alpha, [SII], [NeIII], Ca H&K,
     MgI, NaD) over a z grid 0-4.5 (dz=0.002).
   - Computes a weighted redshift-score curve, finds candidate redshifts, and
     ranks them by *number of secure lines* (|S/N|>=3), so multi-line
     consistent solutions beat single-line aliases.
   - Continuum diagnostics: D4000 break, power-law slope, Ly-alpha forest
     indicator, global S/N; sky-line proximity flag for every line.
2. **LLM reasoning agent** (`vi_agent.py`): a single LLM call
   (deepseek-v4-flash, temperature 0) that receives a text report with:

   - spectrum global properties and target-selection prior (informational),
   - the redrock pipeline fit (z, spectype, chi2, deltachi2, zwarn) as a
     second opinion, plus the measured line S/N **at the redrock z**,
   - ranked candidate redshifts with per-line S/N, contrast and sky flags,
   - noise/confidence context and a deterministic quality hint.
     The model applies the official DESI VI grading rubric and returns JSON:
     `{vi_z, vi_spectype, vi_quality, evidence[], flags[], reasoning}`.
     Post-hoc checks: quality is capped to match the rubric (q>=3 requires

   > =1 secure line, q=4 requires >=2) and a deterministic fallback covers
   > LLM failures.
   >

`run_agent.py` orchestrates redrock (bulk `rrdesi` run) + feature extraction
(cached) + threaded LLM calls; `evaluate.py` scores against the expert VI
ground truth; `export_results.py` writes a per-spectrum CSV.

## Grading scheme

We adopt the official DESI VI grading scheme unchanged:

- `[4]` Confident classification, two or more secure features
- `[3]` Probable classification, >=1 secure feature + continuum (or many weak features)
- `[2]` Possible classification, one strong emission feature, unsure
- `[1]` Unlikely classification, unidentified features
- `[0]` Nothing there — no signal

Optional flags: `R` bad redshift fit, `C` bad spectype fit, `S` bad spectrum.

## How to run

```shell
# 1. (optional, ~1.5 min) bulk redrock on all spectra
conda activate langchain
export RR_TEMPLATE_DIR=/data2/wbc/redrock/py/redrock/templates
rrdesi -i <all spectra> -o work/redrock_bulk/rr_all.fits --ncpu 8

# 2. extract deterministic features (cached)
python extract_all.py

# 3. run the LLM agent (needs .env API credentials)
python run_agent.py

# 4. evaluate + export
python evaluate.py
python export_results.py
```

## Results (final configuration)

81 training spectra (BGS/ELG/LRG/QSO). Redshift agreement uses
|dz|/(1+z) < 0.003 (DESI-style tolerance); 0.005 also reported.

| metric                        | agent           | redrock baseline |
| ----------------------------- | --------------- | ---------------- |
| redshift accuracy             | 85.2%           | 88.9%            |
| type accuracy (QSO/GALAXY)    | 93.8%           | 87.7%            |
| VI correct (z AND type)       | **85.2%** | 80.3%            |
| VI correct (dz tol 0.005)     | 87.7%           | —               |
| quality-aware VI correct      | 92.6%           | —               |
| quality MAE (0-4 scale)       | 0.37            | —               |
| quality Spearman rho          | 0.70            | —               |
| quality within +-1            | 92.6%           | —               |
| quality secure(>=3) agreement | 93.8%           | —               |

Science-useful subset (expert q>=3, n=71): z accuracy 95.8%, type 97.2%,
**VI correct 94.4%** (97.2% at 0.005 tol), quality within +-1 97.2%.

Per type (VI correct): BGS 95%, LRG 95%, ELG 75%, QSO 76% (QSO 86% at
0.005 tol).

Remaining failure classes (all documented per-spectrum in `work/audit/`):

- ~9 noise-dominated spectra with expert q<=2; the agent usually grades
  these q<=2 as well, and z/type on them is inherently unreliable;
- 2 broad-line QSOs with dz/(1+z) ~ 0.003-0.004 (measurement scatter;
  correct at 0.005 tolerance);
- a few genuinely degenerate cases (single-line [OII]-vs-CIII]/H-gamma
  aliases; low-S/N high-contrast lines) where the expert call depends on
  target-selection priors we pass only as weak hints.

## Auditability

For every spectrum, `work/audit/{TYPE}_{index}.txt` contains:

- the full deterministic feature report given to the LLM,
- the LLM's raw chain-of-thought (reasoning trace),
- the final JSON result with evidence list.

`work/vi_results_table.csv` summarizes all 81 spectra side by side with the
expert VI. `work/vi_results.json` holds the machine-readable output.
