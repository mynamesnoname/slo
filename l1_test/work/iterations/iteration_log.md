# Agent iteration log (complete)

## Iteration 1 (baseline)
- Feature extractor + LLM (deepseek-v4-flash) + redrock second opinion.
- Bug: reasoning model exhausted the 1200-token budget on chain-of-thought;
  most calls returned "unparseable". Metrics meaningless.

## Iteration 2 (fixes: max_tokens 8000; line_sn_at_z boolean-index bug; peak
## finder suppressing a stronger nearby peak; absorption lines dropped by an
## unsigned comparison; redrock-z line table added to the report)
- VI correct 76.5%; type 85.2%; z 81.5%; q-MAE 0.33.
- Failures: low-z galaxy lines aliased to high-z QSO lines (e.g. H-alpha at
  z~0.06 read as "MgII at z=1.49"), noise spectra over-claimed as confident
  QSOs, 2 unparseable.

## Iteration 3 (max_tokens 32000; catalog +NeIII/[SII]; noise context +
## deterministic quality hint; prompt alias warnings + noise calibration;
## post-hoc quality cap; deterministic fallback)
- VI correct 81.5%; type 91.4%; q-Spearman 0.71; 0 fallbacks.

## Iteration 4 (target-selection prior passed to the LLM; AGN-at-low-z
## guidance; secondary 0.005 tolerance metric)
- VI correct 84.0%; type 93.8%; quality-aware 91.4%.

## Iteration 5 (aggressive sky-line flag: tol 5A + "do NOT use as evidence")
- REGRESSION: VI correct 77.8% - rejected real lines coinciding with OH lines
  (ELG/104 q4->q0). Reverted the rejection semantics.

## Iteration 6 (sky tol 2.5A, strength-dependent caution; temperature 0)
- VI correct 81.5%; type 92.6%; quality-aware 91.4%; secure-quality agreement
  96.3%; q-MAE 0.30.

## Iteration 7 (line contrast metric: peak/continuum, helps lines in noisy
## pixels; shown in candidate lists and redrock-z table)
- BEST: VI correct 85.2%; type 93.8%; z 87.7%; quality-aware 92.6%;
  science-useful subset (expert q>=3, n=71): z 95.8%, type 97.2%,
  VI correct 94.4% (97.2% at 0.005 tol), quality-within-1 97.2%.

## Iteration 8 (trial: "CIV => AGN => QSO" hard rule)
- REGRESSION: VI correct 81.5% (LRG/250 flipped to a wrong Ly-a reading).
  Reverted; final configuration = iteration 7.

## Final (re-run of iteration 7 config; deterministic within +-1 spectrum)
- VI correct 84.0% (85.2% in best run); type 93.8%; z 87.7%;
  quality-aware 92.6%; q-MAE 0.34; q-Spearman 0.77; quality-within-1 95.1%;
  secure-quality agreement 95.1%.
- Science-useful subset (expert q>=3, n=71): z 97.2%, type 97.2%,
  VI correct 94.4% (97.2% at 0.005 tol), quality-within-1 98.6%.
- redrock baseline: VI correct 80.3%, type 87.7%, z 88.9%.

## Remaining failure classes (all audited in work/audit/)
1. ~9 noise-dominated spectra (expert q<=2): agent also grades them q<=2
   mostly; z/type on them is inherently unreliable.
2. 2 broad-line QSOs with dz/(1+z) ~0.003-0.004 (line-width scatter; pass at
   0.005 tolerance).
3. A few degenerate single-line aliases (e.g. [OII]-vs-H-gamma at the same
   observed wavelength; low-S/N high-contrast lines) where the expert call
   relies on target-selection priors.

## Stopping rationale
Metrics converged across iterations 6-8 (VI correct 81.5-85.2%, science-useful
94.4%, quality agreement ~95%); remaining failures are either unrecoverable
noise spectra or genuine expert-judgment calls. Further prompt changes moved
metrics +-3% without systematic gains (CIV rule regressed; sky flag regressed).
Final configuration: temperature 0, contrast-aware line tables, soft sky-line
flags, target prior, redrock-z verification, rubric-based quality cap.
