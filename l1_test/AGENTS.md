# Visual Inspection Agent for DESI Spectra

## 1. Task

Build an LLM agent system that performs **visual inspection (VI)** of DESI spectra from scratch. For each spectrum, the agent must assess:

- the **redshift**,
- the **object type** (QSO / Galaxy),
- the spectral **characteristics** (emission/absorption lines, continuum shape, breaks, forests, etc.).

The agent's results must be **auditable** — for each spectrum, it should be possible to trace how the conclusion was reached.

## 2. What is visual inspection?

In spectroscopic surveys such as DESI, every spectrum needs a redshift and a classification before downstream science can use it. In production, this is done automatically: a template-fitting tool such as `redrock` produces a best-fit redshift and classification, and subtype-specific machine-learning pipelines (e.g. `MgII` and `QuasarNET` for QSOs) cross-check the results.

However, to evaluate and improve the spectroscopic pipeline, to quantitatively assess target selection, and to identify physically interesting subsets of the target population, a small subset of spectra is given to human experts who examine them with their own eyes. This is called **visual inspection**.

In practice, an expert is shown a spectrum together with the `redrock` best-fit model (a redshift plus a classification), and judges whether the model matches the data. Experts typically:

- apply Gaussian smoothing to the spectrum (σ ≈ 15 is generally a good choice) to bring out broad features,
- switch to σ = 0 to inspect fine details such as unresolved doublets,
- adjust the redshift (coarsely or finely) if the best fit looks slightly off.

The key evidence is the **emission/absorption lines**: if the lines expected at the model's redshift line up with real features in the spectrum, the fit is probably correct.

## 3. Grading standard

The official DESI VI grading scheme is as follows.

Classification flags (VI_QUALITY):

- `[4]` Confident classification, two or more secure features
- `[3]` Probable classification, at least one secure feature + continuum; or many weak features
- `[2]` Possible classification, one strong emission feature, but not sure what it is
- `[1]` Unlikely classification, one or some unidentified features
- `[0]` Nothing there — no signal

Optional indications:

- `[Bad redshift fit (R)]` — misestimation of redshift by the pipeline fitter
- `[Bad spectype fit (C)]` — misidentification of spectral category by the pipeline fitter, e.g. star vs QSO
- `[Bad spectrum (S)]` — bad spectrum, e.g. cosmic / skyline subtraction residuals

Your agent should grade each spectrum with a similar scheme. If you consider this standard flawed or unreasonable in some respect, you may design your own grading scheme instead — but you must define it explicitly, apply it consistently, and justify the change.

## 4. What to do

Develop an agent that does this job automatically. For each input spectrum, the agent should produce a VI-style result (redshift, type, quality/confidence, and supporting evidence) in an auditable form.

## 5. Environment and software

- A conda environment named `langchain` is available, with `langchain` and `redrock` installed.
- `redrock` can be run via its command-line entry point:

```shell
rrdesi
```

- See `/data2/wbc/redrock/README.rst` for details on `redrock` usage.
- You may need `redrock-archetypes`. The related path is `/data2/wbc/redrock/redrock-archetypes/`.

## 6. Data

`/data2/wbc/slo/l1_test/training_set/` contains example spectra of four target types: `BGS`, `ELG`, `LRG`, `QSO`.

Each type directory contains:

- `{index}.fits` — the spectra, named by the `index` column of the CSV,
- `{type}_tr_set.csv` — metadata for these spectra, including expert VI results (`VI_SPECTYPE`, `VI_Z`, `VI_QUALITY`) that can serve as ground truth.

Use these data to develop and evaluate your agent.

## 7. Configuration

`.env_example` lists the environment variables used for LLM access. `API_KEY`, `BASE_URL` and `LLM_MODEL` are already set in the `.env` file. You may add more parameters as needed.

## 8. Working mode: autonomous iteration

Work fully autonomously. Do NOT ask the user for requirements, opinions, confirmations, or any other input at any point — make all design decisions yourself and act on them.

Iterate on your agent until you are satisfied with it:

1. Define quantitative evaluation metrics yourself (e.g. agreement with the expert VI results in `{type}_tr_set.csv` on redshift, type, and quality).
2. Run the agent on the training set, evaluate against the metrics, and analyze the failures.
3. Improve the agent, then re-evaluate. Record what changed and how the metrics moved, so the iteration history is auditable.
4. Decide yourself when to stop — for example when you judge the results satisfactory, when the metrics converge, or when further changes no longer bring meaningful improvement — and state your reasoning for stopping.
