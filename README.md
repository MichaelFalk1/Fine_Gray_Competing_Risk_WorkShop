# Competing risks: who is still in the risk set?

A beginner-friendly, zero-cost Python workshop using synthetic first-stroke,
death-before-stroke and censoring data. **Start with [workshop.ipynb](workshop.ipynb).**
One notebook contains every teaching helper; no local project imports or dataset downloads.

**Available now:** simulation, CIFs, cause-specific Cox, weighted Fine–Gray risk-set
walkthroughs, reference-checked full-risk linear Fine–Gray regression, NCC/case-cohort
sampling visuals and cost experiments, an optional tiny TensorFlow exercise, and
instructor-requested conceptual NCC/CC neural score prototypes.
**Not yet validated:** TensorFlow execution in a compatible runtime, sampled neural
Fine–Gray fitting, and an actual GitHub → free Colab Run All.

## Start in free Google Colab

The repository owner/default branch have not been supplied, so there is no fabricated
“Open in Colab” badge. After you publish the repository, open
[Google Colab](https://colab.research.google.com/), choose **File → Open notebook → GitHub**,
and paste your public repository URL; select `workshop.ipynb`.
Before publication, **Upload notebook** also accepts this same file.

Once the public location is known, the badge target will have the form
`https://colab.research.google.com/github/OWNER/REPOSITORY/blob/BRANCH/workshop.ipynb`.
This project has not configured a remote, created a repository, committed, or pushed.

1. Choose a CPU runtime (no paid tier or GPU required).
2. Run **0. Setup & environment checks**. If needed, enable the explicit missing-package
   installation cell. Packages installed on your laptop are not installed in Colab.
3. Choose **Runtime → Run all**. Use **Shift+Enter** for individual cells.
4. Change parameters near the top, then restart and Run All to reset the experiment.
5. For a shorter session, set `FAST_MODE=True`. Keep `TRAIN_NEURAL=True` for the core path.

Colab needs internet and a normal free account; installations are per remote session.
Free resources can disconnect and are not guaranteed. No credentials, Drive mounts,
external datasets or API keys are used. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Run locally in VS Code / Jupyter

Use a Python environment with the packages in `requirements.txt`. Core execution was
tested on Windows/Python 3.14.3. The optional neural path needs a TensorFlow-compatible
Python version; current TensorFlow package metadata lists Python 3.10–3.13.
Check [TensorFlow installation instructions](https://www.tensorflow.org/install/pip)
before selecting a runtime. No R/rpy2 or PyTorch dependency is needed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open `workshop.ipynb` in VS Code, select `.venv` as the notebook kernel, and **Run All**.
Jupyter users can select the same environment through their existing Jupyter interface.
The development environment created here uses `--system-site-packages` to reuse existing
scientific packages; the commands above create a clean environment for participants.

The notebook checks imports before offering installation. Do not blindly upgrade a
working runtime. TensorFlow is optional and can be a large download; the commented
installation cell never runs automatically. Use only `tf.keras`, not mixed Keras backends.

## Workshop route

| Module | Exercise |
|---|---|
| 0 | Environment, seed, cohort size, rates, sampling budget, neural on/off |
| 1 | Eight features, first-event timelines, actual counts, Aalen–Johansen vs naive 1−KM |
| 2 | Ten-person risk-set timeline/table/grid; censoring retention; cause-specific Cox |
| 3 | NCC event-time draws; fixed baseline case-cohort; hidden-assay mask; unique counts |
| 4 | Analytical Fine–Gray benchmark, full-risk linear fit/CIFs, optional 361-parameter NN |
| 5–6 | Budget curves, repeated selection cost, honest status table, learner questions |

Default: 800 people; 4 continuous + 4 binary inputs; seed 2026; 2 NCC controls per case;
25% baseline case-cohort fraction; 20 optional neural epochs. Fast mode: 300 people,
5 epochs. Everything uses CPU. The network is exactly **8 → 8 → 16 → 8 → 1**, hidden
`tanh` layers and a linear score output, **361 trainable parameters including biases**.

No slides or sophisticated web UI are part of this repository.

## Implementation notes / verification status

### Completed numerical evidence

Prototype update (2026-11-09): **18 tests passed, 2 TensorFlow tests skipped**.
A fresh-kernel fast-mode Run All passed (42 cells, 14 figures), with neural paths
explicitly skipped because TensorFlow is unavailable. New tests cover exact NCC
ever-inclusion probabilities, CC probabilities, full-coverage objective recovery,
finite-difference prototype gradients, zero-support detection, and cases-only guards.
The illustrative CIF calculation passes hand-calculated baseline/link checks,
score-shift invariance, full-coverage agreement with the reference-checked linear
FG predictions, and bounds/monotonicity checks. These are numerical checks, not
validation of sampled population CIF estimation.
The optional TensorFlow test also checks hidden-feature isolation and CC fraction-1
training recovery, but those checks remain unexecuted here.

- Deterministic synthetic data and analytical conditional CIF checks.
- Hand-calculated Cox/FG membership, no-censoring retention, censored retention,
  reverse-KM comparison with lifelines, and the 0.72/0.90 example.
- NCC eligibility, own-case exclusion, no within-draw duplicates, repeated selections,
  sparse risk sets and zero controls. Case-cohort fixed baseline draws, overlap,
  zero/full fractions, and masked unmeasured covariates.
- A four-person analytical FG optimum: `beta=log(1/sqrt(2))`; exact baseline jumps
  and patient CIFs; zero-score loss; finite-difference score gradient; score-shift invariance.
- Independent **pycmprsk 1.0.0** comparison on a 160-person censored synthetic cohort
  (seed 401, eight covariates): coefficient and patient CIF agreement within `2e-5`,
  baseline hazard jumps within `2e-6`, event times within `1e-12`.
- Fresh local Jupyter-kernel Run All passes. The initial default run took approximately
  15 seconds, with neural training off. Runtime varies by computer.
- Final local suite: **14 passed, 1 skipped** (TensorFlow unavailable). The default
  notebook produces 14 figures. Fast mode with zero sampling and neural requested also
  runs successfully, explicitly skipping the unavailable TensorFlow module.

The custom implementation is dense and educational, with distinct event times, independent
marginal censoring and time-fixed covariates. It rejects event ties and poor censoring
support instead of silently clipping weights. Administrative censoring ties are supported.
No variance estimator, p-values, clinical validation, delayed entry or time-varying
covariates are provided. Hazard ratios are not probability ratios.

The full-risk objective follows the weighted construction in
[Kawaguchi et al. (2021), equations 2–3](https://journal.r-project.org/articles/RJ-2021-010/).
Weights use left limits at distinct event times. Baseline/CIF prediction is checked
separately. The generating cause-specific model need not obey proportional subdistribution
hazards, so its coefficients must not be treated as true FG coefficients.

### Why pycmprsk is only a development reference

[pycmprsk 1.0.0](https://pypi.org/project/pycmprsk/) requires Python >=3.13 and uses
Numba; its GPL-3.0-or-later license and API were checked. We invoke the installed package
in an optional test and do not copy its implementation. Its stated R `cmprsk` parity
tests provide useful provenance, supplemented by our independent analytical benchmark.
It installed successfully in the local project environment, but is **not a required
attendee dependency**: requiring it would exclude older Colab runtimes. It was not
installation-tested in free Colab. The observed dependency-install batch took a few
minutes and included plotting/testing packages; no isolated install-time claim is made.

### Neural execution gate

The notebook includes the exact architecture, reproducible initial weights, full-risk
`GradientTape` training, stable log-sum-exp, train-only preprocessing/censoring, baseline
estimation, loss curves and held-out CIF comparison to synthetic truth. Before training,
it checks TensorFlow loss and gradients against the NumPy implementation on censored
and uncensored examples; finite-gradient and probability checks remain active.

**TensorFlow/Keras are absent in this machine's Python 3.14 environment.** Consequently
the neural test is skipped and no neural loss curve, runtime or accuracy result is claimed.
The optional code is implemented but runtime validation remains open. Next step: use a
compatible TensorFlow environment, run the tests and `TRAIN_NEURAL=True`, and record
versions and outputs. Passing runtime checks alone does not establish neural inference
theory or calibration. The output bias is unidentifiable in the partial likelihood and
is absorbed by the baseline; sigmoid(score) is never used as a CIF.

### Sampled neural prototypes; statistical estimator validation still blocked

| Mode | Current state | Concrete next work |
|---|---|---|
| Full-risk linear FG | Reference-checked educational fit | Broader simulation checks if extending scope |
| Full-risk neural FG | Implemented, opt-in; runtime verification pending | Execute on compatible TensorFlow CPU runtime |
| NCC neural prototype | Implemented conceptual score fit; not a validated FG estimator | Derive design-specific event, retention and inclusion contributions; benchmark before neural adaptation |
| Case-cohort neural prototype | Implemented conceptual score fit; not a validated FG estimator | Implement and validate a published case-cohort subdistribution score and baseline; justify neural substitution |

[Wolkewitz et al. (2014)](https://doi.org/10.1097/EDE.0000000000000029) describe an NCC
subdistribution sampling method. Its existence does not validate our ordinary event-free
NCC sampler as an FG fitting design. Full equations and the sampling/retention treatment
must be reconciled before implementation. [Kim et al. (2020)](https://doi.org/10.1111/sjos.12461)
provide a case-cohort proportional subdistribution method; a generic Cox case-cohort
loss or universal `1/pi` substitution is not an implementation of that method.
Neural adaptation and design-specific uncertainty remain unverified for both modes.

The instructor has explicitly requested didactic approximations. Set
`TRAIN_SAMPLED_PROTOTYPES=True` together with `TRAIN_NEURAL=True` to run them.
They substitute the same 361-parameter neural score into the conceptual denominator
`sum(selected q_i(t) * exp(score_i) / pi_i)` and minimize mean log-denominator
minus case score. This is **PROTOTYPE / NOT VALIDATED — NOT A STATISTICAL
FINE–GRAY ESTIMATOR**, not an implementation of either published sampled estimator.

For CC, cases have inclusion probability 1 and noncases have the actual fixed-size
baseline fraction. For ordinary NCC, noncase probability of ever being selected is
`1 - product(1 - p_ik)` across independent case-time draws, conditional on observed
training outcomes; cases have probability 1. Early competing deaths can have zero
inclusion probability: the notebook reports their count and labels the resulting
restricted denominator as unable to represent the full risk set. Even complete
ordinary NCC event-time coverage cannot recover people with zero inclusion support.
No clipping or invented inclusion probabilities repair that gap.

Each fit gets only its measured training features and fits its own scaler. Full
training outcomes supply censoring weights; holdout features are used only for
score evaluation. Same architecture, initial weights, Adam rate, and epochs are
used. CC fraction 1 recovers the full-risk construction; NCC complete coverage
recovers it only when every contributing person has inclusion support.

The table reports actual training-loop wall time on CPU and held-out centered-score
MAE versus the full-risk neural score. This measures score disagreement, not truth
or probability accuracy. An instructor-requested illustrative CIF panel uses each prototype's own baseline
jumps (reciprocal weighted training denominators) and the subdistribution link.
One CIF panel compares the same evaluation group across full-risk neural FG, NCC,
CC and simulation truth. Its legend shows controls per case and the baseline subcohort
fraction; curves average individual predictions over that common group. This numerically checked plug-in calculation does not
validate sampled population probabilities; the NCC coverage gap remains. Sampled
CIF MAE stays explicitly unavailable rather than implying validated prediction. Loss levels can differ and are not a method ranking.
The new TensorFlow paths still require execution in a compatible environment;
Python 3.14 in this workspace has no TensorFlow. No sampled runtime is fabricated.

All source-cohort outcomes are observed. NCC measures all stroke cases plus selected
controls; CC measures one baseline subcohort plus all stroke cases. Death/censoring
subjects outside the selected measured set have missing features in the displayed
analysis view. No sampled fitting function reads the complete synthetic feature table.
Sampling IPW and censoring IPCW are separate mechanisms. At q=0/m=0 we show costs only.

The split is 75% training / 25% untouched holdout. The scaler and censoring model are
training-only. Held-out error compares predicted CIFs to known synthetic conditional
truth over a stated time grid; it is not an observed-outcome Brier score. No holdout
tuning, fake confidence bands or fabricated sampled predictions are included.

## Tested version matrix (2026-11-06)

| Component | Local executed version | Free Colab |
|---|---|---|
| Python | 3.14.3, Windows CPU | Not yet tested |
| NumPy | 2.4.6 | Not yet tested |
| pandas | 2.3.3 | Not yet tested |
| Matplotlib | 3.11.2 | Not yet tested |
| SciPy | 1.17.1 | Not yet tested |
| scikit-learn | 1.8.0 | Not yet tested |
| lifelines | 0.30.3 | Not yet tested |
| TensorFlow / Keras | Not installed; optional checks skipped | Not yet tested |
| pycmprsk | 1.0.0, development reference only | Not required |
| nbformat / nbclient / pytest | 5.11.1 / 0.11.0 / 9.2.0 | Not required for attendee Run All |

Requirements are not speculatively pinned across platforms. In particular, lifelines
selected pandas 2.3.3 inside the project environment without changing system pandas.

## Tests and conference rehearsal

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_workshop.py -q
.\.venv\Scripts\python.exe tests/execute_notebook.py
.\.venv\Scripts\python.exe tests/execute_notebook.py --fast --neural --zero-sampling
```

Tests read definitions directly from the canonical notebook. `pycmprsk` and TensorFlow
tests skip explicitly when absent. To reproduce the optional external reference check,
install `pycmprsk==1.0.0` in a compatible development environment. The Run All script
uses a fresh kernel and saves its executed copy/PNG previews only under ignored
`.validation/`; this is test output, not a second participant source.

Saved outputs in `workshop.ipynb` are instructor fallback examples from the printed
default parameters. After edits, rerun or clear outputs to avoid stale figures.

- [x] M0: self-contained skeleton, setup, local/Colab instructions.
- [x] M1: simulation, timelines, counts, Aalen–Johansen/KM.
- [x] M2: risk sets, censoring weights, Cox and reference-checked linear FG.
- [x] M3: NCC/CC selection, diagrams, cost experiments and tests.
- [ ] M4: neural code complete; compatible-runtime execution still required.
- [ ] M5: conceptual NCC/CC score prototypes implemented at instructor request;
  TensorFlow execution and design-specific statistical validation remain pending.
- [x] M6 local core: learner prompts, fast mode, tests, figures and troubleshooting.
- [ ] Instructor: publish public repository and add real Colab badge.
- [ ] Instructor: fresh free Colab CPU Run All from GitHub, including visual review.
- [ ] Instructor: neural CPU rehearsal and tested TensorFlow/Keras versions.

## Troubleshooting

- **Missing imports:** run setup, opt into the missing-core installation cell, restart
  if packages changed, then Run All. Broken installed packages need diagnosis, not
  repeated blind upgrades.
- **TensorFlow unavailable:** leave training off; all earlier modules still run.
  Select a compatible Python version before a deliberate install.
- **Few strokes / constant inputs:** restore defaults or increase N; model cells skip
  sparse configurations with a message.
- **Poor censoring support / ties:** shorten follow-up or review the data-generating
  settings. Do not invent a numerical workaround that changes the estimator.
- **Slow runtime:** use fast mode; keep neural training off, reduce epochs/repetitions.
- **Stale plots or undefined names:** restart and Run All. The notebook has no private
  files, local imports or hidden required state.
- **Package cache warnings in restricted shells:** the execution helper uses project-local
  cache/history directories. A Windows ZeroMQ event-loop warning may appear without
  affecting successful notebook execution.

This is a synthetic educational exercise, not a medical prediction system.
