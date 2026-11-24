# Experiment policy and limitations

Contributor: Tejaswi Sapkota. Original paper: Nonaka et al., Boosting
Phonocardiogram Classification Performance with Function Generated Data,
https://proceedings.mlr.press/v297/nonaka26a.html (ML4H 2025; PMLR 297).

## Question

Compare Real, Syn, Syn-to-Real, and a new jointly trained Mixed condition for
AS classification with existing team PyHealth task/model interfaces unchanged.
This is a partial reproduction plus extension, not an exact published-code replay.

## Data and partitions

Real task is Michael's AS/exclude task: target AS is positive (including AS plus
another disease); Normal is negative; other-disease-only and N/A diagnoses are
excluded. The pinned source has 58 eligible patients / 464 recordings. Five frozen
patient-stratified assignments have 36 training, 10 validation, 12 test patients
(288 / 80 / 96 recordings). Test identities are constant. Processors fit only through
the training part of split-aware set_task. Smoke keeps two patients per class
within each train/validation partition, not across partition boundaries.

Synthetic Normal (0) / AS (1) recordings are generated at 10 seconds / 8 kHz. Full uses 5,000
train and 1,000 validation per class. Reuse one population: run 0 keeps original
membership; runs 1–4 reshuffle within class with 42–45; training seeds 0–4 are
actually applied by the runner. Across-run overlap is expected. Within-run
partitions are disjoint. Pilot and smoke generate smaller separate pools.

## Training and selection

All modes share code. ResNet18-style layers [2, 2, 2, 2], width 64, normalize=True,
Adam lr 2e-5, binary cross-entropy. Real pos_weight is Nnegative/Npositive from
that run's selected training recordings only. Synthetic weight 1. Full defaults
batch 512, cap 500 per stage, validate every 5 epochs, stop after 5 unsuccessful
validation evaluations. Save lowest-validation-loss checkpoint; no test selection.
All training batches, including a small last batch, are retained.

Syn selects ONLY on synthetic validation. Real, Syn-to-Real, Mixed select on
real validation. Syn-to-Real loads the corresponding completed Syn best weights
including classifier, retains real weighting, and starts a fresh Adam optimizer.
Full epoch caps are fixed in saved configs; changing caps on resume is rejected.
Validation also runs at the fixed final cap. Operational early interruption does
not create an extra validation event or modify the schedule.

Mixed starts from random weights. Each step contains equal-size real/synthetic
halves, with half weighted real BCE plus half unweighted synthetic BCE. One
epoch passes real training once; synthetic samples are independently shuffled
without replacement within cycles and repeated when needed. A partial final
real batch gets the same-sized synthetic half. Mixed therefore usually has more
optimizer updates than Real at the same nominal batch/epochs. Record domain
exposures, steps and walltime; never claim compute matching from epochs alone.
Shared Syn pretraining cost is recorded separately from fine-tuning, not counted
twice as actual executed work. No synthetic pretraining is charged to Mixed.

## Primary deviations to report

- Team task is AS versus Normal; pinned authors' loader uses AS versus all
  non-AS recordings. Cohort and split counts differ from the paper/source.
- Team processing remains mono 8 kHz, first 10 s / zero-pad and model-side z-score.
  Reference classifier config uses 1 kHz / 5 s, normalizes before random training
  crop/center evaluation crop. Synthetic generation rate is a separate setting.
- NO augmentation in this package. The paper uses masking, shifting, polarity
  flipping, breathing, stretching and time-varying scaling. Changing these is
  deliberately left to the task/model owners, not silently patched here.
- The model backbone/initialization and generator differ as documented in
  GENERATOR_README.md and Franko's model documentation.
- One fixed synthetic pool across resplits, rather than independently generated
  populations. Pilot has one run and reduced synthetic data; no final inference.
- Full batch reductions/shorter caps must be reported. A reduced batch does not
  emulate 512 exactly, particularly with BatchNorm. No implicit OOM adjustment.

## Test-set rules and reporting

No held-out performance in smoke/pilot or training. Explicit freeze/evaluate
requires all five completed full runs and unchanged selected artifact hashes.
Compare selected models on identical real-test patients. Report mean and sample
SD (ddof=1) for AUROC, average precision, F1, sensitivity, specificity, PPV, NPV.
Threshold 0.5; undefined PPV/NPV are 0. Report denominators/confusion counts.
Paper Table 17 ResNet18 AS AUROC: Real .7165 ± .0351; Syn .5000 ± .0000;
Syn-to-Real .7857 ± .0074. These are reference values, not pass/fail thresholds.
Mixed has no paper baseline. Do not tune after test evaluation or interpret
tiny-pilot validation differences as confirmed gains.

## Attribution

Generator reference: seitalab/SynPCG revision
1a5f600ce2a0a9e43514e20582da67081159a14a. Preserve attribution. Reference reuse
terms have not been declared resolved; this package adds no license clearance
claim. Michael's task and Franko's model remain owned/maintained by their authors.
