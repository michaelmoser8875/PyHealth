# Synthetic PCG experiments

Tejaswi Sapkota's Group 3 contribution: generate Normal/AS heart sounds and run
Real, Syn, Syn-to-Real and Mixed using the team's existing PyHealth interfaces.
One implementation has three configuration presets. No training or downloads
run automatically on import. No generated data or checkpoints belong in Git.

This folder does not replace Michael's BMDHSBinaryClassification or Franko's
ResNet1D. It requires the integrated team PyHealth checkout containing both.
Full mode enables a descriptive comparison with the paper; it is NOT an exact
reproduction. See EXPERIMENT_POLICY.md for consequential differences.

## Files

| File | Purpose |
| --- | --- |
| synthetic_pcg.py | Unchanged, previously checked Normal/AS generator |
| synthetic_manifest.py, synthetic_splits.py | Unchanged manifest and five-run helpers |
| generate_dataset_cli.py | Original standalone generator CLI |
| prepare_data.py | Download/reuse BMD-HS, generate fresh audio, create plans |
| real_split_template.json | Exact previously checked patient memberships, labels and relative filenames; no audio |
| workflow.py | Mode presets, policy and provenance helpers |
| experiment_data.py | Existing team task and split-aware loading; deterministic sampling |
| run_experiments.py | All four training protocols, all three modes and resume |
| inspect_results.py | Verify own saved artifacts and print validation-only results |
| evaluate_results.py | Separate full-only freeze and held-out evaluation |
| test_*.py | Existing generator tests and new workflow boundary tests |
| synthetic_pcg_examples.ipynb | Original notebook with embedded generator figures |
| SynPCG_Presentation_Demo.ipynb | Presentation figures, synthetic audio and optional completed-run curves |
| GENERATOR_README.md | Generator behavior, provenance and differences |
| COPIED_SOURCE_HASHES.json | Hashes of the eight contribution files at initial copy |
| LOCAL_ENVIRONMENT_REFERENCE.json | Observed local package versions, not a CUDA lockfile |

The original generator notebook retains its original cell sources; saved execution
outputs/metadata can change after rerunning it. `COPIED_SOURCE_HASHES.json` records
the initial copies; `PACKAGE_SHA256.json` records the current handoff files.

## Teammate CUDA quick-start

**Prerequisites:** Git/Conda are installed, NVIDIA drivers work, and the checked-out
team branch already contains Michael's task, Franko's model and this folder.
The package has not yet been pushed from Tejaswi's local checkout; this block is
for the published, integrated branch after review. Do not assume an old copy of
the shared branch contains every component. No automatic branch merge is done.

First install a compatible CUDA-enabled PyTorch/torchvision/torchaudio set using
the [official selector](https://pytorch.org/get-started/locally/) and match
TorchCodec to that PyTorch version. The exact wheel-index command depends on the
GPU driver and available builds; there is no single verified CUDA command for
an unknown machine. Use Python 3.12 or 3.13. The commands below assume that
environment is activated. Replace only the repository location on the first line.

```bash
PCG_REPO="/path/to/integrated/PyHealth"
PCG_CODE="$PCG_REPO/CS Project /Github final"
PCG_RUNS="$HOME/DL4H-runs"

cd "$PCG_REPO"
python -m pip install -e .
cd "$PCG_CODE"
python -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg
python -m pip check
python -c "import torch; from pyhealth.models import ResNet1D; from pyhealth.tasks import BMDHSBinaryClassification; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
python -m unittest -v test_synthetic_pcg test_synthetic_manifest test_synthetic_splits test_experiment_workflow

python prepare_data.py --mode smoke --workspace "$PCG_RUNS/cuda-smoke-data"
python run_experiments.py --mode smoke --workspace "$PCG_RUNS/cuda-smoke-data" --output "$PCG_RUNS/cuda-smoke-results" --device cuda:0
python inspect_results.py --output "$PCG_RUNS/cuda-smoke-results"
```

**Stop if any command fails.** Continue only after the CUDA smoke run and artifact
inspection pass. All workspace/output paths must be new for their first use.
Record the successful environment, GPU model and source revision:

```bash
python -m pip freeze > "$PCG_RUNS/cuda-environment.txt"
git -C "$PCG_REPO" rev-parse HEAD > "$PCG_RUNS/team-revision.txt"
nvidia-smi > "$PCG_RUNS/nvidia-smi.txt"
```

Optional one-hour pilot, reusing the downloaded real data:

```bash
python prepare_data.py --mode pilot --workspace "$PCG_RUNS/cuda-pilot-data" --existing-real "$PCG_RUNS/cuda-smoke-data/real"
python run_experiments.py --mode pilot --workspace "$PCG_RUNS/cuda-pilot-data" --output "$PCG_RUNS/cuda-pilot-results" --device cuda:0
python inspect_results.py --output "$PCG_RUNS/cuda-pilot-results"
```

For full experiments, first choose a feasible batch size and caps from the CUDA
resource check. The example below **explicitly reduces the paper's batch 512 to
8** and retains the 500-epoch caps. This is a resource example, not a CUDA-verified
recommendation or a promise of runtime. Change `PCG_BATCH` before the run if the
agreed GPU check supports another even batch size. Record the deviation.

```bash
PCG_BATCH=8
python prepare_data.py --mode full --workspace "$PCG_RUNS/full-data" --existing-real "$PCG_RUNS/cuda-smoke-data/real"
python run_experiments.py --mode full --workspace "$PCG_RUNS/full-data" --output "$PCG_RUNS/full-results" --device cuda:0 --batch-size "$PCG_BATCH"
python inspect_results.py --output "$PCG_RUNS/full-results"
```

To resume, repeat the exact training command with `--resume`. Do not change
Python source files, mode, dataset plans, batch size or epoch caps mid-run.
After all five full runs are complete and settings are frozen, use Section 8's
explicit held-out evaluation commands. Smoke and pilot never evaluate that set.

## 1. Install and verify the team checkout

Use Python 3.12 or 3.13 in a separate environment. Install your integrated
PyHealth checkout, NOT an unrelated PyPI release. From that repository root:

```bash
python -m pip install -e .
```

On CUDA, first use the matching torch/torchvision/torchaudio installation command
from https://pytorch.org/get-started/locally/ for your NVIDIA machine. Then
install the integrated checkout and this folder's requirements. Keep a matching
TorchCodec version; consult https://github.com/meta-pytorch/torchcodec#installing-torchcodec.
The Mac reference environment is not proof that a CUDA wheel set works.

Enter THIS folder (quote its path if it contains spaces):

```bash
python -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg
python -m pip check
python -c "from pyhealth.models import ResNet1D; from pyhealth.tasks import BMDHSBinaryClassification; print('Team interfaces available')"
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

If ResNet1D cannot be imported, the team checkout does not include Franko's
model branch. Arrange its integration with the team before proceeding. This
package never merges branches or modifies those files.

## 2. Run isolated tests first

From this folder:

```bash
python -m unittest -v test_synthetic_pcg test_synthetic_manifest test_synthetic_splits test_experiment_workflow
```

Expected: 50 tests, no skips in a complete environment. Three original tests
need PyHealth/audio integration. New workflow tests do not train a model,
download real data, or read real patient recordings. They check mode settings,
cohort leakage guards, Mixed sampling/remainders, checkpoint corruption guards,
and metrics. Training and CUDA runtime are verified by smoke mode, not these
unit tests. Also run the team's task/model tests from its repository root:

```bash
python -m unittest -v tests.core.test_bmd_hs_binary_classification tests.core.test_resnet1d
```

## 3. Prepare a mode-specific workspace from scratch

Keep the workspace OUTSIDE the repository. Set a path appropriate for your
machine. The following examples use your Documents directory:

```bash
python prepare_data.py --mode smoke --workspace "$HOME/Documents/DL4H-runs/smoke-data"
```

This downloads BMD-HS automatically and checks out revision
4745751855d15ff5fc9aaed29c32ad3ec7a5d409. It validates the exact eligible cohort
against the saved template, generates Normal/AS WAVs with seed 42, and writes
five reproducible synthetic split assignments. Mode controls generated counts:

| Mode | Synthetic train per class | Synthetic validation per class | Runs trained |
| --- | ---: | ---: | ---: |
| smoke | 10 | 10 | 1 |
| pilot | 500 | 100 | 1 |
| full | 5000 | 1000 | 5 |

To reuse an existing real checkout instead of downloading:

```bash
python prepare_data.py --mode smoke --workspace "$HOME/Documents/DL4H-runs/smoke-data" --existing-real /path/to/BMD-HS-Dataset
```

Use either command, not both on the same workspace. Existing source files are
not changed. The real CSV must match the pinned dataset; a changed source is
rejected rather than silently changing the cohort. Optionally reuse a generated
pool with --existing-synthetic-manifest /path/to/manifest.csv. Its original
per-class counts must match the chosen mode; different sizes are rejected.
--verify-audio-checksums performs an additional complete WAV checksum pass at
setup. Run loading always checks synthetic WAV checksums before processing;
real eligible-recording checksums are saved at setup and checked at loading.

Existing workspaces are never overwritten. If setup is interrupted, inspect the
failure and use a new workspace path. The original generator cannot resume an
incomplete audio batch. Successful workspaces are reused directly by training;
do not rerun setup. Setup may take minutes. Full WAVs use about 3.84 GB; processed
synthetic caches add about 3.6 GiB. Leave ample additional space for real caches
and checkpoints (at least 20 GiB recommended for full work, more if keeping copies).

The portable plans use paths relative to their location. Move an entire workspace
without changing layout. A workspace pointing to an external existing dataset
retains that relative reference; copy the dataset along with the same layout or
prepare a new workspace on the other machine. Do not hand-edit plan hashes.

## 4. Smoke mode: validate the packaged training path

On the Mac:

```bash
python run_experiments.py --mode smoke --workspace "$HOME/Documents/DL4H-runs/smoke-data" --output "$HOME/Documents/DL4H-runs/smoke-results" --device mps
python inspect_results.py --output "$HOME/Documents/DL4H-runs/smoke-results"
```

On NVIDIA use --device cuda:0; on a CPU-only machine use --device cpu. Smoke
uses the full width 64 ResNet18-style model, two epochs per stage, batch 4, and
two patients per class in each real training/validation partition. The saved
test partition is not sampled or predicted. It uses the same protocol code as
full mode, including transfer and Mixed loss. Expect several minutes, not a
fixed runtime. It is a software check, not scientific evidence.

For an optional epoch-boundary resume check, use a NEW output directory:

```bash
python run_experiments.py --mode smoke --workspace "$HOME/Documents/DL4H-runs/smoke-data" --output "$HOME/Documents/DL4H-runs/resume-check" --device mps --stop-after-epoch 1
python run_experiments.py --mode smoke --workspace "$HOME/Documents/DL4H-runs/smoke-data" --output "$HOME/Documents/DL4H-runs/resume-check" --device mps --resume
python inspect_results.py --output "$HOME/Documents/DL4H-runs/resume-check"
```

The first command stops after Real epoch 1. The second resumes to the unchanged
epoch 2 cap, then runs remaining protocols. It does not force an extra validation
on interruption. This demonstrates operational resume; exact bitwise equivalence
to uninterrupted training is not promised, especially across hardware.

## 5. Pilot mode: preliminary validation results in approximately one hour

```bash
python prepare_data.py --mode pilot --workspace "$HOME/Documents/DL4H-runs/pilot-data" --existing-real /path/to/BMD-HS-Dataset
python run_experiments.py --mode pilot --workspace "$HOME/Documents/DL4H-runs/pilot-data" --output "$HOME/Documents/DL4H-runs/pilot-results" --device mps
python inspect_results.py --output "$HOME/Documents/DL4H-runs/pilot-results"
```

One paired run; all eligible real train/validation data; 1000 synthetic training
and 200 synthetic validation examples; batch 8; at most 100 epochs per stage;
validation every epoch. --minutes 60 allocates 12 minutes Real, 24 Syn, 12 Syn-to-Real,
12 Mixed. Stages stop at an epoch boundary and may finish early under patience.
The runner avoids starting another epoch when the previous duration would exceed
the remaining stage budget. The first epoch must complete. Setup/loading,
initialization, checkpoint overhead and a long first epoch can exceed one hour.
This is an approximate budget, not a hard process-kill timer. Syn pretraining is
shared; fine-tuning does not repeat it. Reduced-data pilot checkpoints are never
promoted automatically into full runs. No held-out real-test evaluation is allowed.

## 6. Full mode on CUDA

Run CUDA smoke mode first. Benchmark memory/throughput to choose the batch size
and practical caps BEFORE starting full experiments. This package defaults to
the paper's nominal ResNet18 batch 512 and 500 epochs per stage, with validation
every 5 epochs and patience 5 unsuccessful evaluations. It does not assume a GPU
can hold 512 full 80,000-sample waveforms: OOM aborts; no automatic downscaling.
A batch reduction changes BatchNorm/optimization and must be reported.

```bash
python prepare_data.py --mode full --workspace /path/outside/repo/full-data --existing-real /path/to/BMD-HS-Dataset
python run_experiments.py --mode full --workspace /path/outside/repo/full-data --output /path/outside/repo/full-results --device cuda:0
```

Explicit documented alternatives, chosen on resource grounds, not test results:

```bash
python run_experiments.py --mode full --workspace /path/outside/repo/full-data --output /path/outside/repo/full-results-b8 --device cuda:0 --batch-size 8 --syn-epochs 50 --real-epochs 100 --syn2real-epochs 100 --mixed-epochs 100
```

The second command is an example reduced budget, not a measured recommendation.
Do not run both alternatives as test-guided selection. All overrides are saved.
--run-indices 0 can check a single full-data run; final freeze/evaluation requires
all five indices 0–4 in one completed suite. Five synthetic runs reuse one pool
with different saved assignments, not independent populations. No workers are
spawned by data loaders, improving reproducibility/portability over maximum speed.

## 7. Resume and failure behavior

Repeat the EXACT training command with --resume. Architecture, epoch caps,
mode, source hashes, split hashes, batch size, library/device identity and other
settings must match. Completed stages are verified/skipped; unfinished stages
resume from their last completed epoch. Interrupted partial epochs are repeated.
Python/NumPy/Torch/device RNG states, Adam state, validation history, counters
and elapsed training time are restored. No scheduler is used.

Only load checkpoints you created/trust: last.pt uses Python pickle for RNG
state. Do not resume an unrelated downloaded checkpoint. A process crash can
leave .running.lock. Confirm no training process is running, inspect the output,
then remove ONLY that lock manually before resuming. A stage that fails before
its first checkpoint needs a new output root after fixing the cause. Never
delete an output with useful checkpoints merely to bypass a mismatch.

## 8. Freeze and evaluate the untouched real test set

Only after all five full runs finish and all settings/overrides are final:

```bash
python evaluate_results.py freeze --output /path/outside/repo/full-results
python evaluate_results.py evaluate --output /path/outside/repo/full-results --workspace /path/outside/repo/full-data --device cuda:0 --batch-size 8 --confirm-frozen
```

Freeze hashes configurations, histories and best checkpoints without predicting.
Evaluation verifies hashes, predicts each selected checkpoint on the same saved
real-test patients, and writes heldout/report.json plus per-recording predictions.
It reports mean and sample SD (ddof=1) across five runs for AUROC, AUPRC (average
precision), F1, sensitivity, specificity, PPV and NPV. Threshold is 0.5; undefined
predictive values are defined as 0. Syn's final target is REAL test data, although
its checkpoint was chosen only on synthetic validation. Never select on test.
Existing evaluation outputs are not overwritten. If interrupted, inspect partial
predictions and plan recovery deliberately; automatic evaluation resume is not
implemented. Do not alter hyperparameters in response to those predictions.

## Outputs and handoff

Each run/protocol gets config.json, history.json, best.pt, last.pt and
stage_status.json. Suite configuration records code/plan/team-source hashes,
versions, device, overrides and deviations. Histories record optimizer updates
and real/synthetic exposures. Fine-tuning provenance hashes its selected Syn
checkpoint. Mixed starts RANDOM and does not require/load Syn weights.

Before handing to a teammate, send this code, source revision and all commands.
After their CUDA smoke passes, retain pip freeze, GPU model, environment, split
plans, dataset manifests and all result/checkpoint files outside Git. Return
suite_config.json, suite_status.json, histories, selected checkpoint hashes and
heldout/report.json for analysis; share large checkpoints separately if needed.

This folder is a preparation artifact. For an upstream PyHealth PR, agree with
the team on final placement (typically examples/synpcg and tests/core), adapt
test imports if relocating, and rerun tests/smoke from that final location.
The current local merge of Franko's files is pending; this package does not
finish it. Review exact staged files. Never git add the whole CS Project folder.
No package code commits or pushes anything.

Timing note: per-stage elapsed budget includes training, validation and best
checkpoint writes; last-checkpoint serialization and JSON logging are excluded.
Whole invocation walltime includes them. Timing is approximate, never a claim
of a hard one-hour limit. MPS/CUDA synchronization is used around timed epochs.

## Presentation notebook

Open `SynPCG_Presentation_Demo.ipynb` in the **Python (PyHealth Project)** kernel
after the current pilot finishes. Run All creates component plots, full and
zoomed Normal/AS waveforms, optional synthetic audio players, shared-scale
spectrograms, illustrative spectral distributions, the protocol diagram and
the patient-level split chart. It optionally plots JSON histories only when the
entire selected suite has status `complete`; incomplete runs are skipped.

Configuration paths, figure export and audio controls are in the first code
cell. The notebook's working directory must contain `synthetic_pcg.py` (or set
`CODE_DIR` explicitly). PNGs at 300 dpi and SVGs export under
`presentation_assets/`. Generated examples are synthetic demonstrations, not
clinical validation. Pilot curves are preliminary validation results and use
different validation domains for Syn versus the other protocols. They are not
a paper-result comparison. No model is trained or checkpoint loaded by the
notebook, and no real-test predictions are made. Avoid running it concurrently
with a timed pilot if you want minimally disturbed runtime measurements.

## Verification status

Verified manually on Tejaswi's Mac using the `pyhealth-project` Python 3.12
environment and MPS:

- All **50 package tests passed with no skips**, including the three original
  PyHealth integration tests.
- Fresh smoke setup downloaded the pinned BMD-HS revision and generated all
  **40 synthetic recordings** successfully.
- The packaged runner completed **all four smoke protocols**, two epochs each,
  and checked saved configurations, histories, counters and selected checkpoints.
- `inspect_results.py` completed successfully on those smoke artifacts.
- An intentional Real epoch-1 stop resumed at epoch 2 and then completed Syn,
  Syn-to-Real and Mixed. This is an **operational Real-stage resume check**, not
  proof of exact uninterrupted equivalence or a resume test of every new stage.
- Pilot setup generated **1,200 recordings** successfully. All four pilot
  stages completed and passed saved-artifact inspection: Real 17 epochs, Syn 9,
  Syn-to-Real 15 and Mixed 8. Each stopped at its allocated time budget.
- The presentation notebook ran successfully and exported all nine figures
  as PNG and SVG, including completed-pilot learning curves and workload charts.
  Saved notebook outputs contain no errors. Selected epochs and workload counts
  match the inspected pilot histories. The two pilot-result figures were also
  visually checked. No retraining or held-out evaluation was performed.

**Not yet verified:** CUDA execution, full five-run training, operational
interruption of each other
new-runner stage, or full-only freeze/test evaluation. No full paper-comparison
results are available, and the real test set has not been evaluated in this work.
The older T8 prototype had separate operational checks; those do not substitute
for tests of new code paths. Unit and smoke success establish software behavior,
not clinical fidelity or reproduced paper performance.
