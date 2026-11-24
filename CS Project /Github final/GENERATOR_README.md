# Synthetic Normal/AS phonocardiograms

Function-generated heart sounds for the team's binary Normal versus aortic
stenosis (AS) PyHealth experiment. The generator produces mono, 10-second,
8 kHz float32 recordings and a reproducible training/validation manifest.
Normal is label **0**; AS is label **1**. AR and MR are outside this implementation.

## Files

| File | Purpose |
| --- | --- |
| `synthetic_pcg.py` | Signal components, parameter sampling, complete recordings, batch export |
| `synthetic_manifest.py` | Manifest validation and optional PyHealth dataset creation |
| `generate_dataset_cli.py` | Command-line entry point for batch generation |
| `synthetic_splits.py` | Configurable fixed-pool run plans and per-run loading |
| `test_synthetic_splits.py` | Split reproducibility, isolation and run-loader tests |
| `EXPERIMENT_POLICY.md` | Agreed data-use policy and comparison rules |
| `test_synthetic_pcg.py` | Generator and export tests using temporary data |
| `test_synthetic_manifest.py` | Manifest tests and optional PyHealth integration tests |
| `synthetic_pcg_examples.ipynb` | Executed examples with plotting code and embedded figures |
| `requirements.txt` | Generator, visualization and notebook dependencies |

All synthesis and loading logic lives in Python modules. The notebook imports
these modules; it does not contain a second implementation. Generated WAVs,
datasets, caches, and local comparison reports are excluded from this folder.

## Installation

Run commands from this folder. Generator and CSV-only loading work with
Python 3.11-3.13. Optional integration with the team's current PyHealth checkout
requires Python 3.12 or 3.13, so use one of those if you intend to run models.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If using an existing environment, activate it and install the requirements there.
For generator-only use, `numpy` and `soundfile` are sufficient; plotting also
needs `matplotlib`. Requirements specify supported ranges, not a locked runtime.
The visualization notebook was executed with Python 3.11, NumPy 2.4.6,
SoundFile 0.14.0 and Matplotlib 3.11.2; integration was checked separately in
Python 3.12. Keep dependency versions fixed when comparing regenerated datasets.

## Visual examples

```bash
jupyter lab synthetic_pcg_examples.ipynb
```

Select the environment containing the requirements and run all cells. The
committed notebook includes rendered figures: raw beat components with and
without a murmur, four complete recordings, and three-second detail views.
It creates no audio files and does not run full dataset generation.
These plots demonstrate synthesis behavior, not clinical disease fidelity.

## Generate data

Start with a pilot: 10 training and 2 validation recordings per class, 24 total.

```bash
python generate_dataset_cli.py --output synthetic_data/pilot_seed42 --train-per-class 10 --validation-per-class 2 --seed 42
```

The paper's binary task uses 5,000 training and 1,000 validation recordings per
class: **10,000 training + 2,000 validation = 12,000 recordings**.

```bash
python generate_dataset_cli.py --output synthetic_data/paper_seed42 --seed 42
```

Allow roughly 3.84 GB for the full float32 WAV dataset, plus metadata. Generation
time depends on the machine. Output directories must be new or empty. The
current implementation does not resume interrupted batches or overwrite
nonempty directories; use a different output path for another attempt.

Python callers can use the same implementation directly:

```python
from synthetic_pcg import generate_recording, generate_dataset

waveform, metadata = generate_recording("AS", seed=42)
assert waveform.shape == (80000,)

manifest_path = generate_dataset(
    "synthetic_data/small_seed7",
    train_per_class=2,
    validation_per_class=1,
    master_seed=7,
)
```

`save_examples(examples, output_dir)` optionally exports a list of
`(waveform, metadata)` tuples for listening; it is separate from batch generation.

## Generated output

| Output | Contents |
| --- | --- |
| WAV files | Relative paths listed in the manifest; mono 8 kHz, 80,000 float32 samples by default |
| `manifest.csv` | IDs, partition, class, label, relative audio path, seed, audio dimensions and WAV SHA-256 |
| `dataset.json` | Configuration, source hash, seed policy, counts, library versions and generation status |
| `recording_metadata.jsonl` | Per-recording parameters, timing, retry counts and generation metadata |
| `generation_failures.jsonl` | Failed recording attempts; may be empty |

Manifest columns are `recording_id`, `patient_id`, `split`, `class_name`, `label`,
`signal`, `seed`, `sample_rate`, `samples`, `duration_seconds`, and `sha256`.
Partitions are `train` and `validation`. Synthetic patient IDs identify individual
generated recordings; they do not represent real people. Keep the complete
output directory together so relative audio paths continue to resolve.

## Read a manifest

CSV reading does not import PyHealth. This verifies all manifest rows before
returning a limited subset; checksum verification also reads each audio file.

```python
from synthetic_manifest import read_synthetic_manifest

samples = read_synthetic_manifest(
    "synthetic_data/pilot_seed42/manifest.csv",
    limit_per_class=2,
    verify_checksums=True,
)
print(len(samples["train"]), len(samples["validation"]))
```

The reader checks the expected labels/audio dimensions, missing paths, duplicate
IDs/paths/seeds across partitions, and completion/count information when
`dataset.json` is present. It preserves the generated partitions.

## Optional PyHealth loading

Use Python 3.12/3.13 and install the team's PyHealth checkout in that environment:

```bash
python -m pip install -e /path/to/PyHealth
```

Also install audio dependencies compatible with that environment's PyTorch
version: `torchaudio` and the decoder required by that version (the tested setup
uses `torchcodec` and native FFmpeg libraries). Follow the matching packages'
installation instructions for your platform. These optional dependencies are
not installed by this folder's requirements file. If audio decoding fails,
check those dependencies before running the full dataset through PyHealth.

```python
from synthetic_manifest import load_synthetic_datasets

train, validation = load_synthetic_datasets(
    "synthetic_data/pilot_seed42/manifest.csv",
    limit_per_class=2,
    in_memory=True,
)
```

The input schema is `signal: audio` at 8 kHz for 10 seconds; the output is a
binary `label`, matching the team's real-data task. Training fits processors;
validation reuses them. A batch signal has shape `(batch, 1, 80000)` and labels
have shape `(batch, 1)`. `in_memory=True` is convenient for small checks; the
loader's default is disk-backed. Loading all samples may create a substantial
additional processed cache. Model-side normalization and training are separate.

## Tests

```bash
python -m unittest -v test_synthetic_pcg test_synthetic_manifest test_synthetic_splits
```

There are 21 generator tests, 11 manifest/loader tests, and 8 split-policy tests.
Three tests require PyHealth and audio dependencies; they skip when prerequisites
are not detected. A generator-only run normally passes 37 tests with three
skipped; a fully configured integration run passes all 40. Tests use temporary
fixtures and do not modify an existing full dataset.

## Method and reproducibility

1. Sample recording parameters, then perturb that recording's parameters for
   each beat. Between-recording variation is larger than between-beat variation.
2. Construct S1/S2 from sinusoidal carriers and asymmetric Gaussian envelopes.
3. Add three low-frequency oscillations and two block Gaussian noise processes.
4. For AS, add shaped Gaussian noise with valid moving-average smoothing.
5. Concatenate beyond 1.5 times the target length, min-max scale to `[-1, 1]`,
   apply a valid recording-level moving average, and randomly crop 10 seconds.
6. Export WAVs and metadata with separate derived seeds for each partition,
   class, recording index and recording attempt.

The primary configuration is the reference's `syn02` Normal/AS configuration,
recorded in `SYNTHESIS_PROFILE`. It uses the reference murmur timing calculated
from S1 before its placement shift; this can overlap placed S1. The optional
`--timing positioned_s1` changes that rule and is an extension, not the primary
replication setting.

Known differences from the reference are intentional and documented:

- PCG64 streams and parameter draw order differ from global legacy NumPy random
  state; the same integer seed does not reproduce the reference's exact audio.
- Invalid parameters/beats use bounded rejection and retries rather than dropping
  the complete recording. This can affect the sampling distribution.
- Sample times use `n / sample_rate`; the reference uses a duration-based
  `linspace`. They can differ when beat duration times sampling rate is noninteger.
- Final audio is float32 and exported as FLOAT WAVs; the reference's primary
  in-memory/pickled recordings are float64.
- Output uses WAVs/manifests instead of pickle batches. The reference scripts
  create a larger candidate pool; these defaults match the paper's experimental
  sample counts for the binary task.

A local comparison checked all 50 Normal/AS parameter specifications, aligned
random draws for raw beats, and 400 full trial recordings. Integer-grid beats
matched to floating-point precision; independently sampled recordings had
similar descriptive summaries. This does not establish distributional
equivalence, disease fidelity, or reproduced classification performance.

Use `dataset.json` to retain the source hash, settings and dependency versions.
Changing generator source changes its recorded hash. Our selected policy reuses
one pool across five stratified splits; see the run-plan instructions below and
[EXPERIMENT_POLICY.md](EXPERIMENT_POLICY.md).

## Five-run synthetic-data policy

Reuse the existing pool with five training/validation assignments. By default,
run 1 keeps the original partition, runs 2-5 reshuffle within each class, and
training seeds are 0-4. For the full pool, each run has 5,000 training and 1,000
validation recordings per class. No new audio is generated or copied.

For a dataset generated in this folder:

```bash
python synthetic_splits.py --manifest synthetic_data/paper_seed42/manifest.csv --output synthetic_runs/five_runs.json
```

For the existing dataset beside `Github folder`, use
`--manifest ../synthetic_data/paper_seed42/manifest.csv` instead.
Run plans are local generated artifacts excluded by `.gitignore`. They store
exact recording assignments, the source manifest hash and relative location,
and each run's split/training seeds. Keep their relative layout when moving
files. Existing plans are not overwritten.

Customize all runs, seeds and counts through the Python API:

```python
from synthetic_splits import create_split_plan, read_synthetic_run, load_synthetic_run

plan_path = create_split_plan(
    "synthetic_data/paper_seed42/manifest.csv",
    "synthetic_runs/custom_five_runs.json",
    n_runs=5,
    preserve_first=True,
    split_seeds=[42, 43, 44, 45],  # One per reshuffled run, excluding preserved run 1.
    training_seeds=[0, 1, 2, 3, 4],
    train_per_class=5000,
    validation_per_class=1000,
)
samples = read_synthetic_run(plan_path, run_index=1)  # Zero-based: run 2.

# Optional PyHealth integration:
train, validation = load_synthetic_run(
    plan_path, run_index=1, limit_per_class=2, in_memory=True,
)
```

`n_runs`, seed lists and counts are configurable. Counts default to the original
manifest, so small pilot pools work without changes. Training plus validation
counts must equal the pool size per class. To change partition sizes or reshuffle
run 1, use `preserve_first=False`; provide one split seed for every run. Omit
explicit seed lists to use consecutive split seeds starting at `split_seed=42`
and training seeds starting at zero. All seed lists must contain distinct,
nonnegative integers. The CLI exposes the same settings; use `--help`.

Training seeds are **recorded, not automatically applied to PyTorch**. The later
experiment runner must set its random seeds, select the corresponding real split,
and reuse the same synthetic assignments/checkpoint across conditions within
that run. The run loader fits processors on that run's training data and shares
them with its validation set. Record IDs retain their original names even when
assigned to a different partition; the run plan, not the ID text or original
manifest partition, determines membership for that run.

Independent reshuffles can overlap across runs; they are not disjoint folds or
independently generated populations. The real test set remains fixed and held
out. The current code creates/loads synthetic splits; it does not yet train
models, select real splits or implement the experiment runner.

## Reference and attribution

Naoki Nonaka, Hiroshi Seki, Tomohiro Komatsu, and Jun Seita. **Boosting
Phonocardiogram Classification Performance with Function Generated Data.**
Proceedings of the 5th Machine Learning for Health Symposium,
PMLR 297:473-524, 2026.
[Paper](https://proceedings.mlr.press/v297/nonaka26a.html).

Reference implementation: [seitalab/SynPCG](https://github.com/seitalab/SynPCG),
revision `1a5f600ce2a0a9e43514e20582da67081159a14a`. Consulted files include
`src/prep/syn/syn_utils.py`, `syn_pcg.py`, `syn_as.py`, and the Normal/AS
`resources/*/syn02.yaml` configurations. Contributor: Tejaswi Sapkota.

At review time the reference README's license section was unspecified. This
folder preserves attribution and does not add a new license declaration or
claim that upstream reuse terms have been resolved.
