# ResNet1D ablations

## Setup and smoke test

From the repository root, install this checkout with `pip install -e .`.
Audio runs also need torchaudio matching the installed Torch build, soundfile,
and torchcodec if required by that torchaudio release.

```sh
python -m unittest tests.core.test_resnet1d -v
python -m examples.resnet1d.ablation --smoke --output /local/runs/resnet1d-smoke
```

Use an empty output directory outside the checkout. The smoke run uses four
random tensors per split, one epoch, width 4, and two shallow stages. It checks
software integration, not PCG classification performance.

## Run ablations

Group 1 supplies processed `train`, `validation`, and `test` datasets using
`BMDHSDataset` and `BMDHSBinaryClassification`. Fit processors only on training
patients, then reuse them for held-out data. The runner does not create splits.

```python
from examples.resnet1d.ablation import run_ablation

run_ablation(train, validation, test, '/local/runs/group2',
             seeds=(0, 1, 2), epochs=20, batch_size=8, device='cuda:0')
```

The grid compares learning rates {2e-5, 1e-4}, widths {32,64}, and depths
{18,34}: eight configurations, or 24 fits with three seeds. Each fit starts
with fresh weights and seeded shuffling on the same supplied splits.
`results.json` records configuration, seed, parameter count, device, Torch
version, runtime, and validation metrics. Checkpoints contain the best weights
by validation AUROC. Metrics are AUROC, average precision (PyHealth `pr_auc`),
and F1 at threshold 0.5. Validation must contain both classes.

The runner rejects patient overlap and mismatched validation schemas or fitted
processor states. Equivalent loaded copies are allowed. These checks cannot
prove fitting provenance: Group 1 must ensure preprocessing saw only training
patients. Patient metadata is used where available to avoid decoding test
waveforms. Test scores never enter selection. Select by mean validation AUROC
across seeds, then freeze the configuration before Group 3's final test run.

Keep the split manifest, disease, processor settings, dataset revision, and
environment with the output. Use a new directory for each invocation.
Deterministic Torch algorithms are enabled, but results may differ across
hardware or library versions. CUDA may require setting
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python.

## Compute and handoff

For Colab, select a GPU runtime, install this checkout and matching audio
packages, mount BMD-HS and Group 1's split manifest, and call `run_ablation` as
above. No data is downloaded by this example. Use `device='cpu'` without CUDA;
full-recording training is not launched by the smoke command.

A float32 mono input uses 0.305 MiB. The width-64 stem alone produces about
9.8 MiB of activations per example; gradients and optimizer states add memory.
Start with batch size 2-8 and measure GPU usage. The three-seed grid costs
`24 * epochs * ceil(training_recordings / batch_size)` training steps plus
validation. Benchmark a real-data epoch before estimating wall-clock time.
ResNet-34 costs more; reduce the grid if needed.

Architecture, reference differences, and synthetic-to-real checkpoint examples
are in `docs/api/models/pyhealth.models.ResNet1D.rst`. Group 3 owns synthetic
generation and final protocols. Generator arrays must match the channels-first
float tensor contract, or be exported as WAV files for the audio processor.
No full BMD-HS or synthetic-to-real experimental results have been measured.
