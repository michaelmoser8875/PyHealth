"""
Task ablation for BMDHSBinaryClassification with ContraWR.

Paper (replicated):
    Nonaka et al. "Boosting Phonocardiogram Classification Performance with
    Function Generated Data." ML4H 2025 (PMLR 297).

Paper link:
    https://proceedings.mlr.press/v297/nonaka26a.html

Authors:
    Franko Hysaj (fhysaj2@illinois.edu)
    Tejaswi Sapkota (sapkota3@illinois.edu)
    Michael Moser (mjmoser2@illinois.edu)

Description:
    Trains an existing PyHealth model (ContraWR, which works on the STFT of
    the raw signal) on BMD-HS with different task settings, to show how the
    task's choices change the result:

    - other_disease_policy: "exclude" vs "negative" (AS as the disease).
    - disease: AS vs MR (both with "exclude").

    Every run uses one fixed patient-level test split (20%) and a seeded
    80/20 train/validation split of the other patients
    (split_by_patient_fixed_test), and reports test AUROC, AUPRC and F1 as
    mean +/- SD over the seeds. SparcNet was tried first but took over 60 s
    per batch of 10 s, 8 kHz signals on a laptop CPU, so ContraWR is used.

Usage:
    python examples/bmd_hs_binary_classification_contrawr.py \\
        --root /path/to/BMD-HS-Dataset

Results:
    Defaults (5 seeds, Adam lr 1e-3, batch 32, up to 50 epochs, early
    stopping after 5 epochs without validation AUROC improvement), CPU only
    (Apple M5, about 12 minutes in total). Test metrics, mean +/- SD:

    ======= ======== ======= =============== =============== ===============
    disease policy   samples AUROC           AUPRC           F1
    ======= ======== ======= =============== =============== ===============
    AS      exclude  464     0.637 +/- 0.172 0.675 +/- 0.146 0.721 +/- 0.023
    AS      negative 864     0.616 +/- 0.039 0.315 +/- 0.031 0.088 +/- 0.154
    MR      exclude  472     0.569 +/- 0.061 0.677 +/- 0.041 0.785 +/- 0.010
    ======= ======== ======= =============== =============== ===============

    "negative" nearly doubles the data but adds AR/MR/MS patients as
    negatives, so positives drop from 64% to 34% of samples: AUROC barely
    moves, AUPRC halves (it tracks the positive rate), and F1 at the 0.5
    threshold collapses because the model rarely predicts positive. With
    "exclude", AS AUROC varies a lot between seeds (+/- 0.17), as the test
    split holds only 11 patients (6 AS, 5 Normal). MR is harder to
    separate from Normal than AS for this small model. All AUROCs are well
    below the paper's ResNet18 (0.7165 for AS, Real), as expected for a
    ~2k-parameter model without augmentation.
"""

import argparse

import numpy as np

from pyhealth.datasets import (
    BMDHSDataset,
    get_dataloader,
    split_by_patient_fixed_test,
)
from pyhealth.models import ContraWR
from pyhealth.tasks import BMDHSBinaryClassification
from pyhealth.trainer import Trainer

# Each setting changes one task argument from the AS/exclude baseline.
SETTINGS = [
    {"disease": "AS", "other_disease_policy": "exclude"},
    {"disease": "AS", "other_disease_policy": "negative"},
    {"disease": "MR", "other_disease_policy": "exclude"},
]
METRICS = ["roc_auc", "pr_auc", "f1"]


def run(samples, seed: int, args: argparse.Namespace) -> dict:
    """Trains and tests ContraWR on one seeded train/validation split."""
    train, val, test = split_by_patient_fixed_test(samples, seed=seed)
    train_loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val, batch_size=args.batch_size)
    test_loader = get_dataloader(test, batch_size=args.batch_size)

    model = ContraWR(dataset=samples)
    trainer = Trainer(model=model, metrics=METRICS, device=args.device)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor="roc_auc",
        patience=args.patience,
    )
    return trainer.evaluate(test_loader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", required=True, help="BMD-HS-Dataset folder")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = BMDHSDataset(root=args.root)
    rows = []
    for setting in SETTINGS:
        samples = dataset.set_task(BMDHSBinaryClassification(**setting))
        scores = [run(samples, seed, args) for seed in range(args.seeds)]
        summary = {
            m: (np.mean([s[m] for s in scores]), np.std([s[m] for s in scores]))
            for m in METRICS
        }
        rows.append((setting, len(samples), summary))

    print("\ndisease | policy   | samples | " + " | ".join(METRICS))
    for setting, n, summary in rows:
        cells = " | ".join(f"{mean:.3f} +/- {sd:.3f}" for mean, sd in summary.values())
        print(
            f"{setting['disease']:7} | {setting['other_disease_policy']:8} "
            f"| {n:7} | {cells}"
        )


if __name__ == "__main__":
    main()
