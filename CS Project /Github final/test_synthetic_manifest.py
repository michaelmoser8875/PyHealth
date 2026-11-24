"""Tests for the synthetic manifest loader using temporary WAV fixtures.

Contributor: Tejaswi Sapkota (sapkota3@illinois.edu).
Paper: Nonaka et al., Function Generated PCG Data, ML4H 2025 (PMLR 297).
https://proceedings.mlr.press/v297/nonaka26a.html

Run: python -m unittest -v test_synthetic_manifest
PyHealth integration tests require its Python 3.12/3.13 environment. CSV tests
also run in the generator's Python 3.11 environment without importing PyHealth.
"""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

from synthetic_manifest import read_synthetic_manifest, load_synthetic_datasets


class ManifestFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "manifest.csv"
        self.rows = []
        time = np.arange(80000) / 8000
        for split in ("train", "validation"):
            for label, name in enumerate(("Normal", "AS")):
                for index in range(2):
                    rid = f"syn_{split}_{name}_{index}"
                    path = self.root / f"{rid}.wav"
                    # Fixtures test loading, not synthetic disease realism.
                    wave = 0.1 * np.sin(2 * np.pi * (24 + len(self.rows)) * time)
                    sf.write(path, wave, 8000, subtype="FLOAT")
                    self.rows.append(
                        {
                            "recording_id": rid,
                            "patient_id": rid,
                            "split": split,
                            "class_name": name,
                            "label": label,
                            "signal": path.name,
                            "seed": len(self.rows),
                            "sample_rate": 8000,
                            "samples": 80000,
                            "duration_seconds": 10.0,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
        self.write_manifest()

    def write_manifest(self):
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


class TestManifestReading(ManifestFixture):
    def test_labels_paths_and_partitions(self):
        result = read_synthetic_manifest(self.manifest, verify_checksums=True)
        self.assertEqual(set(result), {"train", "validation"})
        self.assertEqual([len(result[s]) for s in result], [4, 4])
        for samples in result.values():
            self.assertEqual({s["label"] for s in samples}, {0, 1})
            self.assertTrue(all(Path(s["signal"]).is_absolute() for s in samples))
            self.assertTrue(all(Path(s["signal"]).is_file() for s in samples))
        train = {s["patient_id"] for s in result["train"]}
        val = {s["patient_id"] for s in result["validation"]}
        self.assertFalse(train & val)

    def test_limit_is_per_class_and_partition(self):
        result = read_synthetic_manifest(self.manifest, limit_per_class=1)
        for samples in result.values():
            self.assertEqual(len(samples), 2)
            self.assertEqual({s["label"] for s in samples}, {0, 1})

    def test_hidden_rows_are_still_validated(self):
        self.rows[1]["signal"] = "missing.wav"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Missing audio"):
            read_synthetic_manifest(self.manifest, limit_per_class=1)

    def test_invalid_rows_are_rejected(self):
        changes = [
            {"label": 1},
            {"split": "test"},
            {"class_name": "MR"},
            {"sample_rate": 4000},
            {"samples": 40000},
            {"seed": -1},
            {"signal": "../outside.wav"},
            {"patient_id": ""},
        ]
        original = dict(self.rows[0])
        for change in changes:
            self.rows[0] = {**original, **change}
            self.write_manifest()
            with self.subTest(change=change), self.assertRaises(ValueError):
                read_synthetic_manifest(self.manifest)
        self.rows[0] = original

    def test_duplicate_ids_and_seeds_across_partitions(self):
        original = dict(self.rows[-1])
        for field in ("recording_id", "patient_id", "signal", "seed"):
            self.rows[-1] = {**original, field: self.rows[0][field]}
            self.write_manifest()
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "duplicate"
            ):
                read_synthetic_manifest(self.manifest)

    def test_checksum_mismatch(self):
        self.rows[0]["sha256"] = "0" * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            read_synthetic_manifest(self.manifest, verify_checksums=True)

    def test_missing_class_and_incomplete_generation(self):
        self.rows = [
            r
            for r in self.rows
            if not (r["split"] == "validation" and r["class_name"] == "AS")
        ]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "each contain"):
            read_synthetic_manifest(self.manifest)
        (self.root / "dataset.json").write_text(json.dumps({"status": "running"}))
        with self.assertRaisesRegex(ValueError, "not complete"):
            read_synthetic_manifest(self.manifest)

    def test_invalid_limit_and_missing_columns(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(limit=value), self.assertRaises(ValueError):
                read_synthetic_manifest(self.manifest, limit_per_class=value)
        self.manifest.write_text("signal,label\nfile.wav,0\n")
        with self.assertRaisesRegex(ValueError, "missing required"):
            read_synthetic_manifest(self.manifest)

    def test_completed_dataset_counts_are_checked(self):
        (self.root / "dataset.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "completed_total": 12,
                    "counts_per_class": {"train": 3, "validation": 3},
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "dataset total"):
            read_synthetic_manifest(self.manifest, limit_per_class=1)


HAS_PYHEALTH = (
    sys.version_info >= (3, 12)
    and importlib.util.find_spec("pyhealth") is not None
    and importlib.util.find_spec("torchaudio") is not None
)


@unittest.skipUnless(HAS_PYHEALTH, "Requires PyHealth/audio in Python 3.12/3.13")
class TestPyHealthIntegration(ManifestFixture):
    def test_processed_shapes_labels_and_shared_processors(self):
        import torch
        from pyhealth.datasets import get_dataloader

        train, val = load_synthetic_datasets(
            self.manifest, limit_per_class=1, in_memory=True
        )
        self.assertEqual((len(train), len(val)), (2, 2))
        self.assertIs(train.input_processors["signal"], val.input_processors["signal"])
        self.assertIs(train.output_processors["label"], val.output_processors["label"])
        for dataset in (train, val):
            batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
            self.assertEqual(tuple(batch["signal"].shape), (2, 1, 80000))
            self.assertEqual(tuple(batch["label"].shape), (2, 1))
            self.assertEqual(batch["signal"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(batch["signal"]).all())
            self.assertEqual(set(batch["label"].flatten().tolist()), {0.0, 1.0})

    def test_disk_backed_datasets(self):
        train, val = load_synthetic_datasets(
            self.manifest, limit_per_class=1, in_memory=False
        )
        self.assertEqual((len(train), len(val)), (2, 2))
        self.assertEqual(tuple(train[0]["signal"].shape), (1, 80000))
        self.assertEqual(tuple(val[0]["signal"].shape), (1, 80000))


if __name__ == "__main__":
    unittest.main()
