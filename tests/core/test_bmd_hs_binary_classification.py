"""
Tests for BMDHSBinaryClassification and split_by_patient_fixed_test.

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
    Unit tests on synthetic BMD-HS-style patients and 0.1 s .wav files.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, split_by_patient_fixed_test
from pyhealth.processors import AudioProcessor
from pyhealth.tasks import BMDHSBinaryClassification

try:
    import torchaudio  # noqa: F401

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

FLAGS = ["AS", "AR", "MR", "MS", "N"]


class TestBMDHSBinaryClassification(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        for name in ("rec_a", "rec_b"):
            sf.write(self.root / f"{name}.wav", rng.standard_normal(400), 4000)

    def tearDown(self):
        shutil.rmtree(self.root)

    def _patient(self, pid, flags, recordings=("rec_a", "rec_b")):
        """Builds a patient shaped like BMDHSDataset output."""
        paths = [str(self.root / f"{name}.wav") for name in recordings]
        paths += [None] * (8 - len(paths))
        diag = {f"diagnoses/{k}": [str(v)] for k, v in zip(FLAGS, flags)}
        recs = {f"recordings/recording_{i + 1}": [p] for i, p in enumerate(paths)}
        events = pl.concat(
            [
                pl.DataFrame({"event_type": ["diagnoses"], **diag}),
                pl.DataFrame({"event_type": ["recordings"], **recs}),
            ],
            how="diagonal",
        ).with_columns(
            pl.lit(pid).alias("patient_id"),
            pl.lit(None, dtype=pl.Datetime).alias("timestamp"),
        )
        return Patient(patient_id=pid, data_source=events)

    def test_normal_and_disease_labels(self):
        task = BMDHSBinaryClassification(disease="AS")
        normal = task(self._patient("n", [0, 0, 0, 0, 1]))
        sick = task(self._patient("as", [1, 0, 0, 0, 0]))
        self.assertEqual([s["label"] for s in normal], [0, 0])
        self.assertEqual([s["label"] for s in sick], [1, 1])
        self.assertTrue(all(s["patient_id"] == "as" for s in sick))

    def test_multi_disease_is_positive(self):
        task = BMDHSBinaryClassification(disease="AS")
        samples = task(self._patient("as_mr", [1, 0, 1, 0, 0]))
        self.assertEqual([s["label"] for s in samples], [1, 1])

    def test_other_disease_policy(self):
        mr_only = [0, 0, 1, 0, 0]
        exclude = BMDHSBinaryClassification(disease="AS")
        negative = BMDHSBinaryClassification(
            disease="AS", other_disease_policy="negative"
        )
        self.assertEqual(exclude(self._patient("mr", mr_only)), [])
        samples = negative(self._patient("mr", mr_only))
        self.assertEqual([s["label"] for s in samples], [0, 0])

    def test_na_diagnosis_is_excluded(self):
        task = BMDHSBinaryClassification(other_disease_policy="negative")
        self.assertEqual(task(self._patient("p22", ["N/A"] * 5)), [])

    def test_missing_recordings_are_skipped(self):
        task = BMDHSBinaryClassification()
        patient = self._patient("n", [0, 0, 0, 0, 1], ("rec_a", "", "gone"))
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0]["signal"].endswith("rec_a.wav"))

    def test_positions(self):
        task = BMDHSBinaryClassification(positions=[2])
        samples = task(self._patient("n", [0, 0, 0, 0, 1]))
        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0]["signal"].endswith("rec_b.wav"))

    def test_invalid_arguments_raise(self):
        for kwargs in (
            {"disease": "XYZ"},
            {"other_disease_policy": "drop"},
            {"positions": [0]},
            {"positions": []},
        ):
            with self.assertRaises(ValueError):
                BMDHSBinaryClassification(**kwargs)

    @unittest.skipUnless(TORCHAUDIO_AVAILABLE, "torchaudio not available")
    def test_signal_length_after_resample_and_crop(self):
        _, kwargs = BMDHSBinaryClassification.input_schema["signal"]
        signal = AudioProcessor(**kwargs).process(self.root / "rec_a.wav")
        self.assertEqual(tuple(signal.shape), (1, 80000))


class TestSplitByPatientFixedTest(unittest.TestCase):
    def setUp(self):
        samples = [
            {"patient_id": f"p{i}", "feature": i, "label": i % 2}
            for i in range(20)
            for _ in range(2)
        ]
        self.dataset = create_sample_dataset(
            samples, {"feature": "raw"}, {"label": "raw"}
        )

    @staticmethod
    def _patients(part):
        return {part[i]["patient_id"] for i in range(len(part))}

    def test_test_set_fixed_across_seeds(self):
        splits = [split_by_patient_fixed_test(self.dataset, seed=s) for s in (0, 1)]
        self.assertEqual(self._patients(splits[0][2]), self._patients(splits[1][2]))
        self.assertNotEqual(self._patients(splits[0][0]), self._patients(splits[1][0]))

    def test_parts_are_disjoint_and_sized(self):
        train, val, test = split_by_patient_fixed_test(self.dataset, seed=0)
        parts = [self._patients(p) for p in (train, val, test)]
        self.assertEqual([len(p) for p in parts], [13, 3, 4])
        self.assertEqual(len(set().union(*parts)), 20)


if __name__ == "__main__":
    unittest.main()
