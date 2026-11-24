"""Tests for the Normal/AS synthetic PCG generator.

Paper:
    Nonaka et al. "Boosting Phonocardiogram Classification Performance with
    Function Generated Data." ML4H 2025 (PMLR 297).
    https://proceedings.mlr.press/v297/nonaka26a.html

Contributor:
    Tejaswi Sapkota (sapkota3@illinois.edu)

Description:
    Uses generated signals and temporary folders only. Does not load BMD-HS
    or modify the full synthetic dataset. Run from this file's directory:
    python -m unittest -v test_synthetic_pcg
"""

import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import synthetic_pcg as generator


class TestSignalComponents(unittest.TestCase):
    def test_asymmetric_envelope(self):
        coordinate, envelope = generator.asymmetric_gaussian(8001, 0.75, 0.25)
        peak = len(coordinate) // 2
        self.assertEqual(coordinate[peak], 0)
        self.assertTrue(np.isfinite(envelope).all())
        self.assertTrue((envelope >= 0).all())
        self.assertEqual(np.argmax(envelope), peak)
        # At equal distances from the peak, the rising side is narrower.
        self.assertLess(envelope[peak - 100], envelope[peak + 100])

    def test_burst_height_and_position(self):
        sound = generator.generate_heart_sound(
            frequency=24,
            peak_height=2.5,
            envelope_width=0.75,
            shift_ratio=0.20,
            seed=42,
        )
        self.assertEqual(sound.shape, (8000,))
        self.assertTrue(np.isfinite(sound).all())
        self.assertFalse(np.any(sound[:1600]))
        self.assertAlmostEqual(np.max(np.abs(sound)), 2.5)

    def test_noise_has_fresh_randomness(self):
        first_rng = np.random.default_rng(42)
        _, first = generator.generate_background_noise(rng=first_rng)
        _, second = generator.generate_background_noise(rng=first_rng)
        _, replay = generator.generate_background_noise(seed=42)
        np.testing.assert_array_equal(first, replay)
        self.assertFalse(np.array_equal(first, second))
        self.assertGreater(np.std(first), 0)

    def test_murmur_is_limited_to_its_window(self):
        time, murmur = generator.generate_as_murmur(
            start_ratio=0.40, end_ratio=0.55, seed=42
        )
        self.assertEqual(len(time), 8000)
        self.assertFalse(np.any(murmur[:3200]))
        self.assertFalse(np.any(murmur[4400:]))
        self.assertGreater(np.sum(murmur[3200:4400] ** 2), 0)
        self.assertTrue(np.isfinite(murmur).all())

    def test_normal_beat_has_no_murmur(self):
        rng = np.random.default_rng(303)
        base, _ = generator.sample_recording_parameters("Normal", rng)
        parameters, _ = generator.sample_beat_parameters(base, "Normal", rng)
        beat = generator.generate_beat("Normal", parameters, rng)
        self.assertFalse(np.any(beat["murmur"]))
        self.assertIsNone(beat["murmur_window"])
        np.testing.assert_array_equal(beat["waveform"], beat["baseline"])

    def test_as_beat_adds_murmur_to_baseline(self):
        rng = np.random.default_rng(303)
        base, _ = generator.sample_recording_parameters("AS", rng)
        parameters, _ = generator.sample_beat_parameters(base, "AS", rng)
        beat = generator.generate_beat("AS", parameters, rng)
        self.assertGreater(np.sum(beat["murmur"] ** 2), 0)
        np.testing.assert_allclose(
            beat["waveform"] - beat["baseline"], beat["murmur"], atol=1e-14
        )

    def test_invalid_component_arguments(self):
        cases = [
            (generator.generate_sine_wave, {"frequency": 4000}),
            (generator.asymmetric_gaussian, {"num_samples": 100, "std_pos": 0}),
            (generator.generate_background_noise, {"widths": (0,), "scales": (0.1,)}),
            (generator.generate_as_murmur, {"start_ratio": 0.6, "end_ratio": 0.4}),
            (generator.generate_as_murmur, {"smoothing_width": 0}),
        ]
        for function, arguments in cases:
            with self.subTest(function=function.__name__, arguments=arguments):
                with self.assertRaises(ValueError):
                    function(**arguments)


class TestCompleteRecordings(unittest.TestCase):
    def test_paper_shape_and_range(self):
        for label in ("Normal", "AS"):
            with self.subTest(label=label):
                waveform, metadata = generator.generate_recording(label, seed=201)
                self.assertEqual(waveform.shape, (80000,))
                self.assertEqual(waveform.dtype, np.float32)
                self.assertTrue(np.isfinite(waveform).all())
                self.assertGreater(np.std(waveform), 0)
                self.assertLessEqual(np.max(np.abs(waveform)), 1.000001)
                self.assertEqual(metadata["binary_label"], int(label == "AS"))
                self.assertEqual(metadata["sample_rate"], 8000)
                self.assertEqual(metadata["timing"], "reference")

    def test_same_seed_reproduces_audio_and_metadata(self):
        first, first_metadata = generator.generate_recording("AS", 42)
        second, second_metadata = generator.generate_recording("AS", 42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_metadata, second_metadata)

    def test_different_seeds_change_recordings(self):
        first, _ = generator.generate_recording("Normal", 42)
        second, _ = generator.generate_recording("Normal", 43)
        self.assertFalse(np.array_equal(first, second))

    def test_recording_and_beat_parameters_vary(self):
        rng = np.random.default_rng(42)
        first, _ = generator.sample_recording_parameters("Normal", rng)
        second, _ = generator.sample_recording_parameters("Normal", rng)
        saved_base = dict(first)
        beat_a, _ = generator.sample_beat_parameters(first, "Normal", rng)
        beat_b, _ = generator.sample_beat_parameters(first, "Normal", rng)
        self.assertNotEqual(first, second)
        self.assertNotEqual(beat_a, beat_b)
        self.assertEqual(first, saved_base)

    def test_alternative_timing_is_recorded(self):
        waveform, metadata = generator.generate_recording(
            "AS", seed=99, timing="positioned_s1"
        )
        self.assertEqual(waveform.shape, (80000,))
        self.assertEqual(metadata["effective_settings"]["as_timing"], "positioned_s1")

    def test_invalid_recording_arguments(self):
        for arguments in ({"label": "MR"}, {"duration": 0}, {"timing": "invalid"}):
            options = {"label": "Normal", "seed": 42, **arguments}
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                generator.generate_recording(**options)

    def test_example_wav_round_trip(self):
        waveform, metadata = generator.generate_recording("Normal", seed=42)
        with tempfile.TemporaryDirectory() as folder:
            manifest = generator.save_examples([(waveform, metadata)], folder)
            filename = json.loads(manifest.read_text())[0]["filename"]
            loaded, sample_rate = sf.read(Path(folder) / filename, dtype="float32")
            np.testing.assert_array_equal(waveform, loaded)
            self.assertEqual(sample_rate, 8000)
            # The same export can be repeated without replacing different data.
            generator.save_examples([(waveform, metadata)], folder)


class TestBatchGeneration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def batch(self, folder="batch", **options):
        defaults = {
            "train_per_class": 2,
            "validation_per_class": 1,
            "progress_every": 0,
        }
        return generator.generate_dataset(self.root / folder, **(defaults | options))

    @staticmethod
    def rows(manifest):
        with manifest.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_batch_counts_seeds_and_partition_separation(self):
        rows = self.rows(self.batch())
        counts = Counter((row["split"], row["class_name"]) for row in rows)
        self.assertEqual(
            counts,
            {
                ("train", "Normal"): 2,
                ("train", "AS"): 2,
                ("validation", "Normal"): 1,
                ("validation", "AS"): 1,
            },
        )
        for field in ("recording_id", "patient_id", "seed", "signal", "sha256"):
            with self.subTest(field=field):
                self.assertEqual(len({row[field] for row in rows}), 6)
                train = {r[field] for r in rows if r["split"] == "train"}
                validation = {r[field] for r in rows if r["split"] == "validation"}
                self.assertFalse(train & validation)

    def test_batch_files_and_metadata(self):
        manifest = self.batch()
        rows = self.rows(manifest)
        for row in rows:
            path = manifest.parent / row["signal"]
            self.assertTrue(path.resolve().is_relative_to(manifest.parent.resolve()))
            waveform, sample_rate = sf.read(path, dtype="float32")
            self.assertEqual((waveform.shape, sample_rate), ((80000,), 8000))
            self.assertTrue(np.isfinite(waveform).all())
            self.assertEqual(sf.info(path).channels, 1)
            self.assertEqual(row["label"], str(int(row["class_name"] == "AS")))
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"]
            )
        settings = json.loads((manifest.parent / "dataset.json").read_text())
        self.assertEqual(settings["status"], "complete")
        self.assertEqual(settings["completed_total"], 6)
        self.assertEqual(len(list(manifest.parent.rglob("*.wav"))), 6)
        metadata = [
            json.loads(line)
            for line in (manifest.parent / "recording_metadata.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            {r["recording_id"] for r in rows}, {r["recording_id"] for r in metadata}
        )
        by_id = {row["recording_id"]: row for row in rows}
        for item in metadata:
            row = by_id[item["recording_id"]]
            self.assertEqual(item["seed"], int(row["seed"]))
            self.assertEqual(item["binary_label"], int(row["label"]))

    def test_batch_reproducibility(self):
        first = self.rows(self.batch("first"))
        second = self.rows(self.batch("second"))
        self.assertEqual(first, second)

    def test_existing_data_is_not_overwritten(self):
        manifest = self.batch()
        before = manifest.read_bytes()
        with self.assertRaises(FileExistsError):
            self.batch()
        self.assertEqual(manifest.read_bytes(), before)

    def test_invalid_batch_arguments_do_not_create_output(self):
        for options in (
            {"train_per_class": -1},
            {"validation_per_class": 1.5},
            {"master_seed": True},
            {"max_recording_attempts": 0},
            {"timing": "invalid"},
            {"train_per_class": 0, "validation_per_class": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.batch("invalid", **options)
            self.assertFalse((self.root / "invalid").exists())

    def test_failed_recording_is_retried_and_logged(self):
        original = generator.generate_recording
        calls = 0

        def fail_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("Simulated invalid recording")
            return original(*args, **kwargs)

        with patch.object(generator, "generate_recording", side_effect=fail_first):
            manifest = self.batch(train_per_class=1, validation_per_class=0)
        self.assertEqual(len(self.rows(manifest)), 2)
        settings = json.loads((manifest.parent / "dataset.json").read_text())
        self.assertEqual(settings["failed_attempts"], 1)
        self.assertEqual(settings["status"], "complete")
        failures = (
            (manifest.parent / "generation_failures.jsonl").read_text().splitlines()
        )
        self.assertEqual(len(failures), 1)

    def test_persistent_failure_does_not_report_success(self):
        with patch.object(
            generator, "generate_recording", side_effect=ValueError("Bad draw")
        ):
            with self.assertRaises(RuntimeError):
                self.batch(max_recording_attempts=2)
        settings = json.loads((self.root / "batch/dataset.json").read_text())
        self.assertEqual(settings["status"], "failed")
        self.assertEqual(settings["completed_total"], 0)
        self.assertEqual(settings["failed_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
