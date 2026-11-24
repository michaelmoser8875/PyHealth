"""Function-generated Normal and AS heart sounds for the SynPCG replication.

Paper:
    Nonaka et al. "Boosting Phonocardiogram Classification Performance with
    Function Generated Data." ML4H 2025 (PMLR 297).
    https://proceedings.mlr.press/v297/nonaka26a.html

Contributor:
    Tejaswi Sapkota (sapkota3@illinois.edu)

Description:
    Generates mono 10-second recordings at 8 kHz for the team's binary
    Normal-vs-AS task. Returns float32 waveforms and generation metadata.
    Exported WAVs can be read by PyHealth's AudioProcessor; creating the
    SampleDataset is a separate integration step.

Reference:
    https://github.com/seitalab/SynPCG
    The pinned revision and numerical differences are in SYNTHESIS_PROFILE.

Example:
    >>> waveform, metadata = generate_recording("Normal", seed=42)
    >>> waveform.shape
    (80000,)
"""

import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def _get_rng(
    rng: np.random.Generator | None = None, seed: int | None = None
) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Pass rng or seed, not both.")
    return np.random.default_rng(seed) if rng is None else rng


def _sample_count(duration: float, sample_rate: int) -> int:
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite.")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or sample_rate <= 0
    ):
        raise ValueError("sample_rate must be a positive integer.")
    count = int(duration * sample_rate)
    if count < 2:
        raise ValueError("At least two samples are required.")
    return count


def shift_wave(wave: np.ndarray, shift_samples: int) -> np.ndarray:
    """Zero-padded shift: positive moves right; negative moves left."""
    wave = np.asarray(wave, dtype=float)
    shift_samples = int(shift_samples)
    result = np.zeros_like(wave)
    if shift_samples == 0:
        return wave.copy()
    if abs(shift_samples) < len(wave):
        if shift_samples > 0:
            result[shift_samples:] = wave[:-shift_samples]
        else:
            result[:shift_samples] = wave[-shift_samples:]
    return result


def moving_average(wave: np.ndarray, width: int) -> np.ndarray:
    """Algorithm 7: valid convolution; output length is L - width + 1."""
    if (
        isinstance(width, bool)
        or not isinstance(width, (int, np.integer))
        or not 1 <= width <= len(wave)
    ):
        raise ValueError("width must be an integer from 1 to signal length.")
    return np.convolve(wave, np.ones(width) / width, mode="valid")


def generate_sine_wave(
    frequency: float, duration: float = 1.0, sample_rate: int = 8000, phase: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Carrier sin(2*pi*f*t + phase); reference carrier phase is zero."""
    count = _sample_count(duration, sample_rate)
    if (
        not np.isfinite(frequency)
        or not 0 < frequency < sample_rate / 2
        or not np.isfinite(phase)
    ):
        raise ValueError("Require finite phase and 0 < frequency < Nyquist.")
    time = np.arange(count) / sample_rate
    return time, np.sin(2 * np.pi * frequency * time + phase)


def asymmetric_gaussian(
    num_samples: int, std_pos: float, neg_ratio: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 4: widths refer to coordinate [-10,10], NOT seconds."""
    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, (int, np.integer))
        or num_samples < 2
    ):
        raise ValueError("num_samples must be an integer >= 2.")
    if (
        not np.isfinite(std_pos)
        or not np.isfinite(neg_ratio)
        or min(std_pos, neg_ratio) <= 0
    ):
        raise ValueError("Envelope width and ratio must be positive and finite.")
    coordinate = np.linspace(-10, 10, num_samples)
    std = np.where(coordinate < 0, std_pos * neg_ratio, std_pos)
    envelope = np.exp(-0.5 * (coordinate / std) ** 2)
    envelope /= std_pos * np.sqrt(2 * np.pi)
    return coordinate, envelope


def generate_heart_sound(
    frequency: float,
    peak_height: float,
    envelope_width: float,
    shift_ratio: float,
    duration: float = 1.0,
    sample_rate: int = 8000,
    neg_ratio: float = 0.25,
    phase: float = 0.0,
    random_envelope_shift: bool = True,
    start_threshold: float = 1e-4,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    return_unshifted: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Algorithm 3: envelope/carrier, threshold alignment, scaling, placement.

    The reference randomly shifts the envelope by +/-5% of the beat before
    multiplying by the carrier. The placement ratio locates the aligned onset,
    not the envelope peak. Return the aligned pre-placement burst when needed
    to reproduce the reference AS timing rule.
    """
    rng = _get_rng(rng, seed)
    if not np.isfinite(peak_height) or peak_height <= 0 or not 0 <= shift_ratio < 1:
        raise ValueError("Require positive finite height and 0 <= shift_ratio < 1.")
    if not np.isfinite(start_threshold) or start_threshold <= 0:
        raise ValueError("start_threshold must be positive and finite.")
    _, carrier = generate_sine_wave(frequency, duration, sample_rate, phase)
    _, envelope = asymmetric_gaussian(len(carrier), envelope_width, neg_ratio)
    if random_envelope_shift:
        direction = 1 if rng.choice([True, False]) else -1
        envelope = shift_wave(envelope, direction * int(0.05 * len(envelope)))
    burst = carrier * envelope
    active = np.flatnonzero(np.abs(burst) > start_threshold)
    if not len(active):
        raise ValueError("Burst never crosses the onset threshold.")
    burst = shift_wave(burst, -int(active[0]))
    burst *= peak_height / np.max(np.abs(burst))
    positioned = shift_wave(burst, int(shift_ratio * len(burst)))
    return (positioned, burst) if return_unshifted else positioned


def generate_background_oscillations(
    duration: float = 1.0,
    sample_rate: int = 8000,
    amplitudes: tuple[float, ...] = (0.05, 0.15, 0.25),
    frequencies: tuple[float, ...] = (15.0, 12.0, 7.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Three low-frequency sine components; phases restart each beat."""
    count = _sample_count(duration, sample_rate)
    if len(amplitudes) != len(frequencies):
        raise ValueError("Each amplitude needs a frequency.")
    time = np.arange(count) / sample_rate
    background = np.zeros(count)
    for amplitude, frequency in zip(amplitudes, frequencies):
        if not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError("Background amplitudes must be nonnegative and finite.")
        _, carrier = generate_sine_wave(frequency, duration, sample_rate)
        background += amplitude * carrier
    return time, background


def generate_background_noise(
    duration: float = 1.0,
    sample_rate: int = 8000,
    widths: tuple[int, ...] = (12, 80),
    scales: tuple[float, ...] = (0.005, 0.010),
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two repeated-Gaussian noise processes, not independent samplewise noise."""
    rng = _get_rng(rng, seed)
    count = _sample_count(duration, sample_rate)
    if len(widths) != len(scales):
        raise ValueError("Each width needs a scale.")
    epsilon = np.zeros(count)
    for width, scale in zip(widths, scales):
        if (
            isinstance(width, bool)
            or not isinstance(width, (int, np.integer))
            or not 1 <= width <= count
        ):
            raise ValueError("Noise widths must be integers within the beat length.")
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("Noise scales must be finite and nonnegative.")
        values = rng.normal(0.0, scale, size=count // width)
        component = np.repeat(values, width)
        epsilon += np.pad(component, (0, count - len(component)))
    return np.arange(count) / sample_rate, epsilon


def generate_as_murmur(
    duration: float = 1.0,
    sample_rate: int = 8000,
    start_ratio: float = 0.40,
    end_ratio: float = 0.55,
    envelope_width: float = 3.0,
    neg_ratio: float = 0.5,
    min_amp: float = 0.05,
    max_amp: float = 2.75,
    smoothing_width: int = 18,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithms 6–7: shaped Gaussian noise, valid smoothing, placement.

    start_ratio=0.40 is only a standalone illustration. generate_beat below
    computes its start from S1. min_amp/max_amp are scale parameters, not
    hard noise bounds. No extra low-pass filter is enabled in reference AS.
    """
    rng = _get_rng(rng, seed)
    count = _sample_count(duration, sample_rate)
    if not 0 <= start_ratio < end_ratio <= 1:
        raise ValueError("Require 0 <= start_ratio < end_ratio <= 1.")
    if (
        not np.isfinite(min_amp)
        or not np.isfinite(max_amp)
        or not 0 <= min_amp < max_amp
    ):
        raise ValueError("Require finite 0 <= min_amp < max_amp.")
    length = int(sample_rate * duration * (end_ratio - start_ratio))
    _, envelope = asymmetric_gaussian(length, envelope_width, neg_ratio)
    noise = rng.normal(0.0, max_amp - min_amp, size=length)
    segment = moving_average(noise * (envelope + min_amp), smoothing_width)
    start = int(sample_rate * duration * start_ratio)
    if start + len(segment) > count:
        raise ValueError("Murmur extends beyond the beat.")
    murmur = np.zeros(count)
    murmur[start : start + len(segment)] = segment
    return np.arange(count) / sample_rate, murmur


# Primary Normal/AS settings used by recordings and batch generation.
SYNTHESIS_PROFILE = {
    "version": "normal_as_v1",
    "classes": {"Normal": 0, "AS": 1},
    "sample_rate": 8000,
    "duration_seconds": 10.0,
    "channels": 1,
    "dtype": "float32",
    "wav_subtype": "FLOAT",
    "parameter_configuration": "syn02",
    "as_timing": "reference",
    "random_generator": "numpy.default_rng/PCG64",
    "carrier_time_grid": "arange(int(sample_rate * beat_duration)) / sample_rate",
    "invalid_draw_policy": "bounded rejection with recorded rejection counts",
    "postprocessing": "minmax[-1,1] -> valid moving average -> random crop",
    "reference_repository": "https://github.com/seitalab/SynPCG",
    "reference_commit": "1a5f600ce2a0a9e43514e20582da67081159a14a",
    "reference_configuration_files": [
        "src/prep/syn/resources/normal_pcg/syn02.yaml",
        "src/prep/syn/resources/as_pcg/syn02.yaml",
    ],
    "documented_differences": [
        "PCG64 seed sequences differ from the reference's legacy NumPy RNG.",
        "Invalid draws/beats are retried rather than dropping whole recordings.",
        "Physical sample times n/fs differ slightly from reference linspace "
        "when fs*beat_duration is noninteger.",
        "Final arrays and exported WAVs are float32; the reference "
        "computes/stores arrays at NumPy default precision.",
    ],
    "paper_dataset_targets_per_class": {"train": 5000, "validation": 1000},
}

# The reference timing may overlap placed S1. The optional positioned_s1 policy
# remains available only as a separately identified correction/extension.
SAMPLE_RATE = SYNTHESIS_PROFILE["sample_rate"]
RECORDING_DURATION = SYNTHESIS_PROFILE["duration_seconds"]
TIMING_POLICY = SYNTHESIS_PROFILE["as_timing"]
assert int(SAMPLE_RATE * RECORDING_DURATION) == 80000


def _spec(base: float, recording: tuple, beat: tuple) -> dict:
    return {"base": base, "recording": recording, "beat": beat}


# Distribution tuples are ('normal', STANDARD DEVIATION), or ('uniform', lo, hi).
# S1/S2 agree with paper Table 3. Other values come from published syn02 YAMLs.
COMMON_CONFIG = {
    "duration": _spec(1.0, ("normal", 0.15), ("normal", 0.005)),
    "s1_frequency": _spec(24.0, ("normal", 2.5), ("normal", 1.0)),
    "s1_width": _spec(0.75, ("uniform", -0.25, 0.25), ("normal", 0.05)),
    "s1_height": _spec(2.5, ("uniform", -0.5, 0.5), ("normal", 0.025)),
    "s1_ratio": _spec(0.25, ("normal", 0.05), ("normal", 0.005)),
    "s1_shift": _spec(0.20, ("normal", 0.05), ("normal", 0.005)),
    "s2_frequency": _spec(60.0, ("normal", 5.0), ("normal", 1.0)),
    "s2_width": _spec(0.25, ("uniform", -0.10, 0.10), ("normal", 0.025)),
    "s2_height": _spec(3.0, ("uniform", -0.5, 0.5), ("normal", 0.05)),
    "s2_ratio": _spec(0.25, ("normal", 0.05), ("normal", 0.005)),
    "s2_shift": _spec(0.55, ("normal", 0.075), ("normal", 0.005)),
    "background_amp1": _spec(0.05, ("uniform", -0.005, 0.005), ("normal", 0.001)),
    "background_freq1": _spec(15.0, ("normal", 1.5), ("normal", 0.5)),
    "background_amp2": _spec(0.15, ("uniform", -0.01, 0.01), ("normal", 0.002)),
    "background_freq2": _spec(12.0, ("normal", 0.5), ("normal", 0.075)),
    "background_amp3": _spec(0.25, ("uniform", -0.075, 0.075), ("normal", 0.01)),
    "background_freq3": _spec(7.0, ("uniform", -3.0, 3.0), ("normal", 0.1)),
    "noise_width1": _spec(12.0, ("uniform", -4.0, 4.0), ("normal", 0.5)),
    "noise_scale1": _spec(0.005, ("uniform", -0.004, 0.004), ("normal", 0.0005)),
    "noise_width2": _spec(80.0, ("uniform", -20.0, 20.0), ("normal", 0.5)),
    "noise_scale2": _spec(0.01, ("uniform", -0.006, 0.006), ("normal", 0.001)),
    "recording_ma_width": _spec(
        80.0, ("uniform", -5.0, 5.0), ("uniform", -0.0001, 0.0001)
    ),
}
AS_CONFIG = {
    # Published YAML has beat perturbation U(1,1), a constant +1, not U(-1,1).
    "murmur_ma_width": _spec(18.0, ("uniform", -6.0, 6.0), ("uniform", 1.0, 1.0)),
    "murmur_width": _spec(3.0, ("uniform", -0.5, 0.5), ("uniform", -0.05, 0.05)),
    "murmur_ratio": _spec(0.5, ("uniform", -0.01, 0.01), ("uniform", -0.001, 0.001)),
    "murmur_min": _spec(0.05, ("uniform", -0.005, 0.005), ("uniform", -0.0005, 0.0005)),
    "murmur_max": _spec(2.75, ("uniform", -0.5, 0.5), ("uniform", -0.1, 0.1)),
    "s1_end_threshold": _spec(
        0.001, ("uniform", -0.0001, 0.0001), ("uniform", -0.0001, 0.0001)
    ),
}


def _configuration(label: str) -> dict:
    if label not in SYNTHESIS_PROFILE["classes"]:
        raise ValueError("label must be 'Normal' or 'AS'.")
    return {**COMMON_CONFIG, **(AS_CONFIG if label == "AS" else {})}


def _draw(distribution: tuple, rng: np.random.Generator) -> float:
    kind, *values = distribution
    if kind == "normal":
        return float(rng.normal(0.0, values[0]))
    if kind == "uniform":
        return float(rng.uniform(*values))
    raise ValueError(f"Unknown distribution: {kind}")


def _valid_parameters(p: dict[str, float], sample_rate: int = 8000) -> bool:
    if not all(np.isfinite(v) and v > 0 for v in p.values()):
        return False
    if not 0 < p["s1_shift"] < p["s2_shift"] < 1:
        return False
    frequencies = [p[k] for k in p if "frequency" in k or "background_freq" in k]
    if max(frequencies) >= sample_rate / 2:
        return False
    count = int(p["duration"] * sample_rate)
    if count < 2 or any(
        not 1 <= int(p[k]) <= count
        for k in ("noise_width1", "noise_width2", "recording_ma_width")
    ):
        return False
    if "murmur_min" in p and not p["murmur_min"] < p["murmur_max"]:
        return False
    return True


def sample_recording_parameters(
    label: str,
    rng: np.random.Generator,
    sample_rate: int = 8000,
    max_attempts: int = 100,
) -> tuple[dict[str, float], int]:
    """Algorithm 12. Rejection of invalid draws is a documented adaptation."""
    config = _configuration(label)
    for attempt in range(max_attempts):
        p = {
            key: spec["base"] + _draw(spec["recording"], rng)
            for key, spec in config.items()
        }
        if _valid_parameters(p, sample_rate):
            return p, attempt
    raise ValueError("Could not sample valid recording parameters.")


def sample_beat_parameters(
    recording_parameters: dict[str, float],
    label: str,
    rng: np.random.Generator,
    sample_rate: int = 8000,
    max_attempts: int = 100,
) -> tuple[dict[str, float], int]:
    """Algorithm 13. Always perturb the recording base, not the previous beat."""
    config = _configuration(label)
    for attempt in range(max_attempts):
        p = {
            key: recording_parameters[key] + _draw(spec["beat"], rng)
            for key, spec in config.items()
        }
        if _valid_parameters(p, sample_rate):
            return p, attempt
    raise ValueError("Could not sample valid beat parameters.")


def generate_beat(
    label: str,
    parameters: dict[str, float],
    rng: np.random.Generator,
    sample_rate: int = 8000,
    timing: str = "reference",
) -> dict:
    """Algorithms 2/5; return the raw beat and its components for inspection."""
    _configuration(label)
    if timing not in ("reference", "positioned_s1"):
        raise ValueError("Unknown AS timing policy.")
    p = parameters
    sounds = {}
    aligned_s1 = None
    for name in ("s1", "s2"):
        sound, aligned = generate_heart_sound(
            p[f"{name}_frequency"],
            p[f"{name}_height"],
            p[f"{name}_width"],
            p[f"{name}_shift"],
            duration=p["duration"],
            sample_rate=sample_rate,
            neg_ratio=p[f"{name}_ratio"],
            rng=rng,
            return_unshifted=True,
        )
        sounds[name] = sound
        if name == "s1":
            aligned_s1 = aligned
    count = len(sounds["s1"])
    murmur = np.zeros(count)
    window = None
    if label == "AS":
        # Literal published timing uses aligned S1 BEFORE its placement shift.
        timing_sound = aligned_s1 if timing == "reference" else sounds["s1"]
        active = np.flatnonzero(np.abs(timing_sound) > p["s1_end_threshold"])
        if not len(active):
            raise ValueError("No S1 threshold crossing for murmur placement.")
        start_ratio = float(active[-1] / count)
        end_ratio = p["s2_shift"]
        _, murmur = generate_as_murmur(
            p["duration"],
            sample_rate,
            start_ratio,
            end_ratio,
            p["murmur_width"],
            p["murmur_ratio"],
            p["murmur_min"],
            p["murmur_max"],
            int(p["murmur_ma_width"]),
            rng=rng,
        )
        window = (start_ratio * p["duration"], end_ratio * p["duration"])
    time, background = generate_background_oscillations(
        p["duration"],
        sample_rate,
        tuple(p[f"background_amp{i}"] for i in (1, 2, 3)),
        tuple(p[f"background_freq{i}"] for i in (1, 2, 3)),
    )
    _, noise = generate_background_noise(
        p["duration"],
        sample_rate,
        tuple(int(p[f"noise_width{i}"]) for i in (1, 2)),
        tuple(p[f"noise_scale{i}"] for i in (1, 2)),
        rng=rng,
    )
    baseline = sounds["s1"] + sounds["s2"] + background + noise
    waveform = baseline + murmur
    if not np.isfinite(waveform).all():
        raise ValueError("Nonfinite generated waveform.")
    return {
        "time": time,
        **sounds,
        "background": background,
        "noise": noise,
        "murmur": murmur,
        "baseline": baseline,
        "waveform": waveform,
        "murmur_window": window,
        "parameters": dict(p),
    }


def generate_recording(
    label: str,
    seed: int,
    duration: float | None = None,
    sample_rate: int | None = None,
    timing: str | None = None,
    max_attempts: int = 100,
) -> tuple[np.ndarray, dict]:
    """Algorithm 1 with reference-code scaling -> valid MA -> random crop.

    Produces >1.5 target lengths before postprocessing. Invalid parameter/beat
    draws are rejected with bounded retries and counts retained in metadata.
    Output is float32. No per-recording z-score or additional peak scaling is
    applied; model-side z-scoring remains a separate training decision.
    """
    duration = RECORDING_DURATION if duration is None else duration
    sample_rate = SAMPLE_RATE if sample_rate is None else sample_rate
    timing = TIMING_POLICY if timing is None else timing
    target = _sample_count(duration, sample_rate)
    rng = np.random.default_rng(seed)
    base, rejected_recordings = sample_recording_parameters(label, rng, sample_rate)
    width = int(base["recording_ma_width"])
    pieces = [np.zeros(1)]  # Retain the reference's initial zero sample.
    length, beats, rejected_beats, parameter_rejections = 1, 0, 0, 0
    # Extra floor also guarantees a valid crop for unusually short requests.
    while length <= max(1.5 * target, target + width):
        for _ in range(max_attempts):
            p, rejected = sample_beat_parameters(base, label, rng, sample_rate)
            parameter_rejections += rejected
            try:
                beat = generate_beat(label, p, rng, sample_rate, timing)
                break
            except ValueError:
                rejected_beats += 1
        else:
            raise ValueError("Failed to generate a valid beat after bounded retries.")
        pieces.append(beat["waveform"])
        length += len(beat["waveform"])
        beats += 1
    raw = np.concatenate(pieces)
    span = np.ptp(raw)
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Cannot normalize a constant/nonfinite recording.")
    scaled = 2 * (raw - raw.min()) / span - 1
    filtered = moving_average(scaled, width)
    # Match the reference's exclusive upper bound in np.random.choice.
    crop_start = int(rng.integers(0, len(filtered) - target))
    waveform = filtered[crop_start : crop_start + target].astype(np.float32)
    if waveform.shape != (target,) or not np.isfinite(waveform).all():
        raise ValueError("Invalid final recording.")
    metadata = {
        "label": label,
        "binary_label": SYNTHESIS_PROFILE["classes"][label],
        "seed": int(seed),
        "synthesis_profile": copy.deepcopy(SYNTHESIS_PROFILE),
        "effective_settings": {
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "as_timing": timing,
            "samples": target,
        },
        "sample_rate": sample_rate,
        "samples": target,
        "duration": duration,
        "timing": timing,
        "generated_beats": beats,
        "raw_samples": len(raw),
        "crop_start": crop_start,
        "recording_ma_width": width,
        "recording_parameters": base,
        "rejected_recording_draws": rejected_recordings,
        "rejected_beat_parameter_draws": parameter_rejections,
        "rejected_beats": rejected_beats,
        "postprocessing": "minmax[-1,1] -> valid moving average -> random crop",
        "rng": "numpy.default_rng/PCG64; not legacy-reference bitwise parity",
        "numpy_version": np.__version__,
        "configuration": copy.deepcopy(_configuration(label)),
    }
    return waveform, metadata


def save_examples(
    examples: list[tuple[np.ndarray, dict]], output_dir: str | Path
) -> Path:
    """Write float WAVs and a JSON manifest without replacing different audio.

    Each tuple is (waveform, metadata). These are demonstration examples only;
    partition-specific seeds/IDs for a full dataset are a later step.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    pending = []
    for waveform, metadata in examples:
        name = (
            f"{metadata['label'].lower()}_seed{metadata['seed']}_"
            f"{metadata['timing']}.wav"
        )
        path = output_dir / name
        if path.exists():
            previous, sr = sf.read(path, dtype="float32")
            if sr != metadata["sample_rate"] or not np.array_equal(previous, waveform):
                raise FileExistsError(f"Different audio already exists: {path}")
        pending.append((path, waveform, metadata["sample_rate"]))
        manifest.append({"filename": name, **metadata})
    manifest_path = output_dir / "examples_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(
        json.dumps(manifest)
    ):
        raise FileExistsError("Different manifest exists; use a new output directory.")
    for path, waveform, sr in pending:
        if not path.exists():
            sf.write(path, waveform, sr, subtype=SYNTHESIS_PROFILE["wav_subtype"])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def generate_dataset(
    output_dir: str | Path,
    train_per_class: int = 5000,
    validation_per_class: int = 1000,
    master_seed: int = 42,
    duration: float = RECORDING_DURATION,
    sample_rate: int = SAMPLE_RATE,
    timing: str = TIMING_POLICY,
    max_recording_attempts: int = 3,
    progress_every: int = 100,
) -> Path:
    """Generate Normal/AS training and validation WAVs, one recording at a time.

    Defaults produce the paper's 12,000 recordings: 5,000 training and 1,000
    validation examples per class. Use small counts for a pilot. Existing
    nonempty output folders are rejected; interrupted batches cannot resume yet.

    Args:
        output_dir: Empty folder for WAVs, manifests, and generation settings.
        train_per_class: Number of training recordings for each class.
        validation_per_class: Number of validation recordings for each class.
        master_seed: Nonnegative seed controlling the entire dataset.
        duration: Recording duration in seconds; paper default is 10.
        sample_rate: WAV sampling rate; paper default is 8,000 Hz.
        timing: Reference AS timing or the positioned_s1 alternative.
        max_recording_attempts: Bounded retries for an unsuccessful recording.
        progress_every: Print progress after this many files; zero disables it.

    Returns:
        Path to manifest.csv. Relative signal paths are based on output_dir.

    Examples:
        >>> manifest = generate_dataset(  # doctest: +SKIP
        ...     "synthetic_pilot", train_per_class=10, validation_per_class=2
        ... )
    """
    for name, value, minimum in (
        ("train_per_class", train_per_class, 0),
        ("validation_per_class", validation_per_class, 0),
        ("master_seed", master_seed, 0),
        ("max_recording_attempts", max_recording_attempts, 1),
        ("progress_every", progress_every, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    total = 2 * (train_per_class + validation_per_class)
    if total == 0:
        raise ValueError("Request at least one recording per class overall.")
    samples = _sample_count(duration, sample_rate)
    if timing not in ("reference", "positioned_s1"):
        raise ValueError("Unknown AS timing policy.")

    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Use a new or empty output folder for this batch.")
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "status": "running",
        "master_seed": master_seed,
        "counts_per_class": {
            "train": train_per_class,
            "validation": validation_per_class,
        },
        "requested_total": total,
        "completed_total": 0,
        "failed_attempts": 0,
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "samples_per_recording": samples,
        "timing": timing,
        "max_recording_attempts": max_recording_attempts,
        "synthesis_profile": copy.deepcopy(SYNTHESIS_PROFILE),
        "parameter_configuration": {
            label: copy.deepcopy(_configuration(label))
            for label in SYNTHESIS_PROFILE["classes"]
        },
        "numpy_version": np.__version__,
        "soundfile_version": sf.__version__,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed_policy": "SeedSequence([master_seed, split, class, index, attempt])",
    }
    settings_path = output_dir / "dataset.json"
    manifest_path = output_dir / "manifest.csv"
    fields = [
        "recording_id",
        "patient_id",
        "split",
        "class_name",
        "label",
        "signal",
        "seed",
        "sample_rate",
        "samples",
        "duration_seconds",
        "sha256",
    ]
    used_seeds = set()
    started = time.perf_counter()

    def save_status():
        settings["elapsed_seconds"] = time.perf_counter() - started
        settings_path.write_text(json.dumps(settings, indent=2))

    save_status()
    try:
        with (
            manifest_path.open("w", newline="") as csv_file,
            (output_dir / "recording_metadata.jsonl").open("w") as metadata_file,
            (output_dir / "generation_failures.jsonl").open("w") as failures,
        ):
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            partitions = (
                ("train", train_per_class),
                ("validation", validation_per_class),
            )
            for split_number, (split, count) in enumerate(partitions):
                for class_name, binary_label in SYNTHESIS_PROFILE["classes"].items():
                    folder = output_dir / split / class_name.lower()
                    folder.mkdir(parents=True, exist_ok=True)
                    for index in range(count):
                        recording_id = f"syn_{split}_{class_name.lower()}_{index:06d}"
                        # Each item and retry has its own deterministic random stream.
                        for attempt in range(max_recording_attempts):
                            sequence = np.random.SeedSequence(
                                [
                                    master_seed,
                                    split_number,
                                    binary_label,
                                    index,
                                    attempt,
                                ]
                            )
                            seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
                            if seed in used_seeds:
                                raise RuntimeError(
                                    "Seed collision; choose another master seed."
                                )
                            used_seeds.add(seed)
                            try:
                                waveform, metadata = generate_recording(
                                    class_name, seed, duration, sample_rate, timing
                                )
                            except ValueError as error:
                                settings["failed_attempts"] += 1
                                failures.write(
                                    json.dumps(
                                        {
                                            "recording_id": recording_id,
                                            "seed": seed,
                                            "attempt": attempt + 1,
                                            "error": str(error),
                                        }
                                    )
                                    + "\n"
                                )
                                continue
                            break
                        else:
                            raise RuntimeError(
                                f"Could not generate {recording_id} after "
                                f"{max_recording_attempts} attempts."
                            )

                        if (
                            waveform.shape != (samples,)
                            or not np.isfinite(waveform).all()
                            or np.max(np.abs(waveform)) > 1.000001
                        ):
                            raise ValueError(f"Invalid waveform: {recording_id}")
                        path = folder / f"{recording_id}.wav"
                        sf.write(
                            path,
                            waveform,
                            sample_rate,
                            subtype=SYNTHESIS_PROFILE["wav_subtype"],
                        )
                        info = sf.info(path)
                        if (
                            info.frames != samples
                            or info.samplerate != sample_rate
                            or info.channels != 1
                        ):
                            raise ValueError(f"Invalid saved WAV: {path}")
                        writer.writerow(
                            {
                                "recording_id": recording_id,
                                "patient_id": recording_id,
                                "split": split,
                                "class_name": class_name,
                                "label": binary_label,
                                "signal": path.relative_to(output_dir).as_posix(),
                                "seed": seed,
                                "sample_rate": sample_rate,
                                "samples": samples,
                                "duration_seconds": duration,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        )
                        # Shared config is in dataset.json; keep per-file metadata small.
                        metadata.pop("synthesis_profile")
                        metadata.pop("configuration")
                        metadata_file.write(
                            json.dumps(
                                {
                                    "recording_id": recording_id,
                                    **metadata,
                                }
                            )
                            + "\n"
                        )
                        settings["completed_total"] += 1
                        if (
                            progress_every
                            and settings["completed_total"] % progress_every == 0
                        ):
                            csv_file.flush()
                            metadata_file.flush()
                            failures.flush()
                            save_status()
                            print(f"Generated {settings['completed_total']}/{total}")
        settings["status"] = "complete"
    except BaseException as error:
        settings["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        settings["error"] = str(error)
        raise
    finally:
        save_status()

    if progress_every:
        print(f"Complete: {total} recordings. Manifest: {manifest_path}")
    return manifest_path
