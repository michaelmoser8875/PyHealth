"""
PyHealth task for binary Normal-vs-disease classification on BMD-HS.

Paper (replicated):
    Naoki Nonaka, Hiroshi Seki, Tomohiro Komatsu, Jun Seita. "Boosting
    Phonocardiogram Classification Performance with Function Generated Data."
    Proceedings of the Fifth Machine Learning for Health Symposium (ML4H),
    PMLR 297:473-524, 2026.

Paper link:
    https://proceedings.mlr.press/v297/nonaka26a.html

Dataset link:
    https://github.com/sani002/BMD-HS-Dataset

Authors:
    Franko Hysaj (fhysaj2@illinois.edu)
    Tejaswi Sapkota (sapkota3@illinois.edu)
    Michael Moser (mjmoser2@illinois.edu)

Description:
    Turns each BMD-HS heart sound recording into one binary sample, where
    label 1 is the chosen valve disease and label 0 is Normal, resampled to
    8 kHz and cropped to 10 s to match the paper's synthetic PCG data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class BMDHSBinaryClassification(BaseTask):
    """Binary Normal-vs-disease classification task for BMD-HS recordings.

    Works with :class:`~pyhealth.datasets.BMDHSDataset`.

    Emits one sample per recording, labelled 1 if the patient has ``disease``
    (even alongside other diseases) and 0 if Normal. Patients with only other
    diseases follow ``other_disease_policy``. Recordings are resampled to
    8 kHz and cropped to the first 10 s (SynPCG crops randomly in training
    and at the center in evaluation); missing recordings are skipped.

    Attributes:
        disease (str): The disease treated as the positive class.
        other_disease_policy (str): How patients with neither the chosen
            disease nor Normal are handled: ``"exclude"`` or ``"negative"``.
        positions (list): Recording slots (1-8) used for samples.

    Examples:
        >>> from pyhealth.datasets import BMDHSDataset
        >>> from pyhealth.tasks import BMDHSBinaryClassification
        >>> dataset = BMDHSDataset(root="/path/to/BMD-HS-Dataset")
        >>> task = BMDHSBinaryClassification(disease="AS")
        >>> samples = dataset.set_task(task)
    """

    DISEASES: List[str] = ["AS", "AR", "MR", "MS"]
    OTHER_DISEASE_POLICIES: List[str] = ["exclude", "negative"]
    NUM_POSITIONS: int = 8

    task_name: str = "BMDHSBinaryClassification"
    #: ``"signal"`` mapped to the audio processor, resampled to 8 kHz and
    #: cropped to 10 s.
    input_schema: Dict[str, Any] = {
        "signal": ("audio", {"sample_rate": 8000, "duration": 10.0}),
    }
    #: ``"label"`` mapped to ``"binary"``.
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        disease: str = "AS",
        other_disease_policy: str = "exclude",
        positions: Optional[List[int]] = None,
    ) -> None:
        """Initializes the task for one target disease.

        Args:
            disease: The disease used as the positive class. One of
                ``"AS"``, ``"AR"``, ``"MR"``, ``"MS"``.
            other_disease_policy: ``"exclude"`` drops patients that are
                neither the chosen disease nor Normal; ``"negative"`` labels
                them 0.
            positions: Recording slots (1-8) to emit samples for. ``None``
                uses all eight.

        Raises:
            ValueError: If ``disease``, ``other_disease_policy`` or
                ``positions`` is invalid.
        """
        if disease not in self.DISEASES:
            raise ValueError(
                f"Invalid disease: '{disease}'! Must be one of {self.DISEASES}."
            )
        if other_disease_policy not in self.OTHER_DISEASE_POLICIES:
            raise ValueError(
                f"Invalid other_disease_policy: '{other_disease_policy}'! "
                f"Must be one of {self.OTHER_DISEASE_POLICIES}."
            )
        all_positions = list(range(1, self.NUM_POSITIONS + 1))
        if positions is None:
            positions = all_positions
        elif not positions or any(p not in all_positions for p in positions):
            raise ValueError(
                f"Invalid positions: {positions}! Must be a non-empty list of "
                f"recording slots in {all_positions}."
            )
        super().__init__()
        self.disease = disease
        self.other_disease_policy = other_disease_policy
        self.positions = sorted(set(positions))

    @staticmethod
    def _flag(event: Event, key: str) -> bool:
        """Reads a 0/1 diagnosis flag, treating blank values as 0.

        Args:
            event: The patient's ``diagnoses`` event.
            key: The flag name, e.g. ``"AS"`` or ``"N"``.

        Returns:
            True if the flag is set.
        """
        value = event[key]
        if value is None or str(value).strip() == "":
            return False
        return int(float(value)) == 1

    def _label(self, diagnosis: Event) -> Optional[int]:
        """Maps a patient's diagnosis flags to a binary label.

        Args:
            diagnosis: The patient's ``diagnoses`` event, holding the
                AS/AR/MR/MS/N flags.

        Returns:
            1 if the chosen disease is present, 0 if Normal, otherwise 0 for
            ``other_disease_policy="negative"`` or ``None`` (exclude) for
            ``other_disease_policy="exclude"``. Patients whose flags are
            ``"N/A"`` (patient 22, mitral valve prolapse) are always excluded.
        """
        if str(diagnosis["N"]).strip().upper() == "N/A":
            return None
        if self._flag(diagnosis, self.disease):
            return 1
        if self._flag(diagnosis, "N"):
            return 0
        if self.other_disease_policy == "negative":
            return 0
        return None

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Generates one binary sample per recording for a single patient.

        Args:
            patient: A patient with ``diagnoses`` and ``recordings`` events.

        Returns:
            A list of dictionaries, one per kept recording, each with:
                - ``"patient_id"``: the patient identifier, for patient-level
                  splits.
                - ``"signal"``: path to the ``.wav`` recording.
                - ``"label"``: the binary label.
            The list is empty if the patient is excluded by
            ``other_disease_policy``.
        """
        diagnoses = patient.get_events(event_type="diagnoses")
        recordings = patient.get_events(event_type="recordings")
        if not diagnoses or not recordings:
            return []

        label = self._label(diagnoses[0])
        if label is None:
            return []

        samples = []
        skipped = 0
        for position in self.positions:
            path = recordings[0][f"recording_{position}"]
            # BMDHSDataset builds "<root>/train/<name>.wav"; a blank slot is
            # either null or a path ending in "/.wav".
            if not path or Path(path).name == ".wav" or not Path(path).is_file():
                skipped += 1
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "signal": path,
                    "label": label,
                }
            )

        if skipped:
            logger.info(
                "Patient %s: skipped %d missing recording(s)",
                patient.patient_id,
                skipped,
            )
        return samples
