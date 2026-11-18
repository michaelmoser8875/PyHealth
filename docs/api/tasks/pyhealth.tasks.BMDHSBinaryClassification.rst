pyhealth.tasks.BMDHSBinaryClassification
========================================

Binary Normal-vs-disease classification of heart sound recordings from the
BMD-HS dataset, as used in Nonaka et al., "Boosting Phonocardiogram
Classification Performance with Function Generated Data" (ML4H 2025,
`PMLR 297 <https://proceedings.mlr.press/v297/nonaka26a.html>`_). Each
recording becomes one sample, labelled 1 for the chosen valve disease (AS, AR,
MR or MS) and 0 for Normal, and is resampled to 8 kHz and cropped to 10 s to
match the paper's synthetic phonocardiograms.

.. autoclass:: pyhealth.tasks.BMDHSBinaryClassification
    :members:
    :show-inheritance:
