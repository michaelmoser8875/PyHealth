pyhealth.models.ResNet1D
===================================

Binary raw-waveform classifier using the existing ``ResNet1dBackbone``.
Research context: `Nonaka et al., ML4H 2025 (PMLR 2026)
<https://proceedings.mlr.press/v297/nonaka26a.html>`_.

.. autoclass:: pyhealth.models.ResNet1D
    :members:
    :show-inheritance:

Training
--------

Use Group 1's patient-separated datasets with training-fitted processors reused
for validation and test. Select ``BMDHSBinaryClassification`` explicitly;
BMD-HS's default task is multilabel. Its processed ``signal`` is (1,80000),
with binary ``label``. ResNet1D accepts (B,T) or (B,C,T) and does not resample.

.. code-block:: python

    from pyhealth.models import ResNet1D
    from pyhealth.datasets import get_dataloader
    from pyhealth.trainer import Trainer

    model = ResNet1D(train_dataset, normalize=True)
    trainer = Trainer(model=model, device="cpu",
                      metrics=["roc_auc", "pr_auc", "f1"])
    trainer.train(
        get_dataloader(train_dataset, 8, shuffle=True),
        get_dataloader(validation_dataset, 8, shuffle=False),
        optimizer_params={"lr": 1e-4}, epochs=20, monitor="roc_auc",
    )

Forward returns (B,1) ``logit`` and sigmoid ``y_prob``. With labels, it also
returns (B,1) ``y_true`` and scalar BCE ``loss``. Call the model directly for
label-free inference; Trainer.evaluate requires labels. ``pos_weight=None``
uses processor weighting; a supplied weight overrides it.

Checkpoint transfer
-------------------

.. code-block:: python

    import torch

    torch.save(model.state_dict(), "synthetic.pt")
    real_model = ResNet1D(real_train_dataset, normalize=True)
    weights = torch.load("synthetic.pt", map_location="cpu", weights_only=True)
    real_model.load_pretrained(weights, retain_head=True)
    # Use retain_head=False for a fresh classifier, e.g. for another disease.
    # Construct a new Trainer/optimizer before fine-tuning on real data.

Transfer requires matching backbone parameters and checks keys/shapes strictly.
Target-task loss weighting is retained. For same-task restoration, use
``load_state_dict`` or Trainer.load_ckpt with the original configuration,
including whether ``pos_weight`` is set. Save architecture, normalization,
and label semantics alongside weights; these are not encoded in a state dict.

Reference architecture
----------------------

The `authors' implementation
<https://github.com/seitalab/SynPCG/blob/main/src/common/model/resnet.py>`_
uses the same default BasicBlock topology: kernel-7 stride-2 stem, max pooling,
four residual stages with projection shortcuts, and adaptive global pooling.
Depths (2,2,2,2) and (3,4,6,3) correspond to ResNet-18/34-style models.

Differences: this wrapper keeps the existing backbone's constructor
initialization rather than explicit Kaiming-normal fan-out initialization,
accepts channels-first rather than channels-last 3D input, and uses one binary
logit rather than a configurable output projection. Width, dropout, and
normalization are experiment options. Reference checkpoints are not directly
compatible; matching topology does not verify the complete paper experiment.

See ``examples/resnet1d/README.md`` for ablations and compute requirements.
