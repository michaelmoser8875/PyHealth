"""ResNet1D model and Trainer tests using random waveforms."""

import io
import tempfile
import unittest

import torch
import torch.nn.functional as F

from pyhealth.models import ResNet1D
from pyhealth.trainer import Trainer
from pyhealth.datasets import get_dataloader
from examples.resnet1d.ablation import check_splits, smoke_datasets
from examples.resnet1d.ablation import run_ablation


class TestResNet1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        cls.train, cls.val, cls.test = smoke_datasets()

    def model(self, **kwargs):
        config = dict(layers=(1, 1), base_width=4)
        config.update(kwargs)
        return ResNet1D(self.train, **config)

    def test_default_and_custom(self):
        self.assertEqual(len(ResNet1D(self.train).backbone.blocks), 8)
        self.assertEqual(len(self.model(layers=(3, 4, 6, 3)).backbone.blocks), 16)
        model = self.model(in_channels=2)
        self.assertEqual(model(signal=torch.randn(2, 2, 128))["logit"].shape, (2, 1))

    def test_forward_backward(self):
        for shape in ((2, 128), (2, 1, 128)):
            model = self.model()
            out = model(signal=torch.randn(*shape), label=torch.tensor([0.0, 1.0]))
            self.assertEqual(set(out), {"logit", "y_prob", "loss", "y_true"})
            for key in ("logit", "y_prob", "y_true"):
                self.assertEqual(out[key].shape, (2, 1))
            self.assertTrue(torch.isfinite(out["loss"]))
            self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())
            torch.testing.assert_close(out["y_prob"], torch.sigmoid(out["logit"]))
            torch.testing.assert_close(
                out["loss"],
                F.binary_cross_entropy_with_logits(out["logit"], out["y_true"]),
            )
            out["loss"].backward()
            grad = model.backbone.stem[0].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_weight_and_normalization(self):
        model = self.model(pos_weight=3, normalize=True)
        out = model(signal=torch.ones(2, 1, 128), label=torch.ones(2, 1))
        expected = F.binary_cross_entropy_with_logits(
            out["logit"], out["y_true"], pos_weight=torch.tensor([3.0])
        )
        torch.testing.assert_close(out["loss"], expected)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(model.mode, "binary")

    def test_invalid(self):
        for config in (
            dict(layers=()),
            dict(layers=(0,)),
            dict(base_width=0),
            dict(kernel_size=2),
            dict(dropout=1),
            dict(in_channels=0),
            dict(pos_weight=-1),
            dict(pos_weight=float("nan")),
            dict(normalize="yes"),
            dict(feature_key="bad"),
            dict(label_key="bad"),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.model(**config)
        for x in (
            torch.zeros(128),
            torch.zeros(2, 2, 128),
            torch.zeros(2, 1, 1, 128),
            torch.zeros(2, 1, 0),
        ):
            with self.assertRaises(ValueError):
                self.model()(signal=x)

    def test_checkpoint_transfer(self):
        source = self.model().eval()
        x = torch.randn(2, 1, 128)
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)
        weights = torch.load(buffer, weights_only=True)
        target = self.model().eval()
        target.load_state_dict(weights)
        torch.testing.assert_close(source(signal=x)["logit"], target(signal=x)["logit"])
        target = self.model(pos_weight=2).eval()
        head = target.fc.weight.clone()
        target.load_pretrained(weights, retain_head=False)
        torch.testing.assert_close(head, target.fc.weight)
        torch.testing.assert_close(
            source.backbone.stem[0].weight, target.backbone.stem[0].weight
        )
        target.load_pretrained(weights)
        torch.testing.assert_close(source(signal=x)["logit"], target(signal=x)["logit"])
        self.assertEqual(target.pos_weight.item(), 2)
        unweighted = self.model().eval()
        unweighted.load_pretrained(target.state_dict())
        self.assertIsNone(unweighted.pos_weight)
        with self.assertRaises(RuntimeError):
            self.model(base_width=8).load_pretrained(weights)
        malformed = dict(weights)
        del malformed["backbone.stem.0.weight"]
        with self.assertRaises(RuntimeError):
            self.model().load_pretrained(malformed)
        target.train()
        optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)
        before = target.backbone.stem[0].weight.detach().clone()
        optimizer.zero_grad()
        target(signal=x, label=torch.tensor([0.0, 1.0]))["loss"].backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, target.backbone.stem[0].weight))

    def test_production_shape(self):
        with torch.no_grad():
            out = ResNet1D(self.train).eval()(signal=torch.zeros(1, 1, 80000))
        self.assertEqual(out["y_prob"].shape, (1, 1))
        self.assertNotIn("loss", out)

    def test_trainer(self):
        model = self.model()
        trainer = Trainer(
            model=model,
            device="cpu",
            enable_logging=False,
            metrics=["roc_auc", "pr_auc", "f1"],
        )
        loader = get_dataloader(self.train, batch_size=2, shuffle=False)
        before = model.fc.weight.clone()
        trainer.train(loader, epochs=1)
        self.assertFalse(torch.equal(before, model.fc.weight))
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(v))
                for v in trainer.evaluate(loader).values()
            )
        )

    def test_split_guard(self):
        check_splits(self.train, self.val, self.test)
        with self.assertRaises(ValueError):
            check_splits(self.train, self.train, self.test)
        for patient_id in (None, "", "   ", float("nan"), "None"):
            with self.assertRaises(ValueError):
                check_splits([{"patient_id": patient_id}], self.val, self.test)

    def test_normalization_values(self):
        model = self.model(normalize=True).eval()
        captured = []
        handle = model.backbone.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        x = torch.stack((torch.arange(128.0), torch.arange(128.0) * 3 + 5))
        try:
            model(signal=x)
        finally:
            handle.remove()
        actual = captured[0]
        torch.testing.assert_close(
            actual.mean((1, 2)), torch.zeros(2), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(actual.std((1, 2), correction=0), torch.ones(2))
        torch.testing.assert_close(actual[0], actual[1])

    def test_ablation_processor_guard(self):
        import copy
        from examples.resnet1d.ablation import _same_state

        state = {"weight": torch.tensor([2.0])}
        self.assertTrue(_same_state(state, copy.deepcopy(state)))

        validation = copy.copy(self.val)
        validation.output_processors = copy.deepcopy(self.val.output_processors)
        validation.output_processors["label"].label_vocab = {0: 1, 1: 0}
        with tempfile.TemporaryDirectory() as output, self.assertRaises(ValueError):
            run_ablation(self.train, validation, self.test, output, smoke=True)

    def test_ablation(self):
        with tempfile.TemporaryDirectory() as output:
            results = run_ablation(
                self.train,
                self.val,
                self.test,
                output,
                smoke=True,
                seeds=(0,),
                epochs=1,
                batch_size=2,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(
                set(results[0]["validation"]), {"roc_auc", "pr_auc", "f1", "loss"}
            )
            self.assertNotIn("test", results[0])
            with self.assertRaises(ValueError):
                run_ablation(self.train, self.val, self.test, output, smoke=True)

    def test_task_schema(self):
        from pyhealth.tasks import BMDHSBinaryClassification

        task = BMDHSBinaryClassification()
        self.assertEqual(
            task.input_schema["signal"][1], {"sample_rate": 8000, "duration": 10.0}
        )
        self.assertEqual(task.output_schema, self.train.output_schema)

    def test_processor_weight(self):
        from pyhealth.datasets import create_sample_dataset
        from pyhealth.processors import BinaryLabelProcessor

        samples = [{"signal": [[0.0] * 128], "target": i % 2} for i in range(4)]
        data = create_sample_dataset(
            samples,
            {"signal": "tensor"},
            {"target": (BinaryLabelProcessor, {"pos_weight": 4.0})},
        )
        model = ResNet1D(data, label_key="target", layers=(1,), base_width=4)
        out = model(signal=torch.zeros(2, 128), target=torch.ones(2, 1))
        torch.testing.assert_close(
            out["loss"],
            F.binary_cross_entropy_with_logits(
                out["logit"], out["y_true"], pos_weight=torch.tensor([4.0])
            ),
        )
        override = ResNet1D(
            data, label_key="target", layers=(1,), base_width=4, pos_weight=2
        )
        out = override(signal=torch.zeros(2, 128), target=torch.ones(2, 1))
        torch.testing.assert_close(
            out["loss"],
            F.binary_cross_entropy_with_logits(
                out["logit"], out["y_true"], pos_weight=torch.tensor([2.0])
            ),
        )


if __name__ == "__main__":
    unittest.main()
