"""CPU regressions: python -m unittest discover -s tests -v."""

import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

import train
from utils.evaluation import classification_metrics, fuse_logits, summarize_metrics
from utils.vcmm import VCMMAdam, VCMMController


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.labels = torch.tensor([0, 1, 2, 0, 1, 2])
        self.modalities = {
            name: (torch.randn(6, 3), torch.randn(6, width))
            for name, width in (("rgb", 4), ("flow", 5), ("depth", 7))
        }

    def test_probe_matches_autograd_and_does_not_populate_gradients(self):
        features = torch.randn(6, 4, requires_grad=True)
        classifier = nn.Linear(4, 3)
        logits = classifier(features)
        probe = VCMMController._probe_gradient(
            logits, features, self.labels, torch.arange(6)
        )
        self.assertFalse(probe.requires_grad)
        self.assertIsNone(classifier.weight.grad)
        self.assertIsNone(features.grad)
        grads = torch.autograd.grad(
            F.cross_entropy(logits, self.labels), (classifier.weight, classifier.bias)
        )
        torch.testing.assert_close(probe, torch.cat([g.flatten() for g in grads]))

    def test_noise_scale_for_equal_and_unequal_halves(self):
        controller = VCMMController()
        # All residuals are [-0.5, 0.5]; features alternate between 0 and 2.
        # Half-gradient disagreement has mean squared value 1/2.
        for size, expected in ((4, 1.0 / 8), (3, 1.0 / 9)):
            features = torch.tensor([[0.0], [2.0], [0.0], [2.0]])[:size]
            _, noise, _ = controller._measure(
                torch.zeros(size, 2), features, torch.zeros(size, dtype=torch.long)
            )
            self.assertAlmostEqual(noise, expected)

    def test_trimodal_centering_is_order_invariant(self):
        first = VCMMController(warmup_steps=0, adaptation_strength=0.1)
        second = VCMMController(warmup_steps=0, adaptation_strength=0.1)
        for _ in range(3):
            betas = first.update_modalities(self.modalities, self.labels)
            reversed_betas = second.update_modalities(
                dict(reversed(list(self.modalities.items()))), self.labels
            )
        self.assertEqual(set(betas), {"rgb", "flow", "depth"})
        for name in betas:
            self.assertAlmostEqual(betas[name], reversed_betas[name])
        logits = [math.log((1 - beta) / beta) for beta in betas.values()]
        self.assertAlmostEqual(sum(logits) / 3, math.log(0.1 / 0.9))

    def test_two_modality_interface_remains_compatible(self):
        legacy = VCMMController(warmup_steps=0)
        general = VCMMController(warmup_steps=0)
        image, text = self.modalities["rgb"], self.modalities["flow"]
        for _ in range(3):
            expected = legacy.update(*image, *text, self.labels)
            actual = general.update_modalities({"image": image, "text": text}, self.labels)
            self.assertEqual(actual, expected)

    def test_warmup_and_zero_adaptation_keep_base_momentum(self):
        warm = VCMMController(warmup_steps=2)
        fixed = VCMMController(warmup_steps=0, adaptation_strength=0)
        expected = dict.fromkeys(self.modalities, 0.9)
        for _ in range(2):
            self.assertEqual(warm.update_modalities(self.modalities, self.labels), expected)
        self.assertTrue(all(state["probe"].numel() for state in warm.state.values()))
        for _ in range(3):
            self.assertEqual(fixed.update_modalities(self.modalities, self.labels), expected)

    def test_changing_modalities_is_rejected_before_state_changes(self):
        controller = VCMMController()
        controller.update_modalities(self.modalities, self.labels)
        with self.assertRaisesRegex(ValueError, "remain fixed"):
            controller.update_modalities(dict(list(self.modalities.items())[:2]), self.labels)
        self.assertEqual(controller.steps, 1)


class OptimizerTests(unittest.TestCase):
    def test_fixed_momentum_matches_torch_adam_with_weight_decay(self):
        actual = nn.Parameter(torch.tensor([0.3, -0.7], dtype=torch.float64))
        reference = nn.Parameter(actual.detach().clone())
        options = dict(lr=0.01, betas=(0.9, 0.99), weight_decay=0.02)
        optimizer = VCMMAdam([actual], **options)
        adam = torch.optim.Adam([reference], foreach=False, **options)
        for values in ((0.2, -0.4), (-0.1, 0.8), (0.3, 0.2)):
            actual.grad = torch.tensor(values, dtype=torch.float64)
            reference.grad = actual.grad.clone()
            optimizer.step()
            adam.step()
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)

    def test_time_varying_correction_preserves_constant_gradient(self):
        modal = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        shared = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        optimizer = VCMMAdam(
            [{"params": [modal], "modality": "depth"}, {"params": [shared]}],
            weight_decay=0,
        )
        gradient = torch.tensor([0.2, -0.4], dtype=torch.float64)
        for beta in (0.7, 0.99, 0.8, 0.95):
            modal.grad, shared.grad = gradient.clone(), gradient.clone()
            optimizer.set_modal_betas({"depth": beta})
            optimizer.step()
            state = optimizer.state[modal]
            corrected = state["exp_avg"] / (1 - state["beta1_product"])
            torch.testing.assert_close(corrected, gradient, rtol=1e-12, atol=1e-12)
            self.assertEqual(optimizer.param_groups[1]["momentum_beta"], 0.9)


class EvaluationTests(unittest.TestCase):
    def test_macro_f1_on_known_confusion_matrix(self):
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        predictions = torch.tensor([0, 1, 1, 1, 0, 2])
        metrics = classification_metrics(F.one_hot(predictions, 3).float(), targets)
        self.assertAlmostEqual(metrics["acc"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)

    def test_absent_classes_have_zero_f1(self):
        metrics = classification_metrics(torch.tensor([[5.0, 0, 0]]), torch.tensor([0]))
        self.assertEqual(metrics["acc"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)

    def test_fusion_uses_equal_weights_and_backpropagates_to_each_modality(self):
        logits = [torch.randn(4, 3, requires_grad=True) for _ in range(3)]
        fused = fuse_logits(*logits)
        torch.testing.assert_close(fused, sum(logits) / 3)
        fused.sum().backward()
        for modal_logits in logits:
            torch.testing.assert_close(modal_logits.grad, torch.full_like(modal_logits, 1 / 3))

    def test_single_seed_has_no_fabricated_uncertainty(self):
        summary = summarize_metrics([{"test_acc": 0.8, "test_macro_f1": 0.7}])
        self.assertEqual(summary["test_acc_mean"], 0.8)
        self.assertIsNone(summary["test_acc_std"])
        self.assertIsNone(summary["test_macro_f1_ci95"])
        json.dumps(summary, allow_nan=False)

    def test_seed_summary_separates_sample_sd_from_ci(self):
        summary = summarize_metrics([
            {"test_acc": 0.6, "test_macro_f1": 0.4},
            {"test_acc": 0.8, "test_macro_f1": 0.6},
        ])
        self.assertAlmostEqual(summary["test_acc_mean"], 0.7)
        self.assertAlmostEqual(summary["test_macro_f1_mean"], 0.5)
        self.assertAlmostEqual(summary["test_acc_std"], math.sqrt(0.02))
        self.assertAlmostEqual(summary["test_macro_f1_ci95"], 0.196)


class TrainingTests(unittest.TestCase):
    def test_training_checkpoint_and_test_metrics_use_same_fusion(self):
        class TinyModel(nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.image_encoder = nn.Linear(4, 4)
                self.text_encoder = nn.Embedding(12, 4)
                self.image_classifier = nn.Linear(4, 3)
                self.text_classifier = nn.Linear(4, 3)

            def forward(self, images, text_inputs):
                image = self.image_encoder(images)
                text = self.text_encoder(text_inputs["input_ids"])
                return self.image_classifier(image), self.text_classifier(text), image, text

        def tokenize(_, texts, max_tokens, device):
            return {"input_ids": torch.tensor([int(t) for t in texts], device=device)}

        module = types.ModuleType("model.multimodal_model")
        module.MultimodalModel = TinyModel
        config = json.loads((Path(__file__).resolve().parents[1] / "data/config.json").read_text())
        config.update(epochs=2, batch_size=6, num_workers=0, warmup_steps=0)
        args = types.SimpleNamespace(device="cpu", bert_model="unused", resnet_checkpoint=None)
        samples = [(torch.randn(4), str(i), i % 3) for i in range(12)]
        datasets = {name: samples for name in ("train", "dev", "test")}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"model.multimodal_model": module}
        ), patch.object(train, "tokenize", side_effect=tokenize), contextlib.redirect_stdout(io.StringIO()):
            output_dir = Path(directory)
            result = train.run_seed(42, config, args, datasets, None, output_dir)
            checkpoint = torch.load(output_dir / "seed_42_best.pt", map_location="cpu")
            best = max(result["epoch_train_acc"], key=lambda x: (x["dev_acc"], -x["epoch"]))
            self.assertEqual(result["best_epoch"], best["epoch"])
            self.assertEqual(checkpoint["fusion"], "equal_logits")
            self.assertNotIn("fusion_weight", checkpoint)
            self.assertNotIn("fusion_grid_points", config)
            self.assertIn("test_macro_f1", json.loads((output_dir / "seed_42.json").read_text()))
            model = TinyModel()
            model.load_state_dict(checkpoint["model"])
            loader = train.make_loader(samples, 6, 0, False, 42)
            logits = train.collect_logits(model, loader, None, 30, torch.device("cpu"))
            expected = train.fused_metrics(*logits)
            self.assertAlmostEqual(result["test_acc"], expected["acc"])
            self.assertAlmostEqual(result["test_macro_f1"], expected["macro_f1"])


if __name__ == "__main__":
    unittest.main()
