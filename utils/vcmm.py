"""Variance-Calibrated Modal Momentum exactly as defined in the manuscript."""

import math

import torch
from torch.optim import Optimizer


def _logit(value, epsilon=1e-12):
    value = min(max(float(value), epsilon), 1.0 - epsilon)
    return math.log(value) - math.log1p(-value)


def _sigmoid(value):
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class VCMMController:
    """Online R/Q estimation and centered modality-specific momentum."""

    def __init__(
        self,
        base_beta=0.9,
        statistics_ema=0.95,
        adaptation_strength=1.0,
        drift_floor=1e-4,
        gain_min=0.01,
        gain_max=0.30,
        epsilon=1e-20,
    ):
        self.base_beta = float(base_beta)
        self.base_gain = 1.0 - self.base_beta
        self.statistics_ema = float(statistics_ema)
        self.adaptation_strength = float(adaptation_strength)
        self.drift_floor = float(drift_floor)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.epsilon = float(epsilon)
        self.state = {}

    @staticmethod
    @torch.no_grad()
    def _probe_gradient(logits, features, labels, indices):
        logits = logits.detach().float()[indices]
        features = features.detach().float()[indices]
        labels = labels.detach()[indices]
        targets = torch.nn.functional.one_hot(
            labels, num_classes=logits.shape[1]
        ).float()
        residual = torch.softmax(logits, dim=1) - targets
        count = residual.shape[0]
        grad_weight = residual.transpose(0, 1).matmul(features) / count
        grad_bias = residual.mean(dim=0)
        return torch.cat([grad_weight.flatten(), grad_bias.flatten()])

    @torch.no_grad()
    def _measure(self, logits, features, labels):
        batch_size = logits.shape[0]
        if batch_size < 2:
            raise ValueError("VCMM requires at least two samples per minibatch")
        device = logits.device
        first = torch.arange(0, batch_size, 2, device=device)
        second = torch.arange(1, batch_size, 2, device=device)
        full = torch.arange(batch_size, device=device)
        grad_first = self._probe_gradient(logits, features, labels, first)
        grad_second = self._probe_gradient(logits, features, labels, second)
        grad_full = self._probe_gradient(logits, features, labels, full)
        r_instant = 0.25 * torch.mean((grad_first - grad_second).square()).item()
        signal_energy = torch.mean(grad_full.square()).item()
        return grad_full, max(r_instant, self.epsilon), signal_energy

    def _update_statistics(self, modality, probe, r_instant, signal_energy):
        previous = self.state.get(modality)
        if previous is None:
            initial_ratio = self.base_gain**2 / max(self.base_beta, self.epsilon)
            self.state[modality] = {
                "probe": probe.clone(),
                "r_instant": r_instant,
                "r": r_instant,
                "q": initial_ratio * r_instant,
            }
            return False

        temporal_difference = torch.mean((probe - previous["probe"]).square()).item()
        q_instant = max(
            temporal_difference - r_instant - previous["r_instant"],
            self.drift_floor * max(signal_energy, self.epsilon),
        )
        a = self.statistics_ema
        previous["r"] = a * previous["r"] + (1.0 - a) * r_instant
        previous["q"] = a * previous["q"] + (1.0 - a) * q_instant
        previous["probe"] = probe.clone()
        previous["r_instant"] = r_instant
        return True

    @staticmethod
    def _kalman_gain(ratio):
        ratio = max(float(ratio), 0.0)
        if ratio == 0.0:
            return 0.0
        root = math.sqrt(ratio * ratio + 4.0 * ratio)
        return 2.0 * ratio / (root + ratio)

    @torch.no_grad()
    def update(self, image_logits, image_features, text_logits, text_features, labels):
        initialized = True
        for modality, logits, features in (
            ("image", image_logits, image_features),
            ("text", text_logits, text_features),
        ):
            probe, r_instant, signal = self._measure(logits, features, labels)
            initialized &= self._update_statistics(
                modality, probe, r_instant, signal
            )

        # A temporal difference does not exist at t=1, so the base momentum is
        # used for that initialization step. No additional warm-up is applied.
        if not initialized:
            return {"image": self.base_beta, "text": self.base_beta}

        raw_logits = {}
        for modality in ("image", "text"):
            state = self.state[modality]
            ratio = state["q"] / max(state["r"], self.epsilon)
            raw_logits[modality] = _logit(self._kalman_gain(ratio))

        center = 0.5 * (raw_logits["image"] + raw_logits["text"])
        base_logit = _logit(self.base_gain)
        betas = {}
        for modality in ("image", "text"):
            centered = base_logit + self.adaptation_strength * (
                raw_logits[modality] - center
            )
            gain = min(max(_sigmoid(centered), self.gain_min), self.gain_max)
            betas[modality] = 1.0 - gain
        return betas


class VCMMAdam(Optimizer):
    """Adam with modal beta1 and exact time-varying beta-product correction."""

    def __init__(
        self,
        params,
        lr=2e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=2e-4,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_beta=betas[0],
            modality="shared",
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def set_modal_betas(self, modal_betas):
        for group in self.param_groups:
            modality = group["modality"]
            group["momentum_beta"] = modal_betas.get(
                modality, group["betas"][0]
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1 = group["momentum_beta"]
            beta2 = group["betas"][1]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("VCMMAdam does not support sparse gradients")
                if group["weight_decay"]:
                    gradient = gradient.add(
                        parameter, alpha=group["weight_decay"]
                    )
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["beta1_product"] = 1.0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                state["beta1_product"] *= beta1
                state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2
                )
                correction1 = max(1.0 - state["beta1_product"], 1e-16)
                correction2 = 1.0 - beta2 ** state["step"]
                numerator = state["exp_avg"] / correction1
                denominator = (
                    state["exp_avg_sq"].sqrt() / math.sqrt(correction2)
                ).add_(group["eps"])
                parameter.addcdiv_(
                    numerator, denominator, value=-group["lr"]
                )
        return loss
