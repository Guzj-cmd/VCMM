"""VCMM dynamics estimation and Adam with time-varying modal momentum."""

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
    """Online R/Q estimation and centered momentum for any number of modalities.

    Probes are gradients of each modal classifier's own cross-entropy, used as
    surrogates for modal dynamics. Parameter updates still use the gradient of
    the fused training loss. Probes do not add an auxiliary training objective.

    The first observation initializes R directly and Q/R to (1-beta0)^2/beta0.
    During the configurable warm-up (100 steps by default), statistics evolve
    while all modalities retain beta0. Adaptive control starts after warm-up
    and after two observations are available.
    """

    def __init__(
        self,
        base_beta=0.9,
        statistics_ema=0.95,
        adaptation_strength=1.0,
        warmup_steps=100,
        drift_floor=1e-4,
        gain_min=0.01,
        gain_max=0.30,
        epsilon=1e-20,
    ):
        self.base_beta = float(base_beta)
        self.base_gain = 1.0 - self.base_beta
        self.statistics_ema = float(statistics_ema)
        self.adaptation_strength = float(adaptation_strength)
        self.warmup_steps = int(warmup_steps)
        self.drift_floor = float(drift_floor)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.epsilon = float(epsilon)
        self.steps = 0
        self.state = {}

    @staticmethod
    @torch.no_grad()
    def _probe_gradient(logits, features, labels, indices):
        """Analytic gradient of modal CE; detached from the fused-loss graph."""
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
        # For equal halves this is exactly the manuscript's factor 1/4.
        # n_A*n_B/n^2 also gives the full-batch noise scale for odd batches.
        noise_scale = first.numel() * second.numel() / float(batch_size**2)
        r_instant = noise_scale * torch.mean((grad_first - grad_second).square()).item()
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
        """Backward-compatible wrapper for the original image/text interface."""
        return self.update_modalities(
            {
                "image": (image_logits, image_features),
                "text": (text_logits, text_features),
            },
            labels,
        )

    @torch.no_grad()
    def update_modalities(self, modalities, labels):
        """Return beta by name from {name: (logits [N,C], features [N,D])}.

        Names must remain fixed across steps and match optimizer group labels.
        For example, a trimodal caller can provide rgb, flow, and depth.
        """
        if len(modalities) < 2:
            raise ValueError("VCMM requires at least two modalities")
        if self.state and set(modalities) != set(self.state):
            raise ValueError("Modality names must remain fixed across steps")
        if labels.ndim != 1 or labels.shape[0] < 2:
            raise ValueError("VCMM requires one label per sample and at least two samples")
        for name, (logits, features) in modalities.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Modality names must be nonempty strings")
            if (
                logits.ndim != 2 or features.ndim != 2
                or logits.shape[0] != labels.shape[0]
                or features.shape[0] != labels.shape[0]
            ):
                raise ValueError("Each modality must provide [N,C] logits and [N,D] features")

        initialized = True
        for modality, (logits, features) in modalities.items():
            probe, r_instant, signal = self._measure(logits, features, labels)
            initialized &= self._update_statistics(
                modality, probe, r_instant, signal
            )

        self.steps += 1

        # Statistics are collected during warm-up, but parameter updates retain
        # the base momentum until the controller has a stable history.
        if not initialized or self.steps <= self.warmup_steps:
            return {modality: self.base_beta for modality in modalities}

        raw_logits = {}
        for modality in modalities:
            state = self.state[modality]
            ratio = state["q"] / max(state["r"], self.epsilon)
            raw_logits[modality] = _logit(self._kalman_gain(ratio))

        center = sum(raw_logits.values()) / len(raw_logits)
        base_logit = _logit(self.base_gain)
        betas = {}
        for modality in modalities:
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
