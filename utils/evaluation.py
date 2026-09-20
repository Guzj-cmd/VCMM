"""Shared fusion and classification metrics for training and evaluation."""

import math

import torch


def fuse_logits(*modal_logits):
    """Use the same fixed, equal logit fusion for every split and modality."""
    if len(modal_logits) < 2:
        raise ValueError("Fusion requires at least two modalities")
    if any(logits.shape != modal_logits[0].shape for logits in modal_logits):
        raise ValueError("All modal logits must have the same shape")
    return torch.stack(modal_logits, dim=0).mean(dim=0)


@torch.no_grad()
def classification_metrics(logits, targets):
    """Accuracy and Macro-F1 over all C classes; undefined class F1 is zero."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.numel():
        raise ValueError("Expected [N,C] logits and [N] targets")
    if targets.numel() == 0 or logits.shape[1] == 0:
        raise ValueError("Cannot evaluate an empty sample or class dimension")
    num_classes = logits.shape[1]
    predictions = logits.argmax(dim=1)
    confusion = torch.bincount(
        targets * num_classes + predictions, minlength=num_classes**2
    ).reshape(num_classes, num_classes).double()
    true_positives = confusion.diag()
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    f1 = 2.0 * true_positives / denominator.clamp_min(1.0)
    return {
        "acc": true_positives.sum().item() / targets.numel(),
        "macro_f1": f1.mean().item(),
    }


def summarize_metrics(results):
    """Across-seed mean, sample SD, and approximate normal CI half-width.

    SD uses ddof=1. With one seed uncertainty is undefined and reported as null.
    These fields do not reinterpret uncertainty in previously reported results.
    """
    if not results:
        raise ValueError("At least one seed result is required")
    summary = {
        "uncertainty": {
            "std": "sample standard deviation across seeds (ddof=1)",
            "ci95": "approximate normal 95% CI half-width: 1.96 * std / sqrt(n)",
        }
    }
    for metric in ("test_acc", "test_macro_f1"):
        values = [float(result[metric]) for result in results]
        mean = sum(values) / len(values)
        std = (
            math.sqrt(sum((value - mean)**2 for value in values) / (len(values) - 1))
            if len(values) > 1 else None
        )
        summary[metric + "_mean"] = mean
        summary[metric + "_std"] = std
        summary[metric + "_ci95"] = (
            1.96 * std / math.sqrt(len(values)) if std is not None else None
        )
    return summary
