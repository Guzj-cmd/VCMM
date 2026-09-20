#!/usr/bin/env python3
"""Train and evaluate the released VCMM example."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.evaluation import classification_metrics, fuse_logits, summarize_metrics
from utils.vcmm import VCMMAdam, VCMMController


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs/vcmm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--resnet-checkpoint")
    parser.add_argument("--config", default="data/config.json")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(dataset, batch_size, workers, training, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=workers,
        pin_memory=True,
        generator=generator,
    )


def tokenize(tokenizer, texts, max_tokens, device):
    return tokenizer(
        list(texts),
        padding="longest",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    ).to(device)


def parameter_groups(model):
    return [
        {
            "params": list(model.image_encoder.parameters())
            + list(model.image_classifier.parameters()),
            "modality": "image",
        },
        {
            "params": list(model.text_encoder.parameters())
            + list(model.text_classifier.parameters()),
            "modality": "text",
        },
    ]


def train_epoch(model, loader, tokenizer, optimizer, controller, max_tokens, device):
    model.train()
    correct = 0
    sample_count = 0
    for images, texts, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        text_inputs = tokenize(tokenizer, texts, max_tokens, device)
        optimizer.zero_grad(set_to_none=True)
        image_logits, text_logits, image_features, text_features = model(
            images, text_inputs
        )
        modal_betas = controller.update_modalities(
            {
                "image": (image_logits, image_features),
                "text": (text_logits, text_features),
            },
            labels,
        )
        optimizer.set_modal_betas(modal_betas)
        fused_logits = fuse_logits(image_logits, text_logits)
        loss = F.cross_entropy(fused_logits, labels)
        loss.backward()
        optimizer.step()
        correct += (fused_logits.argmax(dim=1) == labels).sum().item()
        sample_count += labels.shape[0]
    return correct / sample_count


@torch.no_grad()
def collect_logits(model, loader, tokenizer, max_tokens, device):
    model.eval()
    image_predictions = []
    text_predictions = []
    targets = []
    for images, texts, labels in loader:
        images = images.to(device, non_blocking=True)
        text_inputs = tokenize(tokenizer, texts, max_tokens, device)
        image_logits, text_logits, _, _ = model(images, text_inputs)
        image_predictions.append(image_logits.cpu())
        text_predictions.append(text_logits.cpu())
        targets.append(labels)
    return (
        torch.cat(image_predictions),
        torch.cat(text_predictions),
        torch.cat(targets),
    )


def fused_metrics(image_logits, text_logits, targets):
    return classification_metrics(fuse_logits(image_logits, text_logits), targets)


def run_seed(seed, config, args, datasets, tokenizer, output_dir):
    from model.multimodal_model import MultimodalModel

    set_seed(seed)
    device = torch.device(args.device)
    loaders = {
        split: make_loader(
            dataset,
            config["batch_size"],
            config["num_workers"],
            split == "train",
            seed,
        )
        for split, dataset in datasets.items()
    }
    model = MultimodalModel(args.bert_model, args.resnet_checkpoint).to(device)
    optimizer = VCMMAdam(
        parameter_groups(model),
        lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_epsilon"],
        weight_decay=config["weight_decay"],
    )
    controller = VCMMController(
        base_beta=config["adam_beta1"],
        statistics_ema=config["statistics_ema"],
        adaptation_strength=config["adaptation_strength"],
        warmup_steps=config["warmup_steps"],
        drift_floor=config["drift_floor"],
        gain_min=config["gain_min"],
        gain_max=config["gain_max"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr_scheduler_step_size"],
        gamma=config["lr_scheduler_gamma"],
    )

    checkpoint = output_dir / f"seed_{seed}_best.pt"
    best_dev_key = None
    epoch_train_acc = []
    for epoch in range(config["epochs"]):
        train_acc = train_epoch(
            model,
            loaders["train"],
            tokenizer,
            optimizer,
            controller,
            config["max_tokens"],
            device,
        )
        dev_logits = collect_logits(
            model, loaders["dev"], tokenizer, config["max_tokens"], device
        )
        dev_metrics = fused_metrics(*dev_logits)
        # Select by validation accuracy only; break ties with the earliest epoch.
        dev_key = (dev_metrics["acc"], -(epoch + 1))
        if best_dev_key is None or dev_key > best_dev_key:
            best_dev_key = dev_key
            torch.save(
                {
                    "model": model.state_dict(),
                    "fusion": "equal_logits",
                    "epoch": epoch + 1,
                    "dev_metrics": dev_metrics,
                    "seed": seed,
                    "config": config,
                },
                checkpoint,
            )

        epoch_result = {
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "dev_acc": dev_metrics["acc"],
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        epoch_train_acc.append(epoch_result)
        print(json.dumps({"seed": seed, **epoch_result}, sort_keys=True))
        scheduler.step()

    selected = torch.load(checkpoint, map_location=device)
    model.load_state_dict(selected["model"])
    test_logits = collect_logits(
        model, loaders["test"], tokenizer, config["max_tokens"], device
    )
    test_metrics = fused_metrics(*test_logits)
    result = {
        "seed": seed,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "best_epoch": selected["epoch"],
        "best_dev_metrics": selected["dev_metrics"],
        "fusion": "equal_logits",
        "epoch_train_acc": epoch_train_acc,
    }
    (output_dir / f"seed_{seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {"seed": seed, "test_acc": result["test_acc"],
         "test_macro_f1": result["test_macro_f1"]},
        sort_keys=True,
    ))
    return result


def main():
    from transformers import AutoTokenizer
    from dataset.image_text_dataset import build_splits

    args = arguments()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = build_splits(
        args.data_root, config["image_size"], config["max_tokens"]
    )
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    results = [
        run_seed(seed, config, args, datasets, tokenizer, output_dir)
        for seed in config["seeds"]
    ]
    summary = {
        "seeds": config["seeds"],
        "config": config,
        "evaluation_protocol": {
            "fusion": "fixed equal logit fusion on train, dev, and test",
            "checkpoint_selection": "highest dev accuracy; earliest epoch on ties",
            "macro_f1": "unweighted mean over all output classes; zero for undefined F1",
        },
        **summarize_metrics(results),
        "per_seed": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
