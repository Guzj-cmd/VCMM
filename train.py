#!/usr/bin/env python3
"""Train and evaluate the released VCMM example."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.image_text_dataset import build_splits
from model.multimodal_model import MultimodalModel, fuse_logits
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
    loss_sum = 0.0
    sample_count = 0
    for images, texts, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        text_inputs = tokenize(tokenizer, texts, max_tokens, device)
        optimizer.zero_grad(set_to_none=True)
        image_logits, text_logits, image_features, text_features = model(
            images, text_inputs
        )
        modal_betas = controller.update(
            image_logits, image_features, text_logits, text_features, labels
        )
        optimizer.set_modal_betas(modal_betas)
        loss = F.cross_entropy(fuse_logits(image_logits, text_logits), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.shape[0]
        sample_count += labels.shape[0]
    return loss_sum / sample_count


@torch.no_grad()
def evaluate(model, loader, tokenizer, max_tokens, device):
    model.eval()
    predictions = []
    targets = []
    for images, texts, labels in loader:
        images = images.to(device, non_blocking=True)
        text_inputs = tokenize(tokenizer, texts, max_tokens, device)
        image_logits, text_logits, _, _ = model(images, text_inputs)
        predictions.append(fuse_logits(image_logits, text_logits).argmax(dim=1).cpu())
        targets.append(labels)
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def run_seed(seed, config, args, datasets, tokenizer, output_dir):
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
        drift_floor=config["drift_floor"],
        gain_min=config["gain_min"],
        gain_max=config["gain_max"],
    )

    checkpoint = output_dir / f"seed_{seed}_best_dev.pt"
    best_dev = None
    history = []
    for epoch in range(config["epochs"]):
        train_loss = train_epoch(
            model,
            loaders["train"],
            tokenizer,
            optimizer,
            controller,
            config["max_tokens"],
            device,
        )
        dev = evaluate(
            model, loaders["dev"], tokenizer, config["max_tokens"], device
        )
        record = {"epoch": epoch + 1, "train_loss": train_loss, "dev": dev}
        history.append(record)
        print(json.dumps({"seed": seed, **record}, sort_keys=True))
        key = (dev["accuracy"], dev["macro_f1"])
        if best_dev is None or key > best_dev[0]:
            best_dev = (key, epoch + 1)
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test = evaluate(
        model, loaders["test"], tokenizer, config["max_tokens"], device
    )
    result = {
        "seed": seed,
        "selected_epoch": best_dev[1],
        "dev": {"accuracy": best_dev[0][0], "macro_f1": best_dev[0][1]},
        "test": test,
        "history": history,
    }
    (output_dir / f"seed_{seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main():
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
    accuracy = np.array([result["test"]["accuracy"] for result in results])
    macro_f1 = np.array([result["test"]["macro_f1"] for result in results])
    summary = {
        "seeds": config["seeds"],
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std(ddof=1)),
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std": float(macro_f1.std(ddof=1)),
        "per_seed": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "per_seed"}))


if __name__ == "__main__":
    main()
