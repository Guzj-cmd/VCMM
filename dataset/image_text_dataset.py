"""Dataset loader and image preprocessing for the released example."""

import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from utils.randaugment import ImageRandomAugment


ImageFile.LOAD_TRUNCATED_IMAGES = True

_NORMALIZE = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)


def build_transform(image_size: int, training: bool):
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.5, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                ImageRandomAugment(num_ops=2, magnitude=7),
                transforms.ToTensor(),
                _NORMALIZE,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            _NORMALIZE,
        ]
    )


def clean_text(text: str, max_words: int) -> str:
    text = " ".join(re.findall(r"\b[A-Za-z]+\b", str(text)))
    text = re.sub(r"([,.'!?\"()*#:;~])", "", text.lower())
    text = text.replace("-", " ").replace("/", " ").replace("<person>", "person")
    text = re.sub(r"\s{2,}", " ", text).strip()
    return " ".join(text.split()[:max_words])


class ImageTextDataset(Dataset):
    def __init__(self, annotation_file, image_root, transform, max_words=30):
        frame = pd.read_csv(annotation_file, sep="\t")
        if not {"Label", "ImageID", "String"}.issubset(frame.columns):
            columns = list(frame.columns)
            if len(columns) < 4:
                raise ValueError(
                    f"{annotation_file} must contain index, label, image and text columns"
                )
            frame = frame.rename(
                columns={columns[1]: "Label", columns[2]: "ImageID", columns[3]: "String"}
            )
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.max_words = max_words

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(self.image_root / str(row["ImageID"])).convert("RGB")
        return (
            self.transform(image),
            clean_text(row["String"], self.max_words),
            int(row["Label"]),
        )


def build_splits(data_root, image_size, max_words):
    root = Path(data_root)
    annotations = root / "annotations"
    images = root / "twitter2015_images"
    expected = [
        annotations / "train.tsv",
        annotations / "dev.tsv",
        annotations / "test.tsv",
        images,
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing example dataset paths: " + ", ".join(missing))
    train_transform = build_transform(image_size, training=True)
    eval_transform = build_transform(image_size, training=False)
    return {
        "train": ImageTextDataset(
            annotations / "train.tsv", images, train_transform, max_words
        ),
        "dev": ImageTextDataset(
            annotations / "dev.tsv", images, eval_transform, max_words
        ),
        "test": ImageTextDataset(
            annotations / "test.tsv", images, eval_transform, max_words
        ),
    }
