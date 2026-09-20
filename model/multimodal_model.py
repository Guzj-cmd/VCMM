"""BERT and ResNet-50 branches for the released multimodal example."""

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from transformers import BertModel

from utils.evaluation import fuse_logits


class MultimodalModel(nn.Module):
    """The two modal branches and their linear classifiers.

    This preserves the reference pipeline: the pretrained ResNet-50
    ImageNet output is the 1,000-dimensional image representation, while the
    BERT [CLS] state is the 768-dimensional text representation.
    """

    def __init__(self, bert_model: str, resnet_checkpoint: Optional[str] = None):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(
            bert_model, add_pooling_layer=False
        )
        if resnet_checkpoint is None:
            self.image_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.image_encoder = resnet50(weights=None)
            state = torch.load(Path(resnet_checkpoint), map_location="cpu")
            self.image_encoder.load_state_dict(state)

        self.image_classifier = nn.Linear(1000, 3)
        self.text_classifier = nn.Linear(self.text_encoder.config.hidden_size, 3)

    def forward(self, images, text_inputs):
        image_features = self.image_encoder(images)
        image_logits = self.image_classifier(image_features)
        text_features = self.text_encoder(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            return_dict=True,
        ).last_hidden_state[:, 0]
        text_logits = self.text_classifier(text_features)
        return image_logits, text_logits, image_features, text_features
