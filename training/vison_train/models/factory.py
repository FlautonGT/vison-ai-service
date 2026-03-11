"""Torchvision model factories for the training pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision import models


def build_feature_extractor(backbone: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    if backbone == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()), int(base.classifier[1].in_features)
    if backbone == "efficientnet_b4":
        base = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()), int(base.classifier[1].in_features)
    if backbone == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(*list(base.children())[:-1], nn.Flatten()), int(base.fc.in_features)
    if backbone == "mobilenet_v3_large":
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()), 960
    if backbone == "vit_b_16":
        base = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        hidden_dim = int(base.heads.head.in_features)
        base.heads = nn.Identity()
        return base, hidden_dim
    raise ValueError(f"Unsupported backbone: {backbone}")


class ClassificationModel(nn.Module):
    def __init__(self, backbone: str, num_outputs: int, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_feature_extractor(backbone, pretrained=pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class AgeGenderModel(nn.Module):
    def __init__(self, backbone: str, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_feature_extractor(backbone, pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.age_head = nn.Linear(feat_dim, 1)
        self.gender_head = nn.Linear(feat_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.dropout(self.backbone(x))
        return self.age_head(features).squeeze(1), self.gender_head(features)


class EmbeddingModel(nn.Module):
    def __init__(self, backbone: str, embedding_dim: int = 512, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_feature_extractor(backbone, pretrained=pretrained)
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.projection(self.backbone(x))
        return nn.functional.normalize(embeddings, dim=1)


class SegmentationModel(nn.Module):
    def __init__(self, backbone: str = "deeplabv3_mobilenet_v3_large", num_classes: int = 19, pretrained: bool = True):
        super().__init__()
        if backbone == "deeplabv3_mobilenet_v3_large":
            weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights, num_classes=num_classes)
        elif backbone == "lraspp_mobilenet_v3_large":
            weights = models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.model = models.segmentation.lraspp_mobilenet_v3_large(weights=weights, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported segmentation backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        if isinstance(outputs, dict):
            return outputs["out"]
        return outputs


class ArcMarginProduct(nn.Module):
    """Angular-margin head for face verification training."""

    def __init__(self, in_features: int, out_features: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = nn.functional.linear(
            nn.functional.normalize(embeddings),
            nn.functional.normalize(self.weight),
        )
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale
