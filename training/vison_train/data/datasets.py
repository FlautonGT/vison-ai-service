"""PyTorch datasets backed by CSV manifests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(input_size: int, augment: bool = False) -> transforms.Compose:
    common = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if not augment:
        return transforms.Compose(common)
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_segmentation_transforms(input_size: int, augment: bool = False):
    del augment
    image_ops = [transforms.Resize((input_size, input_size))]
    mask_ops = [transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST)]
    image_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    mask_ops.append(transforms.PILToTensor())
    return transforms.Compose(image_ops), transforms.Compose(mask_ops)


def _load_rgb(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (128, 128, 128))


class ManifestImageDataset(Dataset):
    """Generic single-image dataset for classification or regression."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_col: str,
        target_cols: list[str],
        transform,
        target_mode: str,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_col = image_col
        self.target_cols = target_cols
        self.transform = transform
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self.transform(_load_rgb(str(row[self.image_col])))
        if self.target_mode in {"binary", "regression"}:
            target = torch.tensor(float(row[self.target_cols[0]]), dtype=torch.float32)
        elif self.target_mode == "multilabel":
            target = torch.tensor(row[self.target_cols].astype(np.float32).values, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported target mode: {self.target_mode}")
        return image, target


class MultiLabelDataset(Dataset):
    """Multi-label dataset with optional missing-label masking."""

    def __init__(self, frame: pd.DataFrame, image_col: str, label_cols: list[str], transform):
        self.frame = frame.reset_index(drop=True)
        self.image_col = image_col
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self.transform(_load_rgb(str(row[self.image_col])))
        values = row[self.label_cols].astype(np.float32).values
        mask = np.isfinite(values) & (values >= 0.0)
        clean_values = np.where(mask, values, 0.0).astype(np.float32)
        return (
            image,
            torch.tensor(clean_values, dtype=torch.float32),
            torch.tensor(mask.astype(np.float32), dtype=torch.float32),
        )


class AgeGenderDataset(Dataset):
    """Multi-task age/gender dataset."""

    def __init__(self, frame: pd.DataFrame, image_col: str, age_col: str, gender_col: str, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_col = image_col
        self.age_col = age_col
        self.gender_col = gender_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self.transform(_load_rgb(str(row[self.image_col])))
        age = torch.tensor(float(row[self.age_col]), dtype=torch.float32)
        gender = torch.tensor(int(row[self.gender_col]), dtype=torch.long)
        return image, age, gender


class IdentityImageDataset(Dataset):
    """Identity-labeled images for metric learning."""

    def __init__(self, frame: pd.DataFrame, image_col: str, identity_col: str, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_col = image_col
        self.identity_col = identity_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self.transform(_load_rgb(str(row[self.image_col])))
        identity = torch.tensor(int(row[self.identity_col]), dtype=torch.long)
        return image, identity


class SegmentationDataset(Dataset):
    """Image and mask dataset for parser/segmentation training."""

    def __init__(self, frame: pd.DataFrame, image_col: str, mask_col: str, image_transform, mask_transform):
        self.frame = frame.reset_index(drop=True)
        self.image_col = image_col
        self.mask_col = mask_col
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self.image_transform(_load_rgb(str(row[self.image_col])))
        try:
            mask_image = Image.open(str(row[self.mask_col])).convert("L")
        except Exception:
            mask_image = Image.new("L", (224, 224), 0)
        mask = self.mask_transform(mask_image).squeeze(0).long()
        return image, mask
