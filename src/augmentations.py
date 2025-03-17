import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

# Определяем аугментации
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    ToTensorV2()
])

def apply_augmentations(image, mask):
    """Применяет аугментации к изображению и маске."""
    augmented = augmentation_pipeline(image=image, mask=mask)
    return augmented["image"], augmented["mask"]
