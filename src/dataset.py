import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from configs import IMAGES_PATH, MASKS_PATH, IMG_SIZE

def load_image(image_path):
    """Загружает изображение и масштабирует его."""
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0  # Нормализация
    return img

def load_mask(mask_path):
    """Загружает маску и преобразует её в бинарный формат."""
    mask = load_img(mask_path, target_size=IMG_SIZE, color_mode="grayscale")
    mask = img_to_array(mask) / 255.0  # Приведение значений к [0,1]
    mask = (mask > 0.5).astype(np.float32)  # Бинаризация
    return mask

def data_generator(image_paths, mask_paths, batch_size):
    """Генератор данных для обучения."""
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_images = [load_image(p) for p in image_paths[i:i+batch_size]]
            batch_masks = [load_mask(p) for p in mask_paths[i:i+batch_size]]

            yield np.array(batch_images), np.array(batch_masks)

def get_data():
    """Загружает пути к изображениям и маскам."""
    image_paths = sorted([os.path.join(IMAGES_PATH, f) for f in os.listdir(IMAGES_PATH)])
    mask_paths = sorted([os.path.join(MASKS_PATH, f) for f in os.listdir(MASKS_PATH)])
    return image_paths, mask_paths
