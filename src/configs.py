import os


# Путь к корневой папке проекта
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_PATH, "dataset")

IMAGES_PATH = os.path.join(DATASET_PATH, "Normal", "images")
MASKS_PATH = os.path.join(DATASET_PATH, "Normal", "masks")


# Гиперпараметры
IMG_SIZE = (128, 128)  # Размер изображений
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# Пути для модели
MODEL_PATH = "./models/covid_segmentation_model.h5"
