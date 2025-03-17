from dotenv import load_dotenv
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем пути из переменных окружения
image_path = os.getenv('IMAGE_PATH')
mask_path = os.getenv('MASK_PATH')
model_path = os.getenv('MODEL_PATH')

# Проверим, были ли правильно загружены все переменные окружения
print(f"IMAGE_PATH: {image_path}")
print(f"MASK_PATH: {mask_path}")
print(f"MODEL_PATH: {model_path}")

if not image_path or not mask_path or not model_path:
    raise ValueError("Одно из значений в файле .env отсутствует")

# Загрузка модели
model = tf.keras.models.load_model(model_path)

# Функция загрузки изображения
def load_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path).convert('RGB')  # Открытие изображения и преобразование в RGB
    image = image.resize(target_size)  # Изменение размера изображения
    image = np.array(image) / 255.0  # Нормализация изображения
    image = np.expand_dims(image, axis=0)  # Добавление размерности для батча
    return image

def postprocess_mask(mask):
    """Применяет морфологические операции и сглаживание к бинарной маске."""
    # Дилатация (расширение) для улучшения маски
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Эрозия (уменьшение) для удаления маленьких шумовых участков
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # Возвращаем результат
    return mask

def predict_mask(image_path, threshold=0.5):
    """Предсказание маски с возможностью настройки порога и постобработки."""
    image = load_image(image_path)
    pred_mask = model.predict(image)[0]

    # Бинаризация с заданным порогом
    pred_mask = (pred_mask > threshold).astype(np.uint8) * 255

    # Применяем постобработку (морфологические операции)
    pred_mask = postprocess_mask(pred_mask)
    
    return pred_mask

def save_images(original_image_path, mask_image_path, pred_mask):
    """Сохраняет изображения после предсказания и постобработки."""
    
    # Загружаем оригинальное изображение с помощью PIL
    original_image = Image.open(original_image_path)

    # Загружаем маску с помощью PIL
    original_mask = Image.open(mask_image_path).convert('L')  # Преобразуем в оттенки серого

    # Преобразуем предсказанную маску в изображение
    pred_mask_image = Image.fromarray(pred_mask.astype(np.uint8))
    
    # Сохраняем изображения
    original_image.save("original_image.png")
    original_mask.save("original_mask.png")
    pred_mask_image.save("predicted_mask.png")

    print("Изображения сохранены: original_image.png, original_mask.png, predicted_mask.png")

# Пример вызова
pred_mask = predict_mask(image_path, threshold=0.4)  # Попробуйте разные пороги

save_images(image_path, mask_path, pred_mask)















