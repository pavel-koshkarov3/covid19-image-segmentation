import tensorflow as tf
from dataset import get_data, data_generator
from model import unet_model
from configs import BATCH_SIZE, EPOCHS, MODEL_PATH

# Загружаем данные
image_paths, mask_paths = get_data()

# Создаём модель
model = unet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Обучение
train_generator = data_generator(image_paths[:80], mask_paths[:80], BATCH_SIZE)
val_generator = data_generator(image_paths[80:], mask_paths[80:], BATCH_SIZE)

# Явно указываем steps_per_epoch и validation_steps
steps_per_epoch = len(image_paths[:80]) // BATCH_SIZE
validation_steps = len(image_paths[80:]) // BATCH_SIZE

model.fit(train_generator, 
          validation_data=val_generator, 
          epochs=EPOCHS,
          steps_per_epoch=steps_per_epoch,  # Указываем steps_per_epoch
          validation_steps=validation_steps)  # Указываем validation_steps

# Сохранение модели
model.save(MODEL_PATH)

