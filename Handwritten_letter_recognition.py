import numpy as np
dataset = np.loadtxt('https://storage.yandexcloud.net/academy.ai/A_Z_Handwritten_Data.csv', delimiter=',')

X = dataset[:,1:785]
Y = dataset[:,0]

from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, shuffle=True)

# Commented out IPython magic to ensure Python compatibility.

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

for i in range(40):
    x = x_train[i]
    x = x.reshape((28, 28))
    plt.axis('off')
    im = plt.subplot(5, 8, i+1)
    plt.title(word_dict.get(y_train[i]))
    im.imshow(x, cmap='gray')

"""# Распознавание рукописных букв



---

Импорты и настройка.

Пояснение:
* Импорты: Загрузка всех необходимых библиотек для работы с данными, построения модели, обучения и визуализации.
* Настройка GPU: Ограничение роста памяти GPU предотвращает исчерпание ОЗУ.
* Смешанная точность: Включение для ускорения обучения на поддерживаемых GPU.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.keras import mixed_precision
import datetime
import gc

print("Версия TensorFlow:", tf.__version__)
print("Доступные устройства:", tf.config.list_physical_devices())

# Ограничение роста памяти GPU для предотвращения исчерпания ОЗУ
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Рост памяти GPU ограничен.")
    except RuntimeError as e:
        print(e)

# Настройка смешанной точности
mixed_precision.set_global_policy('float32')

# Очистка неиспользуемой памяти
gc.collect()

"""

---

Загрузка и предобработка данных.

Пояснение:
* Загрузка данных: Загрузка CSV-файла с метками и пиксельными значениями.
* Нормализация: Преобразование значений пикселей в диапазон [0, 1].
* One-Hot Encoding: Преобразование меток в формат one-hot для многоклассовой классификации.
* Разделение: Разделение данных на обучающую и тестовую выборки.
* Форматирование: Преобразование данных в формат изображений (28x28x1).
* Сохранение Y_test: Копирование Y_test перед очисткой памяти для последующего анализа."""

dataset = np.loadtxt('https://storage.yandexcloud.net/academy.ai/A_Z_Handwritten_Data.csv', delimiter=',')

# Разделение на признаки и метки
X = dataset[:, 1:]
Y = dataset[:, 0]

# Нормализация значений пикселей
X = X / 255.0

# Преобразование меток в формат one-hot encoding и преобразование типа данных в float32
Y = to_categorical(Y, num_classes=26).astype('float32')

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Преобразование данных в формат изображений (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

# Сохранение Y_test перед очисткой
Y_test_original = Y_test.copy()

# Очистка памяти
del dataset, Y_test
gc.collect()

# Вывод формы данных
print(f'Форма X_train: {X_train.shape}')
print(f'Форма Y_train: {Y_train.shape}')
print(f'Форма X_test: {X_test.shape}')
print(f'Форма Y_test_original: {Y_test_original.shape}')

"""

---

Аугментация данных.

Пояснение:
* Функция аугментации: Применение случайных трансформаций к изображениям для увеличения разнообразия обучающих данных.
* Создание tf.data.Dataset: Эффективное создание батчей и предзагрузка данных для ускорения обучения.
* Визуализация: Проверка нескольких примеров аугментированных данных для уверенности в корректности аугментации."""

def augment_image_full(image, label):
    # Случайное горизонтальное отражение
    image = tf.image.random_flip_left_right(image)

    # Случайная яркость
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Случайная контрастность
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Случайный зум
    zoom = tf.random.uniform([], 0.9, 1.1)
    new_size = tf.cast(tf.cast(tf.shape(image)[:2], tf.float32) * zoom, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, 28, 28)

    return image, label

# Создание tf.data.Dataset с аугментацией
batch_size = 128
AUTOTUNE = tf.data.AUTOTUNE

# Обучающая выборка
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.map(augment_image_full, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(AUTOTUNE)

# Тестовая выборка (без аугментации)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test_original))
test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)

# Очистка неиспользуемых переменных
del X_train, Y_train, X_test
gc.collect()

# Проверка нескольких примеров аугментированных данных
import random

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().reshape(28, 28), cmap='gray')
        plt.title(chr(np.argmax(labels[i].numpy()) + 65))
        plt.axis("off")
plt.show()

"""

---

Построение архитектуры модели.

Пояснение:
* Слои CNN: Извлечение признаков с помощью сверточных и пуллинговых слоёв.
* Регуляризация: Использование BatchNormalization и Dropout для стабилизации и предотвращения переобучения.
* Полносвязные слои: Обработка извлечённых признаков и финальная классификация."""

model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(26, activation='softmax')
])

# Краткий обзор модели
model.summary()

"""

---

Компиляция модели.

Пояснение:
* Оптимизатор: Использование Adam с низкой скоростью обучения для стабильного обучения.
* Функция потерь: categorical_crossentropy подходит для многоклассовой классификации.
* Метрики: Отслеживание точности во время обучения."""

optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

"""

---

Настройка колбэков.

Пояснение:
* ReduceLROnPlateau: Автоматическое снижение скорости обучения при отсутствии улучшений.
* TensorBoard: Визуализация процесса обучения, метрик и распределения весов"""

# Настройка ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Настройка TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

"""

---

Обучение модели.

Пояснение:
* Обучение: Запуск процесса обучения модели с использованием подготовленных датасетов и колбэков.
* Epochs: Установлено на 20."""

history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=test_dataset,
    callbacks=[reduce_lr, tensorboard_callback]
)

"""

---

Сохранение модели.

Пояснение:
* Сохранение: Сохранение обученной модели в формате .keras для последующего использования.
* Проверка: Сообщение о успешном сохранении модели."""

model.save('final_cnn_model.keras')
print("Модель сохранена как 'final_cnn_model.keras'.")

"""

---

Оценка модели.

Пояснение:
* Оценка: Оценка производительности модели на тестовой выборке.
* Вывод: Вывод потерь и точности на тестовой выборке."""

loss, accuracy = model.evaluate(test_dataset)
print(f'Потери на тестовой выборке: {loss:.4f}')
print(f'Точность на тестовой выборке: {accuracy*100:.2f}%')

"""

---

Визуализация результатов.

Пояснение:
* Предсказания: Генерация предсказаний модели на тестовой выборке.
* Отчёт классификации: Подробный отчёт о метриках классификации для каждого класса.
* Матрица неточностей: Визуализация матрицы неточностей для анализа ошибок модели.
* Графики обучения: Визуализация динамики точности и потерь на обучении и тесте."""

# Генерация предсказаний
Y_pred = model.predict(test_dataset)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Получение истинных меток
Y_true = np.argmax(Y_test_original, axis=1)

# Отчёт классификации
print("\nОтчет классификации:")
print(classification_report(Y_true, Y_pred_classes))

# Матрица неточностей
conf_matrix = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[chr(i) for i in range(65, 91)],
            yticklabels=[chr(i) for i in range(65, 91)])
plt.ylabel('Истинная метка')
plt.xlabel('Предсказанная метка')
plt.title('Матрица неточностей')
plt.show()

# Визуализация графиков точности и потерь
epochs_range = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(14, 5))

# График точности
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Точность на обучении')
plt.plot(epochs_range, history.history['val_accuracy'], label='Точность на тесте')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.title('График точности')
plt.legend()

# График потерь
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Потери на обучении')
plt.plot(epochs_range, history.history['val_loss'], label='Потери на тесте')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.title('График потерь')
plt.legend()

plt.tight_layout()
plt.show()

"""

---

Запуск TensorBoard.

Пояснение:
Запуск TensorBoard для интерактивного мониторинга процесса обучения."""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs/fit

"""

---

