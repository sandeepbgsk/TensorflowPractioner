import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

train_dir = r'C:\Users\bhgsk\.keras\datasets\flower_photos\train'
val_dir = r'C:\Users\bhgsk\.keras\datasets\flower_photos\val'
image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, zoom_range=0.5, rotation_range=45,
                               width_shift_range=0.15, height_shift_range=0.15)
batch_size = 100
IMAGE_SIZE = 150
train_data_gen = image_gen.flow_from_directory(train_dir, batch_size=batch_size, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               shuffle=True, class_mode='sparse')
val_data_gen = image_gen.flow_from_directory(val_dir, batch_size=batch_size, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             class_mode='sparse')
layers = [
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
]
model = tf.keras.models.Sequential(layers)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=40,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size)))
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(40)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.savefig('FlowerClass_Acc')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig('FlowerClass_Loss')

model.save('flower_classification.h5')
