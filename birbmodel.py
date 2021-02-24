import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Build the birbvision model from scratch.')
parser.add_argument('-e', '--epochs', type=int, nargs='?', const=1, default=15)
parser.add_argument('-b', '--batchsize', type=int, nargs='?', const=1, default=32)
parser.add_argument('-i', '--imagesize', type=int, nargs='?', const=1, default=180)
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batchsize
IMG_SIZE = args.imagesize
AUTOTUNE = tf.data.AUTOTUNE

print("Building model with...")
print("  epochs: ", epochs)
print("  batch size: ", batch_size)
print("  image size: ", IMG_SIZE)

sequential_resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255)
])

def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  return image, label

def augment(image_label, seed):
  image, label = image_label
  image, label = resize_and_rescale(image, label)
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
  # Make a new seed
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  # Random crop back to the original size
  image = tf.image.stateless_random_crop(
      image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  # Random brightness
  image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
  image = tf.clip_by_value(image, 0, 1)
  return image, label

def preprocess_ds(ds, train = False):
  if train:
    ds = ds.shuffle(1000)

  return (
    ds
    .map(augment if train else resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
  )

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

# load caltech_birds2011 dataset
(train_ds, val_ds, test_ds), metadata = tfds.load(
    #'caltech_birds2011',
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features['label'].num_classes
get_label_name = metadata.features['label'].int2str

"""
image, label = next(iter(train_ds))
resized = resize_and_rescale(image)
_ = plt.imshow(resized)
_ = plt.title(get_label_name(label))
plt.show()
"""
#"""

counter = tf.data.experimental.Counter()
train_ds = tf.data.Dataset.zip((train_ds, (counter, counter)))

train_ds = preprocess_ds(train_ds, True)
val_ds = preprocess_ds(val_ds)
test_ds = preprocess_ds(test_ds)
#train_ds = configure_for_performance(train_ds)
#val_ds = configure_for_performance(val_ds)
#test_ds = configure_for_performance(test_ds)

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(IMG_SIZE, 
                                                              IMG_SIZE,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

model = tf.keras.Sequential([
  #sequential_resize_and_rescale,
  #data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history =model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
#"""

#model.save("model/birbmodel")
model.save("model/flowermodel")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
