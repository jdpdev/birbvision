import numpy as np
import os
import PIL
import PIL.Image
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Build the birbvision model from scratch.')
parser.add_argument('-i', '--imagesize', type=int, nargs='?', const=1, default=224, help="The size of the image")
args = parser.parse_args()

IMG_ROOT = "/Volumes/birbstorage/active_training/"
IMG_SIZE = args.imagesize
BATCH_SIZE = 32

def visualize_model_predictions(model, image_batch, class_names):
    predicted_batch = model.predict(image_batch)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id + 1]

    print(predicted_batch)
    print(predicted_id)

    plt.figure(figsize=(10,9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(predicted_label_batch[n].title())
        plt.axis('off')
    _ = plt.suptitle("Model predictions")
    plt.show()

labels_path = tf.keras.utils.get_file('aiy_birds_V1_labelmap.csv','https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv')
model_labels = np.array(open(labels_path).read().splitlines())

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  str(IMG_ROOT),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_SIZE, IMG_SIZE),
  batch_size=BATCH_SIZE)

class_names = np.array(train_ds.class_names)
num_classes = len(class_names)
print("Training on classes: ", class_names)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
  print("Image shape: ", image_batch.shape)
  print("Label shape: ", labels_batch.shape)
  break

classifier_layer = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1', trainable=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

classifier_batch = classifier_layer(image_batch)
print("Classifier shape: ", classifier_batch.shape)

model = tf.keras.Sequential([
  classifier_layer,
  tf.keras.layers.Dense(num_classes)
])

model.summary()

visualize_model_predictions(model, image_batch, model_labels)

"""
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

batch_stats_callback = CollectBatchStats()

history = model.fit(train_ds, epochs=2,
                    callbacks=[batch_stats_callback])

visualize_model_predictions(model, image_batch, model_labels)
"""