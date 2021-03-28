import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Build the birbvision model from scratch.')
parser.add_argument('-i', '--imagesize', type=int, nargs='?', const=1, default=224, help="The size of the image")
parser.add_argument('-m', '--model', choices=['aily', 'imagenet'], const='aily', default="aily", nargs='?', help="Which model to use")
parser.add_argument('file', help="Name of the file to load from birbstorage")
args = parser.parse_args()

IMG_ROOT = "/Volumes/birbstorage/"
IMG_SIZE = args.imagesize

classifier = None
labels_path = None
model_labels = None

classifier = tf.keras.Sequential([
    hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1')
])
labels_path = tf.keras.utils.get_file('aiy_birds_V1_labelmap.csv','https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv')
model_labels = np.array(open(labels_path).read().splitlines())[1:]

img = load_img(IMG_ROOT + args.file, target_size=(IMG_SIZE, IMG_SIZE))
img.show()

imgarray = img_to_array(img) / 255.0
print(imgarray.shape)

prediction = classifier.predict(imgarray[np.newaxis, ...])
print(prediction.shape)

predicted_class = np.argmax(prediction[0], axis=-1)
#print(f'({predicted_class}) {model_labels[predicted_class + 2]}')

def prediction_sort(value):
    return value[0]

all_predictions = zip(prediction[0], model_labels);
all_predictions = list(all_predictions)
all_predictions.sort(key=prediction_sort, reverse=True)

for p, l in all_predictions[0:10]:
    print(f"{l}: {p * 100}")
    #class_id = pred_dict['class_ids'][0]
    #probability = pred_dict['probabilities'][class_id]

    #print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))
