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

IMG_ROOT = "/Volumes/birbstorage/thumb/"
IMG_SIZE = args.imagesize

classifier = None
labels_path = None
model_labels = None

if args.model == 'aily':
    classifier = tf.keras.Sequential([
        hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1')
    ])
    labels_path = tf.keras.utils.get_file('aiy_birds_V1_labelmap.csv','https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv')
    model_labels = np.array(open(labels_path).read().splitlines())

    img = load_img(IMG_ROOT + args.file, target_size=(IMG_SIZE, IMG_SIZE))
    img.show()

    imgarray = img_to_array(img) / 255.0
    print(imgarray.shape)

    prediction = classifier.predict(imgarray[np.newaxis, ...])
    print(prediction.shape)

    predicted_class = np.argmax(prediction[0], axis=-1)
    print(f'({predicted_class}) {model_labels[predicted_class + 2]}')
else:
    imagenet_int_to_str = {}

    with open('imagenet21k_wordnet_lemmas.txt', 'r') as f:
        for i in range(1000):
            row = f.readline()
            row = row.rstrip()
            imagenet_int_to_str.update({i: row})

        def preprocess_image(image):
            image = np.array(image)
            # reshape into shape [batch_size, height, width, num_channels]
            img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            image = tf.image.convert_image_dtype(img_reshaped, tf.float32)  
            return image

    # Show the MAX_PREDS highest scoring labels:
    MAX_PREDS = 5
    # Do not show labels with lower score than this:
    MIN_SCORE = 0.8 

    def show_preds(logits, image, correct_flowers_label=None, tf_flowers_logits=False):

        if len(logits.shape) > 1:
            logits = tf.reshape(logits, [-1])

        fig, axes = plt.subplots(1, 2, figsize=(7, 4), squeeze=False)

        ax1, ax2 = axes[0]

        ax1.axis('off')
        ax1.imshow(image)
        if correct_flowers_label is not None:
            ax1.set_title(tf_flowers_labels[correct_flowers_label])
        classes = []
        scores = []
        logits_max = np.max(logits)
        softmax_denominator = np.sum(np.exp(logits - logits_max))
        for index, j in enumerate(np.argsort(logits)[-MAX_PREDS::][::-1]):
            score = 1.0/(1.0 + np.exp(-logits[j]))
            if score < MIN_SCORE: break
            if not tf_flowers_logits:
            # predicting in imagenet label space
                classes.append(imagenet_int_to_str[j])
            else:
            # predicting in tf_flowers label space
                classes.append(tf_flowers_labels[j])
                scores.append(np.exp(logits[j] - logits_max)/softmax_denominator*100)

        ax2.barh(np.arange(len(scores)) + 0.1, scores)
        ax2.set_xlim(0, 100)
        ax2.set_yticks(np.arange(len(scores)))
        ax2.yaxis.set_ticks_position('right')
        ax2.set_yticklabels(classes, rotation=0, fontsize=14)
        ax2.invert_xaxis()
        ax2.invert_yaxis()
        ax2.set_xlabel('Prediction probabilities', fontsize=11)

    module = hub.KerasLayer('https://tfhub.dev/google/experts/bit/r50x1/in21k/bird/1')
    img = load_img(IMG_ROOT + args.file, target_size=(IMG_SIZE, IMG_SIZE))
    img.show()
    processed_img = preprocess_image(img)
    logits = module(processed_img)

    if len(logits.shape) > 1:
        logits = tf.reshape(logits, [-1])

    for index, j in enumerate(np.argsort(logits)[-MAX_PREDS::][::-1]):
        score = 1.0/(1.0 + np.exp(-logits[j]))
        if score > MIN_SCORE:
            print('[Predict]: ', imagenet_int_to_str[j], "(", score, ")")