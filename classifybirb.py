import cv2
import numpy as np
import argparse
from tensorflow.lite.python.interpreter import Interpreter
import tensorflow as tf
import tensorflow_hub as hub
import sys

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path="./model/aily/lite-model_aiy_vision_classifier_birds_V1_3.tflite")
labels = load_labels("./model/aily/probability-labels-en.txt")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

def load_and_classify(path):
    image = cv2.imread(args.path)
    return classify_array(image)

def classify_array(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    def result_map(i):
        if floating_model:
            return [labels[i], results[i]]
        else:
            return [labels[i], results[i] / 255.0]

    top_k = results.argsort()[-5:][::-1]
    top_matches = map(result_map, top_k)

    return top_matches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the birbvision model from scratch.')
    parser.add_argument('path', help="Name of the file to load from birbstorage")
    args = parser.parse_args()

    print(f"Classifying: {args.path}")
    load_and_classify(args.path)