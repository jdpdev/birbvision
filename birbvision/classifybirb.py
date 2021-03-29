import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import os

class ClassifyResult():
    def __init__(self, label, confidence, lastConfidence):
        self._label = label
        self._confidence = confidence

        if lastConfidence != None:
            self._confidenceDelta = confidence - lastConfidence
        else:
            self._confidenceDelta = None

    @property
    def label(self):
        return self._label

    @property
    def confidence(self):
        return self._confidence

    @property
    def confidenceDelta(self):
        return self._confidenceDelta

class ClassifyResultSet():
    def __init__(self, results, labels):
        def result_map(i):
            if results[i] <= 1:
                return (labels[i], results[i])
            else:
                return (labels[i], results[i] / 255.0)

        sort = results.argsort()
        self.results = list(map(result_map, sort))

    def get_top_results(self, count):
        resultRange = range(count)
        return list(map(lambda i: self.get_result(i), resultRange))

    def get_result(self, index) -> ClassifyResult:
        if index >= len(self.results) - 1:
            return None

        index = index + 1

        result = self.results[-index]
        last = self.results[-(index + 1)]

        return ClassifyResult(result[0], result[1], last[1])

class ClassifyBird():
    def __init__(self):
        model_path = f"{os.path.dirname(__file__)}/aiy/lite-model_aiy_vision_classifier_birds_V1_3.tflite"
        label_path = f"{os.path.dirname(__file__)}/aiy/probability-labels-en.txt"

        print(f"[model_path] {model_path}")
        print(f"[label_path] {label_path}")

        self.interpreter = Interpreter(model_path=model_path)
        self.labels = self.load_labels(label_path)

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

    def load_labels(self, filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def classify_path(self, path):
        image = cv2.imread(path)
        return self.__classify_array(image)

    def classify_image(self, image):
        return self.__classify_array(image)

    def __classify_array(self, image) -> ClassifyResultSet:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        results = np.squeeze(output_data)

        return ClassifyResultSet(results, self.labels)