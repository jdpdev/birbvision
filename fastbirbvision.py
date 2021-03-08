from fastai.vision.all import *
import torch
import PIL
import argparse
import os

parser = argparse.ArgumentParser(description='Build the birbvision model from scratch.')
parser.add_argument('-m', '--model', nargs='?', const='./models/birbmodel.pkl', default='./models/birbmodel.pkl', help="The model to use")
parser.add_argument('image', help="The image to test")
args = parser.parse_args()

def predict_image(modelsrc, imagesrc):
    model = load_learner(modelsrc)
    prediction = model.predict(imagesrc)
    print(prediction)
    print(model.dls.vocab)

if __name__ == '__main__':
    predict_image(args.model, args.image)