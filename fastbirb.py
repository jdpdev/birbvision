from fastai.vision.all import *
import torch
import PIL
import cv2

path = untar_data(URLs.CUB_200_2011)
imagepath = path/'images'
files = get_image_files(imagepath)

print("File count: ", len(files))
print(files[3])
print(files[3].name)
print(files[3].parent)
print(files[3].parent.name)


dls = ImageDataLoaders.from_folder(imagepath, valid_pct=0.2, item_tfms=Resize(460),
                                    batch_tfms=aug_transforms(size=224), num_workers=8)
print("valid count: ", len(dls.valid_ds.items))
print("train count: ", len(dls.train_ds.items))
#dls.show_batch()
#plt.show()

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
lr_min, lr_steep = learn.lr_find()
print(f"Min/10: {lr_min:.2e}, steepest: {lr_steep:.2e}")
plt.show()

prediction = learn.predict(files[0])
print(prediction)
learn.show_results()
plt.show()

interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,10))
plt.show()
