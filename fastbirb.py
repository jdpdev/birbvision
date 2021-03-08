from fastai.vision.all import *
import torch
import PIL
import argparse
import os

parser = argparse.ArgumentParser(description='Build the birbvision model from scratch.')
parser.add_argument('-s', '--save', nargs='?', const=None, default='./models/birbmodel.pkl', help="Where to save the learned model")
parser.add_argument('-d', '--directory', nargs='?', const=None, default=None, help="Directory to load teaching images from")
parser.add_argument('-e', '--epochs', nargs='?', type=int, const=1, default=1, help="Number of epochs to run")
parser.add_argument('-e2', '--epochs2', nargs='?', type=int, const=1, default=0, help="Number of epochs to run after unfreezing")
parser.add_argument('-lf', '--learnfind', nargs='?', const=True, default=False, help="Whether to find the best learn rate")
parser.add_argument('-lr', '--lossrate', nargs='?', const=None, default=None, help="Loss rate")
parser.add_argument('-lr2', '--lossrate2', nargs='?', const=None, default=None, help="Loss rate after unlocking")
args = parser.parse_args()

print("Save to: ", args.save)
print("Model source: ", args.directory)
print("Epochs: ", args.epochs)
print("lr_find: ", args.learnfind)
print("Loss Rate: ", args.lossrate)

def build_model(source, destination, epochs, doLearnFind, lossRate, epochs2, lossRate2):
    torch.backends.cudnn.enabled = False

    if source is None:
        path = untar_data(URLs.CUB_200_2011)
        imagepath = path/'images'
    else:
        imagepath = source
    
    files = get_image_files(imagepath)

    print("File count: ", len(files))
    #print(files[3])
    #print(files[3].name)
    #print(files[3].parent)
    #print(files[3].parent.name)


    dls = ImageDataLoaders.from_folder(imagepath, valid_pct=0.2, item_tfms=Resize(460),
                                        batch_tfms=aug_transforms(size=224), num_workers=0)
    print("valid count: ", len(dls.valid_ds.items))
    print("train count: ", len(dls.train_ds.items))
    #dls.show_batch()
    #plt.show()

    learn = cnn_learner(dls, resnet34, metrics=error_rate, cbs=[SaveModelCallback(), ReduceLROnPlateau()], path='./models').to_fp16()
    #learn = cnn_learner(dls, resnet34, loss_func=CrossEntropyLossFlat(), metrics=accuracy, pretrained=False, cbs=[SaveModelCallback(), ReduceLROnPlateau()], path='./models')    #learn = Learner(dls, xresnet34(n_out=10), metrics=accuracy)

    if doLearnFind:
        lr_min, lr_steep = learn.lr_find()
        print(f"[1] Min/10: {lr_min:.2e}, steepest: {lr_steep:.2e}")
        plt.show()
    elif lossRate is None:
        #learn.fit_one_cycle(epochs)
        learn.fine_tune(epochs)
    else:
        learn.fit_one_cycle(epochs, float(lossRate))
        #learn.fine_tune(epochs, base_lr=float(lossRate))

    if epochs2 > 0:
        learn.unfreeze()
        
        if doLearnFind:
            lr_min, lr_step = learn.lr_find()
            print(f"[2] Min/10: {lr_min:.2e}, steepest: {lr_steep:.2e}")
            plt.show()
        else:
            if lossRate2 is None:
                lr_min, lr_step = learn.lr_find()
                print(f"New LR: {lr_min}")
                learn.fit_one_cycle(epochs2, float(lr_min))
            else:
                learn.fit_one_cycle(epochs2, float(lossRate2))
    
    #prediction = learn.predict(files[0])
    #print(prediction)
    #learn.show_results()
    #plt.show()

    #interp = Interpretation.from_learner(learn)
    #interp.plot_top_losses(9, figsize=(15,10))
    #plt.show()

    if destination != None:
        learn.export(os.path.abspath(destination))

if __name__ == '__main__':
    build_model(args.directory, args.save, args.epochs, args.learnfind, args.lossrate, args.epochs2, args.lossrate2)
