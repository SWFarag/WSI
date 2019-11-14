from __future__ import print_function

import sys
import os
import random
import openslide
import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

import argparse
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

#%matplotlib inline
from PIL import Image
from matplotlib.pyplot import imshow
from IPython.display import HTML, display


def main():
    createInputdata(args.in_images, args.in_labels,args.output, args.mult, args.level, args.data_type)

def createInputdata(imagesFolderPath, imagesTargetsPath, output, mult, level, data_type):
    #### for all Slides #############
    slides = []
    for folder in listdir(imagesFolderPath):
        slides.extend([ join(join(imagesFolderPath, folder), f) for f in listdir(join(imagesFolderPath, folder)) if isfile(join(join(imagesFolderPath, folder), f))])
    labels = pd.read_csv(imagesTargetsPath)
    targets = list(labels["target"])

    if (data_type==0):
        #### Create TrainingSet, ValidationSet, TestSet #############
        X_train, X_test, y_train, y_test = train_test_split(slides, targets, test_size=0.20, random_state=42)
        X_train_modeling, X_val, y_train_modeling, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

        print("All_slides: ", len(slides))
        print("All_slides_no_testset: ", len(X_train))
        print("All_slides_no_testset_no_validationset: ", len(X_train_modeling))

        ###### Final trainingSet #######
        grid = []
        for wsi_path in X_train_modeling:
            wsi = openslide.OpenSlide(wsi_path)
            grid.append(wsi.level_dimensions)
            wsi.close
        print("Trainingset_grid: ", len(grid))
        train_lib = {"slides":X_train_modeling, "grid":grid, "targets":y_train_modeling, "mult":mult, "level":level}
        torch.save(train_lib, output + "/" +"MIL_trainingset")

        ###### Final ValidationSet #######
        grid = []
        for wsi_path in X_val:
            wsi = openslide.OpenSlide(wsi_path)
            grid.append(wsi.level_dimensions)
            wsi.close
        print("Validationset_grid: ", len(grid))
        val_lib = {"slides":X_val, "grid":grid, "targets":y_val, "mult":mult, "level":level}
        torch.save(val_lib, output + "/" + "MIL_validationset")

        ###### Final TestSet #######
        grid = []
        for wsi_path in X_test:
            wsi = openslide.OpenSlide(wsi_path)
            grid.append(wsi.level_dimensions)
            wsi.close
        print("Testset_grid: ", len(grid))
        test_lib = {"slides":X_test, "grid":grid, "targets":y_test, "mult":mult, "level":level}
        torch.save(test_lib, output + "/" + "MIL_testset")
    else:
        grid = []
        for wsi_path in slides:
            wsi = openslide.OpenSlide(wsi_path)
            grid.append(wsi.level_dimensions)
            wsi.close
        print("Testset_grid: ", len(grid))
        test_lib = {"slides":slides, "grid":grid, "targets":, targets, "mult":mult, "level":level}
        torch.save(test_lib, output + "/" + "MIL_testset")

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 RNN aggregator training script')
parser.add_argument('--in_images', type=str, default='', help='path to the folder with all the images')
parser.add_argument('--in_labels', type=str, default='', help='path to the csv file with all the targets')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--mult', type=int, default=1, help='scale factor (float) Usually 1. for no scaling')
parser.add_argument('--level', type=int, default=0, help='WSI pyramid level (integer) from which to read the tiles. Usually 0 for the highest resolution.')
parser.add_argument('--dataType', type=int, default=0, help='Reflect nature of the data, is it a training set (0) or a test set (1), default=0')

args = parser.parse_args()

if __name__ == '__main__':
    main()
