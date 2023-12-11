"""
Model Training

Usage: python3 train.py --tile_dir --tissue_type --pathologies --[tile_size] --[batch_size]

Used to train tile classifying models. The arguments to the script
are as follows:
- <tile_dir> Path to directory of tiles for training
- <tissue_type> Name of the tissue type. i.e., "breast" tissue
- <pathologies> Names of the pathologies you want to classify
- [<tile_size>] (Optional) Resolution of tiles used for neural network input
- [<batch_size] (Optional) Batch size for training neural networks


Authors: Colin Greeley and Larry Holder, Washington State University
"""

import tensorflow as tf
from image_classifier import Classifier
import argparse
import pandas as pd
import numpy as np
import os
from skimage.io import imread, imsave


def get_data(base_dir, classes, tile_size):
    image_dir = base_dir + ('images/' if base_dir[-1] == '/' else '/images/')
    df = pd.read_csv(base_dir + ('tiles.csv' if base_dir[-1] == '/' else '/tiles.csv'))
    images = []
    labels = []
    for image_name, x, y, width, height, ps in zip(df['image'], df['x'], df['y'], df['width'], df['height'], df["pathology"]):
        #if "#017" in image_name:
        p_list = [p.lower() for p in ps.split(' ')]
        if not ("ignore" in p_list and "ignore" not in classes):
            if True: #"B3" not in image_name:
                images.append(("".join(image_name.split('_')[:-1]), x, y, width, height))
                label = [0 for _ in range(len(classes)+1)]
                for pathology in p_list:
                    if pathology.lower() in classes:
                        label[classes.index(pathology)] = 1
                    else:
                        label[-1] = 1
                labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    for i in range(len(classes)):
        print(classes[i] + " images:", np.sum(labels[:,i]))
    print("negative class images:", np.sum(labels[:,len(classes)]))
    return images, labels, set(df['image'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_dir', type=str, required=True, help='Path to directory of tiles for training')
    parser.add_argument('--tissue_type', type=str, required=True, help='Name of the tissue type. i.e., "breast" tissue')
    parser.add_argument('--pathologies', nargs="*", type=str, required=True, help='Name of the pathologies you want to classify. They will be the positive classes for new multiclass classification model. Every other class will be treated as the negative class.')
    parser.add_argument('--tile_size', type=int, required=False, default=256, help='Resolution of tiles used for neural network input')
    parser.add_argument('--batch_size', type=int, required=False, default=77, help='Batch size for training neural networks')
    parser.add_argument('--ensemble_size', type=int, required=False, default=7, help='Number of classifiers in ensemble')
    parser.add_argument('--overlap', type=str, required=False, default=30, help='Tile overlap required for tile to be classifier as positive class')
    return parser.parse_args()

if __name__ == "__main__":


    args = get_args()
    base_dir = args.tile_dir
    tissue_type = args.tissue_type
    pathologies = [p.lower() for p in args.pathologies]
    tile_size = args.tile_size
    batch_size = args.batch_size
    ensemble_size = args.ensemble_size
    overlap = args.overlap
    image_dir = "./qupath/" + tissue_type + "/tif/"
    if '/' in tissue_type:
        tissue_type = tissue_type.replace('/', '-')
        
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Available GPUs:", len(physical_devices))
    print("Tissue type:", tissue_type)
    print("Pathologies:", pathologies)
    
    image_data, labels, image_names = get_data(base_dir, pathologies, tile_size)
    images = {}
    for d in os.listdir(image_dir):
        if True: #"B3" not in os.path.join(image_dir, d):
            if os.path.splitext(d)[-1] == ".tif":
                new_im = np.pad(imread(os.path.join(image_dir, d)), ((1000, 1000), (1000, 1000), (0,0)), 'constant', constant_values=0)
                images.update({os.path.splitext(d)[0]: new_im})
            
            
    classifier = Classifier(image_rez=(tile_size,tile_size), batch_size=batch_size, ensemble_size=ensemble_size)
    classifier.name = tissue_type + str(tile_size) + '_classifier_SA_' + str(overlap)
    classifier.train((images, image_data), labels)
