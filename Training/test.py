"""
Model Training

Usage: python3 test.py --tile_dir --tissue_type --pathologies --[tile_size] --[batch_size]

Used to train tile classifying models. The arguments to the script
are as follows:
- <tile_dir> Path to directory of tiles for training
- <tissue_type> help='Name of the tissue type. i.e., "breast" tissue
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
import cv2


def get_data(base_dir, classes, tile_size):
    image_dir = base_dir + ('images/' if base_dir[-1] == '/' else '/images/')
    df = pd.read_csv(base_dir + ('tiles.csv' if base_dir[-1] == '/' else '/tiles.csv'))
    images = []
    labels = []
    for image_name, ps in zip(df["image"], df["pathology"]):
        p_list = [p.lower() for p in ps.split(' ')]
        if not ("ignore" in p_list and "ignore" not in classes):
            images.append(cv2.resize(cv2.imread(image_dir + image_name + '.png'), (tile_size, tile_size), interpolation=cv2.INTER_AREA))
            label = [0 for _ in range(len(classes)+1)]
            for pathology in p_list:
                if pathology in classes:
                    label[classes.index(pathology)] = 1
                else:
                    label[-1] = 1
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    for i in range(len(classes)):
        print(classes[i] + " images:", np.sum(labels[:,i]))
    print("negative class images:", np.sum(labels[:,len(classes)]))
    return images, labels


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_dir', type=str, required=True, help='Path to directory of tiles for training')
    parser.add_argument('--tissue_type', type=str, required=True, help='Name of the tissue type. i.e., "breast" tissue')
    parser.add_argument('--pathologies', nargs="*", type=str, required=True, help='Name of the pathologies you want to classify. They will be the positive classes for new multiclass classification model. Every other class will be treated as the negative class.')
    parser.add_argument('--tile_size', type=int, required=False, default=256, help='Resolution of tiles used for neural network input')
    parser.add_argument('--batch_size', type=int, required=False, default=50, help='Batch size for training neural networks')
    parser.add_argument('--ensemble_size', type=int, required=False, default=5, help='Number of classifiers in ensemble')
    return parser.parse_args()

if __name__ == "__main__":


    args = get_args()
    base_dir = args.tile_dir
    tissue_type = args.tissue_type
    pathologies = [p.lower() for p in args.pathologies]
    tile_size = args.tile_size
    batch_size = args.batch_size
    ensemble_size = args.ensemble_size
    if '/' in tissue_type:
        tissue_type = tissue_type.replace('/', '-')
        
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Available GPUs:", len(physical_devices))
    print("Tissue type:", tissue_type)
    print("Pathologies:", pathologies)
        
    images, labels = get_data(base_dir, pathologies, tile_size)
    classifier = Classifier(image_rez=(tile_size,tile_size), batch_size=batch_size, ensemble_size=ensemble_size)
    classifier.name = tissue_type + str(tile_size) + '_classifier'
    classifier.test(images, labels, class_labels=pathologies + ["normal_tissue"])
