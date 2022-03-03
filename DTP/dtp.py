# Deep Tissue Pathology (DTP) Tool (v5)
#
# Usage: python3 dtp.py <image> <tissue_type> <pathology> <gTileSize> [<downscale>]
#
# Requirements: "models" folder must be present in the same directory as dty.py.
#               The classifier is loaded in automatically base on the tissue_type,
#               pathology, and gTileSize arguments.
#
#
# DTP identifies pathology in a given tissue image. The arguments to the script
# are as follows:
# - <image> path to the image to be processed.
# - <tissue_type> testis, prostate, or kidney.
# - <pathology> individual pathology that will be predicted by the classifier.
#   For example, if your tissue type is testis, the optional pathology arguments
#   would be "atrophy", "maturation_arrest", or "vacuole". Automatic pathology
#   assignment is given in the run.sh and run.bat files.
# - <gTileSize> is the tile size used to train the model and the size used
#   to tile the input image. The default is 256.
# - [<downscale>] optional variable for the factor in which the output image is
#   downsampled. Since the output images are very large, this optional variable is
#   usually necessary. This value can be any power of 2 greater than zero. The default
#   value is 4, meaning that the output heatmap image will be 0.25 times the resolution
#   of the original input tif image.
#
# DTP first tiles the image according to the <gTileSize> into non-overlapping
# tiles and then classifies each tile as diseased or not. DTP outputs two files.
# One file <image>_<gTileSize>_dtp.jpg is the original image with diseased tiles
# highlighted.
#
# Authors: Colin Greeley and Larry Holder, Washington State University

import multiprocessing
import os
import sys
import numpy as np
from skimage.io import imread, imsave
from skimage.draw import rectangle_perimeter, set_color
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import tensorflow as tf
import csv
import gc
import time
from pynvml import *
from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()
handle = nvmlDeviceGetHandleByIndex(0)
gpu_mem = nvmlDeviceGetMemoryInfo(handle).free

# Global variables
gTileSize = 256
gTileIncrement = gTileSize
gConfidence = 0.95 # threshold on Prob(diseased) for tile to be classified as diseased
downscale = 4

def rescale(tile_image, size):
    """Rescale given tile image to 256x256, which is what network expects."""
    return cv2.resize(tile_image, (size, size), interpolation=cv2.INTER_AREA)

def contains_too_much_background(tile_image):
    """Return True if image contains more than 70% background color (off white)."""
    h,w,c = tile_image.shape
    threshold = int(round(0.7 * h * w))
    lower = (201,201,201)
    upper = (255,255,255)
    bmask = cv2.inRange(tile_image, lower, upper)
    if cv2.countNonZero(bmask) > threshold:
        return True
    return False

def process_image(image, models):
    """Extracts tiles from image and returns bounding box of all diseased tiles."""
    global gTileSize, gTileIncrement
    height, width, channels = image.shape
    num_tiles = int((height * width) / (gTileIncrement * gTileIncrement))
    tile_count = 0
    x1 = y1 = 0
    x2 = y2 = gTileSize # yes, gTileSize, not (gTileSize - 1)
    pred_tiles = []
    tile_images = []
    locs = []
    while y2 <= height: # yes, <=, not <
        while x2 <= width:
            tile_image = image[y1:y2, x1:x2]
            if (not contains_too_much_background(tile_image)):
                tile_images.append(tile_image)
                locs.append((x1, y1))
            x1 += gTileIncrement
            x2 += gTileIncrement
            tile_count += 1
            if len(tile_images) * 256 * 256 * 3 > gpu_mem * 0.9 or tile_count == num_tiles:
                if len(tile_images) > 0:
                    tile_images, locs = filter_tiles(tile_images, locs, models[0], gTileSize//2)        # ignore filter
                if len(tile_images) > 0:
                    probs = classify_tile(tile_images, locs, models[1])                           # disease classifier
                    pred_tiles.extend([(x1, y1, gTileSize, gTileSize, prob) for ((x1,y1), prob) in zip(locs, probs)])
                tile_images = []
                locs = []
            if (tile_count % 1000) == 0:
                print("  processed " + str(tile_count) + " of " + str(num_tiles) + " tiles", flush=True)
        x1 = 0
        x2 = gTileSize
        y1 += gTileIncrement
        y2 += gTileIncrement
    return pred_tiles

def filter_tiles(tile_images, locs, model, tile_size):
    """Returns the prediction value for all tiles: 0 < p(x) < 1"""
    global gConfidence, gTileSize
    scaled_tile_image = np.array([rescale(tile_image, tile_size) for tile_image in tile_images])
    pred = model.predict(scaled_tile_image)
    new_tiles = [sti for i, sti in enumerate(tile_images) if pred[i,1] > gConfidence]
    new_locs = [l for i, l in enumerate(locs) if pred[i,1] > gConfidence]
    return new_tiles, new_locs

def classify_tile(tile_images, locs, model):
    """Returns the prediction value for all tiles: 0 < p(x) < 1"""
    global gConfidence, gTileSize
    tiles = np.asarray(tile_images)
    pred = model.predict(tiles)
    return pred[:,0]

def highlight_tiles(image, tiles):
    """Draws box on image around each tile (in x,y,w,h format) in tiles."""
    global gConfidence
    thickness = 5
    color = (0,255,0) # green
    for tile in tiles:
        x,y,w,h,p = tile
        if p > gConfidence:
            for offset in range(thickness):
                rr,cc = rectangle_perimeter((y-offset,x-offset),end=(y+h+offset,x+w+offset))
                set_color(image, (rr,cc), color)

def generate_heatmap(image, tiles, colormap, downscale, heatmap_intensity=0.75):
    """Superimposes a heatmap onto the input image generated from the pathology predictions on each tile."""
    global gTileSize, gTileIncrement, gConfidence
    # Downscale to save memory and time
    image = cv2.resize(image, (image.shape[1]//downscale, image.shape[0]//downscale), interpolation=cv2.INTER_AREA)
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    prob_sum = 0
    for x1, y1, x_off, y_off, prob in tiles:
        heatmap[y1//downscale:y1//downscale+y_off//downscale, x1//downscale:x1//downscale+x_off//downscale] += np.uint8(prob * 255) if prob >= gConfidence else 0
        prob_sum += prob
    heatmap = np.clip(heatmap, 0, 255)
    heatmap = cv2.resize(heatmap, (heatmap.shape[1]//(gTileSize//downscale), heatmap.shape[0]//(gTileSize//downscale)), interpolation=cv2.INTER_AREA)
    #heatmap = np.uint8(255 * heatmap)
    color = cm.get_cmap(colormap)
    colors = color(np.arange(256))[:, :3]
    colored_heatmap = colors[heatmap]
    #plt.imshow(colored_heatmap)
    #plt.show()
    colored_heatmap = array_to_img(colored_heatmap)
    colored_heatmap = colored_heatmap.resize((image.shape[1], image.shape[0]))
    colored_heatmap = img_to_array(colored_heatmap)
    image = np.add(colored_heatmap * heatmap_intensity, image)
    ratio = "Ratio of {} to non-{} tissue: {}".format(sys.argv[3], sys.argv[3], (prob_sum/len(tiles) if len(tiles) > 0 else 0))
    return image, ratio

def read_tiles(tiles_file_name):
    tiles = []
    with open(tiles_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            x = int(row[0])
            y = int(row[1])
            w = int(row[2])
            h = int(row[3])
            p = float(row[4])
            tiles.append((x,y,w,h,p))
    return tiles

def main1():
    global gTileSize, gTileIncrement, downscale
    image_file_name = sys.argv[1]
    tissue_type = sys.argv[2]
    pathology = sys.argv[3]
    gTileSize = int(sys.argv[4])
    gTileIncrement = gTileSize
    if len(sys.argv) > 5:
        downscale = int(sys.argv[5])
    highlighting = "plasma"
    image_file_root = os.path.splitext(image_file_name)[0]
    print("Reading image...")
    image = imread(image_file_name)
    print("Loading model...")
    ignore_model = load_model('./models/multiclass_models_v5/{}-{}{}.h5'.format(tissue_type, 'Ignore', gTileSize // 2))
    model = load_model('./models/multiclass_models_v5/{}-{}{}.h5'.format(tissue_type, pathology, gTileSize))
    print("Classifying image...")
    start = time.time()
    tiles = process_image(image, (ignore_model, model))
    print("Processing Time:", round(time.time() - start, 2), "seconds")
    print("Writing diseased tiles CSV file...")
    tile_loc_file_name = image_file_root + "_{}{}_tiles.csv".format(pathology, gTileSize)
    with open(tile_loc_file_name, 'w') as f:
        f.write(",".join(("x", "y", "width", "height", "probability")) + '\n')
        for tile in tiles:
            f.write(",".join([str(x) for x in tile]) + '\n')
    #tiles = read_tiles(image_file_root + "_{}256_tiles.csv".format(pathology))

    print("Highlighting diseased tiles in image...")
    image, ratio = generate_heatmap(image, tiles, highlighting, downscale=downscale)
    gc.collect()
    output_image_file_name = image_file_root + "_{}_dtp.tif".format(pathology)
    output_data_file_name = image_file_root + "_{}_dtp.txt".format(pathology)
    print("Writing highlighted image...")
    with open(output_data_file_name, 'w') as f:
        f.write(ratio + '\n')
        f.write("Predicted {} tiles with confidence > 0.95: {}".format(pathology, [i[-1] > gConfidence for i in tiles].count(True)) + '\n')
        f.write("Predicted {} tiles with confidence > 0.5: {}".format(pathology, [i[-1] > 0.5 for i in tiles].count(True)) + '\n')
    image = array_to_img(image)
    image.save(output_image_file_name, compression="jpeg")
    print("Done.")

if __name__ == "__main__":
    main1()
