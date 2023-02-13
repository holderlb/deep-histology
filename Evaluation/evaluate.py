# Heatmap Generation
#
# Usage: python3 generate_geatmap.py --image --pathology_colors --tissue_type --pathologies --[tile_size] --[downscale] --[generate_heatmap] --[output_dir]
#
#
# generate_geatmap identifies pathology locations in a given tissue image and highlights the 
# predicted areas in the output heatmap iamge. The arguments to the script are as follows:
# - <image> path to the image to be processed.
# - <pathology_colors> Path to colors file created from "generatetiles.py". Color information for tiles is stored here.
#   Used for generating new Qupath annotations with the same colors as the original annotations.
# - <tissue_type> Name of the tissue type. i.e., "breast" tissue
# - <pathologies> Names of the pathologies you want to classify. Same as pathologies used for training.
# - [<tile_size>] optional variable that is the tile size used to train the model and the size used
#   to tile the input image. The default is 256.
# - [<downscale>] optional variable for the factor in which the output image is
#   downsampled. Since the output images are very large, this optional variable is
#   usually necessary. This value can be any power of 2 greater than zero. The default
#   value is 2, meaning that the output heatmap image will be 0.5 times the resolution
#   of the original input tif image.
# - [generetate_heatmap] optional boolean variable that determines whether a heatmap image will be created
#   and saved for each input image. (Resource intensive).
# - [output_dir] optional file path to specify where all program outputs will go to
#
# generate_geatmap first tiles the image according to the <gTileSize> into non-overlapping
# tiles and then classifies each tile as diseased or not. DTP outputs two files.
# One file <image>_<gTileSize>_dtp.jpg is the original image with diseased tiles
# highlighted.
#
# Authors: Colin Greeley and Larry Holder, Washington State University

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
from tensorflow_addons.metrics import F1Score
from tensorflow_addons.optimizers import SWA, MovingAverage
import tensorflow as tf
import csv
import json
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import geojson
import gc
import time
import argparse
from pynvml import *
from pynvml.smi import nvidia_smi
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    nvsmi = nvidia_smi.getInstance()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_mem = nvmlDeviceGetMemoryInfo(handle).free
else: 
    gpu_mem = 10**10 # GPU not available, assume > 10 GB of RAM

# Global variables
gImage = None
gTileSize = 256
gTileIncrement = gTileSize // 2
gConfidence = 0.5 # threshold on Prob(diseased) for tile to be classified as diseased
downscale = 2
padding = 1536

def rescale(tile_image, size):
    """Rescale given tile image to 256x256, which is what network expects."""
    return cv2.resize(tile_image, (size, size), interpolation=cv2.INTER_AREA)

def contains_too_much_background(tile_image):
    """Return True if image contains more than 70% background color (off white)."""
    d,h,w,c = tile_image.shape
    threshold = int(round(0.7 * h * w))
    lower = (201,201,201)
    upper = (255,255,255)
    bmask = cv2.inRange(tile_image[0], lower, upper)
    if cv2.countNonZero(bmask) > threshold:
        return True
    return False

def pyramid_tile(x, y, k=3):
    global gTileSize, gTileIncrement, gConfidence, gImage, padding
    image_pyramid = []
    for i in range(k):
        x1 = int(x + (gTileSize//2) - ((gTileSize//2) * (2**i)))
        y1 = int(y + (gTileSize//2) - ((gTileSize//2) * (2**i)))
        x2 = int(gTileSize * (2**i))
        y2 = int(gTileSize * (2**i))
        #assert x2 - x1 == int(xi[3] * (2**i))
        image_pyramid.append(cv2.resize(gImage[y1:y1+y2, x1:x1+x2], (int(gTileSize), int(gTileSize)), interpolation=cv2.INTER_AREA))
    return np.array(image_pyramid[::-1])

def process_image(model, pathologies):
    """Extracts tiles from image and returns bounding box of all diseased tiles."""
    global gTileSize, gTileIncrement, gImage, gConfidence, padding
    height, width, channels = gImage.shape
    num_tiles = int(((height*2) * (width*2)) / (gTileIncrement * gTileIncrement))
    tile_count = 0
    x1 = padding
    y1 = padding
    x2 = padding + gTileSize
    y2 = padding + gTileSize # yes, gTileSize, not (gTileSize - 1)
    pred_tiles = [[] for _ in range(len(pathologies))]
    tiles_polygons = [[] for _ in range(len(pathologies))]
    heated_tiles_list = [[] for _ in range(len(pathologies))]
    tissue_tile_count = 0
    tile_polygons = []
    tile_images = []
    locs = []
    print("Total tiles:", num_tiles)
    while y2 <= height - padding: # yes, <=, not <
        while x2 <= width - padding:
            #tile_image = gImage[y1:y2, x1:x2]
            tile_image = pyramid_tile(x1, y1)
            if (not contains_too_much_background(tile_image)):
                tile_images.append(tile_image)
                locs.append((x1, y1))
            x1 += gTileIncrement
            x2 += gTileIncrement
            tile_count += 1
            if len(tile_images) * gTileSize * gTileSize * 3 * 3 > gpu_mem * 0.5:
                if len(tile_images) > 0:
                    probs = classify_tile(tile_images, model)
                    tf.keras.backend.clear_session()
                    gc.collect()
                    tissue_tile_count += np.sum([(p[0]<1-gConfidence) and (any(p[1:]>gConfidence)) for p in np.asarray(probs)])
                    for i in range(1, len(pathologies)):
                        pred_tiles[i].extend([(x1, y1, gTileSize, gTileSize, prob[i]) for ((x1,y1), prob) in zip(locs, probs) if ((prob[i] > gConfidence) and (prob[0] < 1-gConfidence))])
                tile_images = []
                locs = []
        x1 = padding
        x2 = gTileSize + padding
        y1 += gTileIncrement
        y2 += gTileIncrement
    if len(tile_images) > 0:
        probs = classify_tile(tile_images, model)
        tf.keras.backend.clear_session()
        gc.collect()
        tissue_tile_count += np.sum([((p[0]<1-gConfidence) and (any(p[1:]>gConfidence))) == True for p in np.asarray(probs)])
        for i in range(1, len(pathologies)):
            pred_tiles[i].extend([(x1, y1, gTileSize, gTileSize, prob[i]) for ((x1,y1), prob) in zip(locs, probs) if ((prob[i] > gConfidence) and (prob[0] < 1-gConfidence))])
            #tiles_polygons[i].extend([tile_polygon for tile_polygon, prob in zip(tile_polygons, probs) if prob[i] > gConfidence])
            #heated_tiles_list[i].extend([tile_polygon for tile_polygon, prob in zip(heated_tiles[i], probs) if prob[1] > gConfidence])
    return pred_tiles, tiles_polygons, heated_tiles_list, tissue_tile_count

def classify_tile(tile_images, model):
    """Returns the prediction value for all tiles: 0 < p(x) < 1"""
    global gConfidence, gTileSize
    tiles = np.asarray(tile_images)
    pred = model.predict(tiles)
    pred = np.mean(pred, axis=0)
    return pred.tolist()

@tf.function
def classify_tile_grad_cam(tile_images, model, pathologies):
    """Returns the prediction value for all tiles: 0 < p(x) < 1"""
    global gConfidence, gTileSize
    tiles = tf.convert_to_tensor(tile_images)
    #pred = model.predict(tiles)
    with tf.GradientTape(persistent=True) as tape:
        last_conv_layer_output, preds = model(tiles)
        #preds = np.mean(preds, axis=0)
        class_channels = [[pred[:, i] for pred in preds] for i in range(len(pathologies))]
    grads = [tape.gradient(class_channel, last_conv_layer_output) for class_channel in class_channels]
    pooled_grads = [tf.reduce_mean(grad, axis=(0, 1, 2)) for grad in grads]
    heat_tiles = [last_conv_layer_output @ pooled_grad[..., tf.newaxis] for pooled_grad in pooled_grads]
    heat_tiles = [tf.squeeze(heat_tile) for heat_tile in heat_tiles]
    heat_tiles = [tf.maximum(heat_tile, 0) / tf.math.reduce_max(heat_tile) for heat_tile in heat_tiles]
    return preds, heat_tiles

def generate_heatmap(image, tiles, pathology, tissue_tile_count, colormap, downscale, heatmap_intensity=0.75):
    """Superimposes a heatmap onto the input image generated from the pathology predictions on each tile."""
    global gTileSize, gTileIncrement, gConfidence
    downscale1 = gTileIncrement
    heatmap = np.zeros((image.shape[0]//downscale1, image.shape[1]//downscale1))
    image = cv2.resize(image, (image.shape[1]//downscale, image.shape[0]//downscale), interpolation=cv2.INTER_AREA)
    prob_sum = 0
    for (x1, y1, x_off, y_off, prob) in tiles:
        heatmap[y1//downscale1:y1//downscale1+y_off//downscale1, x1//downscale1:x1//downscale1+x_off//downscale1] += 1 if prob >= 0.95 else 0
        prob_sum += 1
    points_list = []
    ds_inc = gTileSize // gTileIncrement
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i, j] >= (ds_inc ** 2) * 0.75:
                points_list.append(Polygon([(j,i),(j+1,i),(j+1,i+1),(j,i+1)]))
    poly_array = unary_union(points_list)
    if type(poly_array) == MultiPolygon:
        polygons = [geom for geom in poly_array.geoms]
    elif type(poly_array) == Polygon:
        polygons = [poly_array]
    else:
        polygons = []
    polygons = [Polygon(np.array(p.exterior.coords) * downscale1) for p in polygons]
    #heatmap = np.clip(heatmap, 0, 1)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    #heatmap = cv2.resize(heatmap, (heatmap.shape[1]//(gTileSize//downscale), heatmap.shape[0]//(gTileSize//downscale)), interpolation=cv2.INTER_AREA).astype('uint8')
    heatmap = np.uint8(255 * heatmap)
    color = cm.get_cmap(colormap)
    colors = color(np.arange(256))[:,:3]
    colored_heatmap = colors[heatmap]
    #plt.imshow(colored_heatmap)
    #plt.show()
    colored_heatmap = array_to_img(colored_heatmap)
    colored_heatmap = colored_heatmap.resize((image.shape[1], image.shape[0]))
    colored_heatmap = img_to_array(colored_heatmap)
    image = np.add(colored_heatmap * heatmap_intensity, image)
    ratio = ["Ratio of {} tiles to all tissue tiles: {}/{}={}".format(pathology, prob_sum, tissue_tile_count, (round(prob_sum/tissue_tile_count, 5) if len(tiles) > 0 else 0))]
    return image, ratio, polygons

def read_tiles(tiles_file_name):
    tiles = []
    with open(tiles_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i > 0:
                x = int(row[0])
                y = int(row[1])
                w = int(row[2])
                h = int(row[3])
                p = float(row[4])
                tiles.append((x,y,w,h,p))
    return tiles

def write_geojson(filename, tiles_polygons, pathologies, colors):
    allobjects = []
    def make_obj():
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": []
            },
            "properties": {
                "object_type": "annotation",
                "classification": {
                    "name": "",
                    "colorRGB": 0
                },
                "isLocked": False
            }
        }
    for i in range(1, len(pathologies)):
        for tile_polygons in tiles_polygons[i]:
            new_obj = make_obj()
            new_obj["geometry"]["coordinates"] = [tile_polygons]
            new_obj["properties"]["classification"]["name"] = pathologies[i].capitalize().replace('_', ' ')
            new_obj["properties"]["classification"]["colorRGB"] = colors[pathologies[i]]
            allobjects.append(new_obj)
    with open(filename, 'w') as outfile:
        json.dump(allobjects, outfile, indent=2)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--pathology_colors', type=str, required=True, help='Path to colors file created from "generatetiles.py". Color information for tiles is stored here.')
    parser.add_argument('--tissue_type', type=str, required=True, help='Name of the tissue type. i.e., "breast" tissue')
    parser.add_argument('--pathologies', nargs="*", type=str, required=True, help='''Name of the pathologies you want to classify. They will be the positive classes for new multiclass classification model. 
                                                                                   Every other class will be treated as the negative class.''')
    parser.add_argument('--tile_size', type=int, required=False, default=256, help='Resolution of tiles used for neural network input')
    parser.add_argument('--downscale', type=float, required=False, default=2, help='''Optional variable for the factor in which the output image is
                                                                                    downsampled. Since the output images are very large, this optional variable is
                                                                                    usually necessary. This value can be any power of 2 greater than zero. The default
                                                                                    value is 2, meaning that the output heatmap image will be 0.25 times the resolution
                                                                                    of the original input tif image.''')
    parser.add_argument('--generate_heatmap', type=bool, required=False, default=False, help='''Optional boolean variable that determines whether a heatmap image will be created
                                                                                             and saved for each input image. (Resource intensive''')
    parser.add_argument('--output_dir', type=str, required=False, default=None, help='Optional output directory for all program outputs.')
    return parser.parse_args()

def main1():
    global gTileSize, gTileIncrement, downscale, gImage, padding
    args = get_args()
    print(args)
    #exit()
    image_file_name = args.image
    color_dir = args.pathology_colors
    with open(color_dir, 'r') as f:
        colors = json.load(f)
    tissue_type = args.tissue_type.lower()
    pathologies = [p.lower() for p in args.pathologies]
    gTileSize = args.tile_size
    downscale = args.downscale
    gen_image = args.generate_heatmap
    output_dir = args.output_dir
    if '/' in tissue_type:
        tissue_type = tissue_type.replace('/', '-')
    if ' ' in tissue_type:
        tissue_type = tissue_type.replace(' ', '-')
    highlighting = "plasma"
    image_file_root = os.path.splitext(image_file_name)[0]
    print("Reading image:", image_file_name)
    gImage = imread(image_file_name)
    #gImage = np.pad(gImage, ((padding, padding), (padding, padding), (0,0)), 'constant', constant_values=0)
    print("Loading model...")
    if len(physical_devices) > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:{}".format(i) for i in range(len(physical_devices))])
        with mirrored_strategy.scope():
            model = load_model('../Training/models/{}{}_classifier_SA_30.h5'.format(tissue_type, gTileSize))
    else:
        model = load_model('../Training/models/{}{}_classifier_SA_30.h5'.format(tissue_type, gTileSize))
    #model.summary()
    print("Classifying image...")
    start = time.time()
    tiles_list, tiles_polygons, heated_tiles_list, tissue_tile_count = process_image(model, pathologies)
    pathologies_polygons = [[] for _ in range(len(pathologies))]
    print("Processing Time:", round(time.time() - start, 2), "seconds")
    print("Writing diseased tiles CSV file...")
    if output_dir is not None:
        image_name = os.path.split(image_file_root)[-1]
        if not os.path.exists(os.path.join(output_dir, image_name)):
            os.mkdir(os.path.join(output_dir, image_name))
        image_file_root = os.path.join(output_dir, image_name, image_name)

    tile_loc_file_name = image_file_root + "_tiles.csv"
    output_json_file_name = image_file_root + "_output_annotations.json"
    output_txt_file_name = image_file_root + "_results.txt"
    #print(tiles_list)
    with open(tile_loc_file_name, 'w') as f:
        f.write(",".join(("pathology", "x", "y", "width", "height", "probability")) + '\n')
        for i in range(1, len(pathologies)):
            for tile in tiles_list[i]:
                f.write(",".join([pathologies[i]] + [str(x) for x in tile]) + '\n')
    if os.path.exists(output_txt_file_name):
        os.remove(output_txt_file_name)

    print("Highlighting diseased tiles in image...")
    for i in range(1, len(pathologies)):
        image, ratios, intersection = generate_heatmap(gImage, tiles_list[i], pathologies[i], tissue_tile_count, highlighting, downscale=downscale)
        for m in intersection:
            if type(m) is Polygon:
                m = make_valid(m)
                if type(m) is Polygon:
                    pathologies_polygons[i].append(list(m.exterior.coords))
                else:
                    val_geom = make_valid(m)
                    for val_m in val_geom:
                        if type(val_m) is Polygon:
                            pathologies_polygons[i].append(list(val_m.exterior.coords))
        with open(output_txt_file_name, 'a') as f:
            f.write("Predicted {} tiles with confidence > 0.95: {}".format(pathologies[i], [j[-1] > 0.95 for j in tiles_list[i]].count(True)) + '\n')
            f.write("Predicted {} tiles with confidence > 0.5: {}".format(pathologies[i], [j[-1] > 0.5 for j in tiles_list[i]].count(True)) + '\n')
            f.write("Predicted {} instances: {}".format(pathologies[i], len(pathologies_polygons[i])) + '\n\n')
        if gen_image:
            print("Writing highlighted image...")
            output_image_file_name = image_file_root + '_' + pathologies[i] + "_heatmap.jpg"
            image = array_to_img(image)
            image.save(output_image_file_name, compression="jpeg", quality=80)
        print("Done.")
    write_geojson(output_json_file_name, pathologies_polygons, pathologies, colors)

if __name__ == "__main__":
    program_start = time.time()
    main1()
    print("Heatmap generation took:", time.time() - program_start)
