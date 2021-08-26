# Deep Tissue Pathology (DTP) Tool (v4)
#
# Usage: python3 dtp.py <image> <model> [<gTileSize>]
#
# DTP identifies pathology in a given tissue image. The arguments to the script
# are as follows:
# - <image> is the image to be processed.
# - <model> is the model to be used to classify the image.
# - <gTileSize> is the tile size used to train the model and the size used
#   to tile the input image. This parameter is optional. The default is 256.
#
# DTP first tiles the image according to the <gTileSize> into non-overlapping
# tiles and then classifies each tile as diseased or not. DTP outputs two files.
# One file <image>_<gTileSize>_dtp.jpg is the original image with diseased tiles
# highlighted. Second file <image>_<gTileSize>_dtp.csv is a list of the diseased
# tiles' bounding boxes using coordinates from the original image. Each line
# contains x,y,w,h where x,y is the upper-left corner and w,h is the width
# and height of the tile.
#
# Authors: Colin Greeley and Larry Holder, Washington State University

import os
import sys
import numpy as np
from skimage.io import imread, imsave
from skimage.draw import rectangle_perimeter,set_color
import cv2
from tensorflow.keras.models import load_model

# Global variables
gTileSize = 256
gTileIncrement = 256
gConfidence = 0.95 # threshold on Prob(diseased) for tile to be classified as diseased
gHighlightImage = False

def rescale(tile_image):
    """Rescale given tile image to 256x256, which is what network expects."""
    return cv2.resize(tile_image, (256, 256), interpolation=cv2.INTER_AREA)

def contains_too_much_background(tile_image):
    """Return True if image contains more than 30% background color (off white)."""
    h,w,c = tile_image.shape
    threshold = int(round(0.3 * h * w))
    lower = (201,201,201)
    upper = (255,255,255)
    bmask = cv2.inRange(tile_image, lower, upper)
    if cv2.countNonZero(bmask) > threshold:
        return True
    return False

def process_image(image, model):
    """Extracts tiles from image and returns bounding box of all diseased tiles."""
    global gTileSize, gTileIncrement
    height, width, channels = image.shape
    num_tiles = int((height * width) / (gTileSize * gTileSize))
    tile_count = 0
    x1 = y1 = 0
    x2 = y2 = gTileSize # yes, gTileSize, not (gTileSize - 1)
    tiles = []
    while y2 <= height: # yes, <=, not <
        while x2 <= width:
            tile_image = image[y1:y2, x1:x2]
            if (not contains_too_much_background(tile_image)):
                if (classify_tile(tile_image, model) == True):
                    tiles.append((x1, y1, gTileSize, gTileSize))
            x1 += gTileIncrement
            x2 += gTileIncrement
            tile_count += 1
            if (tile_count % 100) == 0:
                print("  processed " + str(tile_count) + " of " + str(num_tiles) + " tiles", flush=True)
        x1 = 0
        x2 = gTileSize
        y1 += gTileIncrement
        y2 += gTileIncrement
    return tiles

def classify_tile(tile_image, model):
    """Return True if tile classified by model as diseased, according to global
    confidence parameter; otherwise, return False."""
    global gConfidence
    scaled_tile_image = rescale(tile_image)
    tiles = np.asarray([scaled_tile_image])
    pred = model.predict(tiles)
    return (pred[0][0] > gConfidence)

def highlight_tiles(image, tiles):
    """Draws box on image around each tile (in x,y,w,h format) in tiles."""
    thickness = 5
    color = (0,255,0) # green
    for tile in tiles:
        x,y,w,h = tile
        for offset in range(thickness):
            rr,cc = rectangle_perimeter((y-offset,x-offset),end=(y+h+offset,x+w+offset))
            set_color(image, (rr,cc), color)
        
def main1():
    global gTileSize, gTileIncrement, gHighlightImage
    image_file_name = sys.argv[1]
    model_file_name = sys.argv[2]
    if len(sys.argv) > 3:
        gTileSize = int(sys.argv[3])
        gTileIncrement = gTileSize
    image_file_root = os.path.splitext(image_file_name)[0]
    print("Reading image...")
    image = imread(image_file_name)
    print("Loading model...")
    model = load_model(model_file_name)
    print("Classifying image...")
    tiles = process_image(image, model)
    print("Writing diseased tiles CSV file...")
    tile_loc_file_name = image_file_root + "_{}_tiles.csv".format(gTileSize)
    with open(tile_loc_file_name, 'w') as f:
        for tile in tiles:
            f.write(",".join([str(x) for x in tile]) + '\n')
    if gHighlightImage:
        print("Highlighting diseased tiles in image...")
        highlight_tiles(image,tiles)
        output_image_file_name = image_file_root + "_{}_tiles.tif".format(gTileSize)
        print("Writing highlighted image...")
        imsave(output_image_file_name, image)
    print("Done.")

if __name__ == "__main__":
    main1()
