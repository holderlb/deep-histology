# Deep Tissue Pathology (DTP) Tool (v4)
#
# Usage: python3 dtp.py <image> <tissue_type> <pathology> <gTileSize> [<highlighting>]
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
# - <highlighting> optional argument for the highlighting type. If no argument is
#   given, then boxes will be drawn around tiles which have been marked positive for
#   disease. All possible arguments are the heatmap options described below.
#   For heatmap options, set the argument to any of the color map names from the
#   following link:
#   <https://matplotlib.org/stable/tutorials/colors/colormaps.html>
#   Recommendations: plasma, gray, cividis
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
from skimage.draw import rectangle_perimeter, set_color
import cv2
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

# Global variables
gTileSize = 256
gTileIncrement = gTileSize
gConfidence = 0.95 # threshold on Prob(diseased) for tile to be classified as diseased
gHighlightImage = True

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
    num_tiles = int((height * width) / (gTileIncrement * gTileIncrement))
    tile_count = 0
    x1 = y1 = 0
    x2 = y2 = gTileSize # yes, gTileSize, not (gTileSize - 1)
    tiles = []
    while y2 <= height: # yes, <=, not <
        while x2 <= width:
            tile_image = image[y1:y2, x1:x2]
            if (not contains_too_much_background(tile_image)):
                #if (classify_tile(tile_image, model) == True):
                prob = classify_tile(tile_image, model)
                tiles.append((x1, y1, gTileSize, gTileSize, prob))
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
    """Returns the prediction value for all tiles: 0 < p(x) < 1"""
    global gConfidence
    scaled_tile_image = rescale(tile_image)
    tiles = np.asarray([scaled_tile_image])
    pred = model.predict(tiles)
    return pred[0][0]

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

def highlight_tiles2(image, tiles, colormap, downscale=4, heatmap_intensity=1.0):
    """Superimposes a heatmap onto the input image generated from the pathology predictions on each tile."""
    global gTileSize, gTileIncrement, gHighlightImage
    # Downscale to save memory and time
    image = cv2.resize(image, (image.shape[1]//downscale, image.shape[0]//downscale), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((image.shape[0], image.shape[1]))
    prob_sum = 0
    for x1, y1, x_off, y_off, prob in tiles:
        canvas[y1//downscale:y1//downscale+y_off//downscale, x1//downscale:x1//downscale+x_off//downscale] += prob
        prob_sum += prob
    canvas = np.clip(canvas, 0, 1)
    canvas = cv2.resize(canvas, (canvas.shape[1]//(gTileSize/2), canvas.shape[0]//(gTileSize/2)), interpolation=cv2.INTER_AREA)
    heatmap = np.uint8(255 * canvas)
    color = cm.get_cmap(colormap)
    colors = color(np.arange(256))[:, :3]
    colored_heatmap = colors[heatmap]
    #plt.imshow(colored_heatmap)
    #plt.show()
    colored_heatmap = array_to_img(colored_heatmap)
    colored_heatmap = colored_heatmap.resize((image.shape[1], image.shape[0]))
    colored_heatmap = img_to_array(colored_heatmap)
    image = colored_heatmap * heatmap_intensity + image
    print("Ratio of {} to non-{}: {}".format(sys.argv[3], sys.argv[3], round(prob_sum/len(tiles), 2)))
    return image

def main1():
    global gTileSize, gTileIncrement, gHighlightImage
    image_file_name = sys.argv[1]
    tissue_type = sys.argv[2]
    pathology = sys.argv[3]
    gTileSize = int(sys.argv[4])
    gTileIncrement = gTileSize
    if len(sys.argv) <= 5:
        highlighting = "boxes"
    else:
        highlighting = sys.argv[5]
    image_file_root = os.path.splitext(image_file_name)[0]
    print("Reading image...")
    image = imread(image_file_name)
    print("Loading model...")
    model = load_model('./models/{}-{}{}.h5'.format(tissue_type, pathology, gTileSize))
    print("Classifying image...")
    tiles = process_image(image, model)
    print("Writing diseased tiles CSV file...")
    tile_loc_file_name = image_file_root + "_{}{}_tiles.csv".format(pathology, gTileSize)
    with open(tile_loc_file_name, 'w') as f:
        for tile in tiles:
            f.write(",".join([str(x) for x in tile]) + '\n')
    if gHighlightImage:
        print("Highlighting diseased tiles in image...")
        if highlighting == "boxes":
            highlight_tiles(image, tiles)
        else:
            image = highlight_tiles2(image, tiles, highlighting)
        output_image_file_name = image_file_root + "_{}{}_tiles.csv".format(pathology, gTileSize)
        print("Writing highlighted image...")
        imsave(output_image_file_name, image)
    print("Done.")

if __name__ == "__main__":
    main1()
