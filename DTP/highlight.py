# highlight.py
#
# Usage: python3 highlight.py <image> <tiles> <pathology> [<highlighting>]
#
#
# DTP identifies pathology in a given tissue image. The arguments to the script
# are as follows:
# - <image> path to the image to be highlighted.
# - <tiles> path to *tiles.csv output from dtp.py.
# - <pathology> individual pathology that will be used to name the image.
# - <highlighting> optional argument for the highlighting type. Default is
#   drawing boxes around diseased tiles.
#   For heatmap options, type in any of the colormap names from the following link:
#   https://matplotlib.org/stable/tutorials/colors/colormaps.html
#   Recommendations: plasma, gray, cividis
#
# Program draws a box on the image for each tile in <tiles> file and then writes
# out the highlighted image. The <tiles> file should be a CSV file where each
# line represents a tile of the form x,y,width,height.
#
# Authors: Colin Greeley and Larry Holder, Washington State University

import os
import sys
import csv
import numpy as np
from numpy.core.fromnumeric import compress
#from skimage.io import imread, imsave
from skimage.draw import rectangle_perimeter,set_color
import cv2
from tifffile import imread, imsave
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img


def highlight_tiles(image, tiles, confidence=0.95):
    """Draws box on image around each tile (in x,y,w,h format) in tiles."""
    thickness = 5
    color = (0,255,0) # green
    for tile in tiles:
        x,y,w,h,p = tile
        if p > confidence:
            for offset in range(thickness):
                rr,cc = rectangle_perimeter((y-offset,x-offset),end=(y+h+offset,x+w+offset))
                set_color(image, (rr,cc), color)


def highlight_tiles2(image, tiles, colormap, downscale=4, heatmap_intensity=1):
    """Superimposes a heatmap on to the input image generated from the pathology predictions on each tile."""
    # Downscale to save memory and time
    image = cv2.resize(image, (image.shape[1]//downscale, image.shape[0]//downscale), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((image.shape[0], image.shape[1]))
    prob_sum = 0
    for x1, y1, x_off, y_off, prob in tiles:
        canvas[y1//downscale:y1//downscale+y_off//downscale, x1//downscale:x1//downscale+x_off//downscale] += prob
        prob_sum += prob
    canvas = np.clip(canvas, 0, 1)
    canvas = cv2.resize(canvas, (canvas.shape[1]//128, canvas.shape[0]//128), interpolation=cv2.INTER_AREA)
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
    image_file_name = sys.argv[1]
    tiles_file_name = sys.argv[2]
    pathology = sys.argv[3]
    if len(sys.argv) <= 4:
        highlighting = "boxes"
    else:
        highlighting = sys.argv[4]
    image_file_root = os.path.splitext(image_file_name)[0]
    print("Reading image...")
    image = imread(image_file_name)
    print("Reading tiles...")
    tiles = read_tiles(tiles_file_name)
    print("Highlighting " + str(len(tiles)) + " tiles in image...")
    if highlighting == "boxes":
        highlight_tiles(image, tiles)
    else:
        image = highlight_tiles2(image, tiles, highlighting)
    output_image_file_name = image_file_root + "_{}_tiles.tif".format(pathology)
    print("Writing highlighted image...")
    image = array_to_img(image)
    image.save(output_image_file_name)
    #imsave(output_image_file_name, image, compress=2)
    print("Done.")

if __name__ == "__main__":
    main1()
