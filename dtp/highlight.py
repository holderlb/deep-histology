# highlight.py
#
# Usage: python3 highlight.py <image> <tiles>
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
from skimage.io import imread, imsave
from skimage.draw import rectangle_perimeter,set_color
import cv2

def highlight_tiles(image, tiles):
    """Draws box on image around each tile (in x,y,w,h format) in tiles."""
    thickness = 5
    color = (0,255,0) # green
    for tile in tiles:
        x,y,w,h = tile
        for offset in range(thickness):
            rr,cc = rectangle_perimeter((y-offset,x-offset),end=(y+h+offset,x+w+offset))
            set_color(image, (rr,cc), color)

def read_tiles(tiles_file_name):
    tiles = []
    with open(tiles_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            x = int(row[0])
            y = int(row[1])
            w = int(row[2])
            h = int(row[3])
            tiles.append((x,y,w,h))
    return tiles
        
def main1():
    image_file_name = sys.argv[1]
    tiles_file_name = sys.argv[2]
    image_file_root = os.path.splitext(image_file_name)[0]
    print("Reading image...")
    image = imread(image_file_name)
    print("Reading tiles...")
    tiles = read_tiles(tiles_file_name)
    print("Highlighting " + str(len(tiles)) + " tiles in image...")
    highlight_tiles(image,tiles)
    output_image_file_name = image_file_root + "_tiles.tif"
    print("Writing highlighted image...")
    imsave(output_image_file_name, image)
    print("Done.")

if __name__ == "__main__":
    main1()
