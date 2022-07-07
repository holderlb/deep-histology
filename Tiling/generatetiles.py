# generatetiles2.py
#
# Usage: python3 generatetiles2.py <image> <annotations>
#
# The program generates tiles from the given <image> according to the given
# <annotations>. The <annotations> file is a GeoJSON-formatted array from
# the QuPath program. Each annotation includes "geometry":"coordinates" of the
# points of a polygon encompassing a region, and "properties":"classification":"name"
# of the pathology of the region. If a tile overlaps one region,
# then it is written to the directory named after that region's pathology. The amount
# of overlap necessary is controlled by the TILE_OVERLAP variable.
#
# The <annotations> file can also be a csv file that contains point coordinates.
# The csv file requires the following headers/columns: pathology, x, y
#
# The tile images are stored in the tiles/<pathology> subdirectory. The tile image
# file name is of the form: <image>_<NNNNN>.jpg, where <NNNNN> is a unique 5-digit,
# 0-padded number assigned to the tile image. The details about the tiles are
# appended to the file tiles/tiles.csv (image, location, pathology, color).
#
# Finally, if the global variable gGenerateTiledImage=True, the program generates
# the image <image>_tiles.tif that shows all the generated tiles as rectangles
# colored according to their pathology.
#
# NOTE: This program does not remove existing tiles, will overwrite
# existing tiles with the same name, and will append tile information to
# an existing tiles/tiles.csv file. This allows you to run the program
# multiple times, once for each image, and collect all the tiles in one place.
#
# Author: Larry Holder, Washington State University

import os
import sys
import json
import numpy as np
from shapely.geometry import Polygon, Point
from skimage.io import imread, imsave
from skimage.draw import rectangle_perimeter,set_color
from skimage.transform import rescale
import cv2
import pandas as pd
import time
from shapely.ops import cascaded_union

TILE_OVERLAP = 0.3 # Fraction of tile that must overlap region
TILE_SIZE = 256 # width and height of tile images
TILE_INCREMENT = TILE_SIZE // 4
# Global image variables
gImage = None
gGenerateOther = False # If True, generates Other tiles that do not overlap any regions

def init_files():
    # Create 'tiles' directory if not already there.
    if not os.path.exists('tiles'):
        os.makedirs('tiles')
    # Create tiles/tiles.csv file if not already there.
    tiles_file = os.path.join('tiles','tiles.csv')
    if not os.path.exists(tiles_file):
        with open(tiles_file,'w') as tf:
            tf.write("image,x,y,width,height,pathology\n")

#def write_tile(tile_num, image_file_name, tile, pathology, color):
def write_tile(num_tiles, image_file_name, tile, pathologies):
    """Write tile image into tile/pathology directory."""
    global gImage, gImageTiles
    # Extract elements of image file name
    base_file_name = os.path.basename(image_file_name)
    base_file_name_noext = os.path.splitext(base_file_name)[0]
    x,y,w,h = tile
    tile_img = gImage[y:(y+h), x:(x+w)]
    tile_file_base_name = base_file_name_noext + '_' + str(num_tiles).zfill(5)
    pathologies = [p.replace(' ','_') for p in pathologies]
    pathologies = [p.replace('*','') for p in pathologies]
    #pathology1 = pathology.replace(' ','_')
    #pathology2 = pathology1.replace('*','')
    # Create pathology directory if not there
    #tile_path = os.path.join('tiles', pathology2)
    #if not os.path.exists(tile_path):
    #    os.makedirs(tile_path)
    tile_file_name = os.path.join('tiles/images/', tile_file_base_name + '.png')
    # Save tile image
    imsave(tile_file_name, tile_img, check_contrast=False)
    # Append tile information to CSV file
    tiles_file = os.path.join('tiles','tiles.csv')
    with open(tiles_file,'a') as tf:
        line = tile_file_base_name
        line += ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)
        line += ',' + " ".join(pathologies)
        #line += ',' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2])
        tf.write(line + '\n')

def parse_qupath_annotations(annotations_file_name):
    print("Reading annotations...", flush=True)
    with open(annotations_file_name) as annotations_file:
        annotations = json.load(annotations_file)
    polygons = []
    pathologies = []
    colors = []
    for annotation in annotations:
        # Check for properly-formatted annotation
        format_okay = True
        if ("geometry" not in annotation) or ("properties" not in annotation):
            format_okay = False
        elif ("type" not in annotation["geometry"]) or ("coordinates" not in annotation["geometry"]) or ("classification" not in annotation["properties"]):
            format_okay = False
        elif ("name" not in annotation["properties"]["classification"]) or ("colorRGB" not in annotation["properties"]["classification"]):
            format_okay = False
        if not format_okay:
            print("Improperly formatted annotation - skipping...")
            continue
        geo_type = annotation["geometry"]["type"]
        if geo_type == 'Polygon':
            coordinates = annotation["geometry"]["coordinates"][0]
        elif geo_type == 'MultiPolygon':
            # Typically one big polygon and a few little ones; use just the big one
            multi_coordinates = annotation["geometry"]["coordinates"]
            lengths = [len(x[0]) for x in multi_coordinates]
            index = lengths.index(max(lengths))
            coordinates = multi_coordinates[index][0]
        else:
            print("Unknown geometry type: " + geo_type)
            continue
            #sys.exit()
        polygon = Polygon(coordinates)
        pathology = annotation["properties"]["classification"]["name"]
        pathology = pathology.replace(' ','_')
        pathology = pathology.replace('*','')
        pathology = pathology.replace('/','')
        pathology = pathology.replace("'",'')
        pathology = pathology.lower()
        colorRGB = annotation["properties"]["classification"]["colorRGB"]
        colorR = (colorRGB >> 16) & 255
        colorG = (colorRGB >> 8) & 255
        colorB = colorRGB & 255
        color = (colorR, colorG, colorB)
        polygons.append(polygon)
        pathologies.append(pathology)
        colors.append(colorRGB)
    return polygons, pathologies, colors

def save_colors(pathologies, colors):
    if os.path.exists('tiles/colors.json'):
        with open("tiles/colors.json", "r") as f:
            color_map = json.load(f)
    else:
        color_map = {}
    for pathology, color in zip(pathologies, colors):
        if pathology not in color_map.keys():
            color_map.update({pathology: color})
    with open("tiles/colors.json", "w") as f:
        json.dump(color_map, f)

def intersects_enough(tile_polygon, polygon1):
    """Returns True if polygon intersects polygon1 by at least TILE_OVERLAP amount."""
    global TILE_SIZE, TILE_OVERLAP
    #if not tile_polygon.intersects(polygon1):
    #    return False
    min_area = TILE_SIZE * TILE_SIZE * TILE_OVERLAP
    intersection = tile_polygon.intersection(polygon1)
    area = intersection.area
    if (area >= min_area) or (area >= polygon1.area*0.99):
        return True    
    return False

def compute_tiles_old(polygons, pathologies):
    """Compute and return tiles (x,y,w,h) for all tiles in image overlapping polygon,
    but not overlapping any others in polygons."""
    global gImage, TILE_SIZE, TILE_INCREMENT
    height, width, channels = gImage.shape
    x1, y1 = 0, 0
    x2, y2 = TILE_SIZE, TILE_SIZE
    tiles = []
    tiles_pathologies = []
    while y2 <= height:
        while x2 <= width:
            tile_pathologies = []
            tile_polygon = Polygon([(x1,y1), (x1,y2), (x2,y2), (x2,y1)])
            for polygon1, pathology1 in zip(polygons, pathologies):
                if pathology1 not in tile_pathologies:
                    if intersects_enough(tile_polygon, polygon1):
                        tile_pathologies.append(pathology1)
            if len(tile_pathologies) > 0:
                tiles_pathologies.append(tile_pathologies)
                tiles.append([x1, y1, TILE_SIZE, TILE_SIZE])
            x1 += TILE_INCREMENT
            x2 += TILE_INCREMENT
        #print("vacuoles:", np.sum(['vacuole' in a for a in tiles_pathologies]), round(y2/height, 2))
        x1 = 0
        x2 = TILE_SIZE
        y1 += TILE_INCREMENT
        y2 += TILE_INCREMENT
    return tiles, tiles_pathologies

def compute_tiles(polygon, polygons, pathologies):
    """Compute and return tiles (x,y,w,h) for all tiles in image overlapping polygon,
    but not overlapping any others in polygons."""
    global gImage, TILE_SIZE, TILE_INCREMENT
    height, width, channels = gImage.shape
    (minx,miny,maxx,maxy) = polygon.bounds
    # Only consider tiles around and inside polygon
    xmin = max(0, (int(minx) - TILE_SIZE))
    ymin = max(0, (int(miny) - TILE_SIZE))
    xmax = min(width, (int(maxx) + TILE_SIZE))
    ymax = min(height, (int(maxy) + TILE_SIZE))
    x1 = xmin
    y1 = ymin
    x2 = x1 + TILE_SIZE
    y2 = y1 + TILE_SIZE
    tiles = []
    tiles_pathologies = []
    while y2 <= ymax:
        while x2 <= xmax:
            tile_pathologies = []
            tile_polygon = Polygon([(x1,y1), (x1,y2), (x2,y2), (x2,y1)])
            for polygon1, pathology1 in zip(polygons, pathologies):
                if pathology1 not in tile_pathologies:
                    if intersects_enough(tile_polygon, polygon1):
                        tile_pathologies.append(pathology1)
            if len(tile_pathologies) > 0:
                tiles_pathologies.append(tile_pathologies)
                tiles.append([x1, y1, TILE_SIZE, TILE_SIZE])
            x1 += TILE_INCREMENT
            x2 += TILE_INCREMENT
        x1 = xmin
        x2 = x1 + TILE_SIZE
        y1 += TILE_INCREMENT
        y2 += TILE_INCREMENT
    return (tiles, tiles_pathologies)

def containsTooMuchBackground(tile):
    """Return True if tile contains more than 70% background color (off white)."""
    global gImage
    x,y,w,h = tile
    tile_img = gImage[y:(y+h), x:(x+w)]
    threshold = int(round(0.7 * h * w))
    lower = (201,201,201)
    upper = (255,255,255)
    bmask = cv2.inRange(tile_img,lower,upper)
    if cv2.countNonZero(bmask) > threshold:
        return True
    return False

def generate_tiles(image_file_name, annotations_file_name):
    num_tiles = 0
    if ".json" in annotations_file_name:
        polygons, pathologies, colors = parse_qupath_annotations(annotations_file_name)
    else:
        print(annotations_file_name.split('.')[-1], "file type not supported")
        exit()
    print("Generating tiles...", flush=True)
    if not os.path.exists('tiles/images/'):
        os.makedirs('tiles/images/')
    save_colors(pathologies, colors)
    for (polygon, pathology) in zip(polygons, pathologies):
        tiles, tiles_pathologies = compute_tiles(polygon, polygons, pathologies)
        for tile, tile_pathologies in zip(tiles, tiles_pathologies):
            num_tiles += 1
            write_tile(num_tiles, image_file_name, tile, tile_pathologies)
    return num_tiles

def main1():
    # Get image and annotations file names
    global gImage
    arg_index = 1
    image_file_name = sys.argv[arg_index]
    annotations_file_name = sys.argv[arg_index+1]
    print("Reading image \"" + image_file_name + "\"...", flush=True)
    gImage = imread(image_file_name)
    # Generate tiles
    init_files()
    num_tiles = generate_tiles(image_file_name, annotations_file_name)
    print("Generated " + str(num_tiles) + " tile images")

if __name__ == '__main__':
    program_start = time.time()
    main1()
    print("Tile generation took:", round(time.time() - program_start), "seconds")

