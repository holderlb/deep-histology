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

TILE_OVERLAP = 0.3 # Fraction of tile that must overlap region
TILE_SIZE = 256 # width and height of tile images
TILE_INCREMENT = TILE_SIZE // 4

# Global image variables
gImage = None
gGenerateTiledImage = False # If True, generates image overlaid with tiles
gGenerateOther = False # If True, generates Other tiles that do not overlap any regions

def init_files():
    # Create 'tiles' directory if not already there.
    if not os.path.exists('tiles'):
        os.makedirs('tiles')
    # Create tiles/tiles.csv file if not already there.
    tiles_file = os.path.join('tiles','tiles.csv')
    if not os.path.exists(tiles_file):
        with open(tiles_file,'w') as tf:
            tf.write("image,x,y,width,height,pathology,red,green,blue\n")

def write_tile(tile_num, image_file_name, tile, pathology, color):
    """Write tile image into tile/pathology directory."""
    global gImage, gImageTiles
    # Extract elements of image file name
    base_file_name = os.path.basename(image_file_name)
    base_file_name_noext = os.path.splitext(base_file_name)[0]
    x,y,w,h = tile
    tile_img = gImage[y:(y+h), x:(x+w)]
    tile_file_base_name = base_file_name_noext + '_' + str(tile_num).zfill(5)
    pathology1 = pathology.replace(' ','_')
    pathology2 = pathology1.replace('*','')
    # Create pathology directory if not there
    tile_path = os.path.join('tiles', pathology2)
    if not os.path.exists(tile_path):
        os.makedirs(tile_path)
    tile_file_name = os.path.join(tile_path, tile_file_base_name + '.png')
    # Save tile image
    imsave(tile_file_name, tile_img, check_contrast=False)
    # Append tile information to CSV file
    tiles_file = os.path.join('tiles','tiles.csv')
    with open(tiles_file,'a') as tf:
        line = base_file_name_noext
        line += ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)
        line += ',' + pathology2
        line += ',' + str(color[0]) + ',' + str(color[1]) + ',' + str(color[2])
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
            sys.exit()
        polygon = Polygon(coordinates)
        pathology = annotation["properties"]["classification"]["name"]
        pathology = pathology.replace(' ','_')
        pathology = pathology.replace('*','')
        pathology = pathology.replace('/','')
        pathology = pathology.replace("'",'')
        colorRGB = annotation["properties"]["classification"]["colorRGB"]
        colorR = (colorRGB >> 16) & 255
        colorG = (colorRGB >> 8) & 255
        colorB = colorRGB & 255
        color = (colorR, colorG, colorB)
        polygons.append(polygon)
        pathologies.append(pathology)
        colors.append(color)
    return polygons, pathologies

def parse_csv_annotations(annotations_file_name, polygon_radius=20):
    df = pd.read_csv(annotations_file_name)
    pathologies = df[df.keys()[0]].to_list()
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    coords = np.array(zip(x, y))
    polygons = []
    for coord in coords:
        polygons.append(Point(coord[1], coord[0]).buffer(polygon_radius))
    return polygons, pathologies

def intersects_one(polygon, polygon1, polygons):
    """Returns True if polygon intersects polygon1 by at least TILE_OVERLAP amount, but none of (polygons - polygon1)."""
    global TILE_SIZE, TILE_OVERLAP
    if not polygon.intersects(polygon1):
        return False
    min_area = TILE_SIZE * TILE_SIZE * TILE_OVERLAP
    intersection = polygon.intersection(polygon1)
    area = intersection.area
    if (area < min_area) and (area != polygon1.area):
        return False
    for p in polygons:
        if polygon.intersects(p) and (p is not polygon1):
            return False
    return True

def intersects_enough(polygon, polygon1):
    """Returns True if polygon intersects polygon1 by at least TILE_OVERLAP amount."""
    global TILE_SIZE, TILE_OVERLAP
    if not polygon.intersects(polygon1):
        return False
    min_area = TILE_SIZE * TILE_SIZE * TILE_OVERLAP
    intersection = polygon.intersection(polygon1)
    intersection_area = intersection.area
    if (intersection_area >= min_area) or (intersection_area == polygon1.area):
        return True    
    return False

def intersects_none(polygon, polygons):
    """Returns True if polygon intersects none of polygons."""
    for p in polygons:
        if polygon.intersects(p):
            return False
    return True

def intersects_bounds_none(polygon, polygons):
    """Returns True if polygon intersects none of the bounding boxes of polygons."""
    for p in polygons:
        (x1,y1,x2,y2) = p.bounds
        p1 = Polygon([(x1,y1), (x1,y2), (x2,y2), (x2,y1)])
        if polygon.intersects(p1):
            return False
    return True

def tile_overlaps(tile, tiles):
    """Returns True if tile does not overlap any in tiles."""
    x,y,w,h = tile
    tile_poly = Polygon([(x,y),(x,y+h),(x+w,y+w),(x+w,y)])
    for tile2 in tiles:
        x,y,w,h = tile2
        tile2_poly = Polygon([(x,y),(x,y+h),(x+w,y+w),(x+w,y)])
        if tile_poly.intersects(tile2_poly):
            return True
    return False

def compute_tiles(polygon, polygons, tile_size, tile_increment):
    """Compute and return tiles (x,y,w,h) for all tiles in image overlapping polygon,
    but not overlapping any others in polygons."""
    global gImage
    height, width, channels = gImage.shape
    (minx,miny,maxx,maxy) = polygon.bounds
    # Only consider tiles around and inside polygon
    xmin = max(0, (int(minx) - tile_size))
    ymin = max(0, (int(miny) - tile_size))
    xmax = min((width - tile_size), int(maxx))
    ymax = min((height - tile_size), int(maxy))
    x1 = xmin
    y1 = ymin
    x2 = x1 + tile_size
    y2 = y1 + tile_size
    tiles = []
    while y1 <= ymax:
        while x1 <= xmax:
            tile_polygon = Polygon([(x1,y1), (x1,y2), (x2,y2), (x2,y1)])
            if intersects_one(tile_polygon, polygon, polygons):
                tile = [x1, y1, tile_size, tile_size]
                tiles.append(tile)
            x1 += tile_increment
            x2 += tile_increment
        x1 = xmin
        x2 = x1 + tile_size
        y1 += tile_increment
        y2 += tile_increment
    return tiles

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

def add_colored_rectangles(tiles_and_colors):
    global gImage
    for tile_and_color in tiles_and_colors:
        tile = tile_and_color[0]
        color = tile_and_color[1]
        x,y,w,h = tile
        draw_rectangle(gImage, x, y, w, h, color, thickness=5)
        
def draw_rectangle(img, x, y, w, h, color, thickness=1):
    """Draws a rectangle on the image according to the given parameters. Color is in RGB."""
    if img.shape[2] == 4:
        color = color + (255,) # add alpha channel
    for offset in range(thickness):
        rr,cc = rectangle_perimeter((y-offset,x-offset),end=(y+h+offset,x+w+offset))
        set_color(img, (rr,cc), color)

def generate_tiles(image_file_name, annotations_file_name, tile_size, tile_increment):
    global gGenerateTiledImage, OTHER_PATHOLOGY, OTHER_COLOR
    num_tiles = 0
    if ".json" in annotations_file_name:
        polygons, pathologies = parse_qupath_annotations(annotations_file_name)
    elif ".csv" in annotations_file_name:
        polygons, pathologies = parse_csv_annotations(annotations_file_name)
    else:
        print(annotations_file_name.split('.')[-1], "file type not supported")
        exit()
    print("Generating tiles...", flush=True)
    for (polygon,pathology) in zip(polygons,pathologies):
        tiles = compute_tiles(polygon, polygons, tile_size, tile_increment)
        for tile in tiles:
            num_tiles += 1
            write_tile(num_tiles, image_file_name, tile, pathology)
    return num_tiles

def main1():
    # Get image and annotations file names
    global gImage, gGenerateTiledImage, TILE_SIZE, TILE_INCREMENT
    arg_index = 1
    image_file_name = sys.argv[arg_index]
    annotations_file_name = sys.argv[arg_index+1]
    print("Reading image \"" + image_file_name + "\"...", flush=True)
    gImage = imread(image_file_name)
    # Generate tiles
    init_files()
    num_tiles = generate_tiles(image_file_name, annotations_file_name, TILE_SIZE, TILE_INCREMENT)
    print("Generated " + str(num_tiles) + " tile images")
    if gGenerateTiledImage:
        base_file_name = os.path.basename(image_file_name)
        base_file_name_noext = os.path.splitext(base_file_name)[0]
        tile_image_file_name = base_file_name_noext + '_tiles.tif'
        print("Writing tiled image to " + tile_image_file_name + " ...", flush=True)
        imsave(tile_image_file_name, gImage, check_contrast=False)
    print("Done.")

if __name__ == '__main__':
    main1()

