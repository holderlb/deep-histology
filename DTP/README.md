# Deep Tissue Pathology (DTP) Tool

Usage: python3 dtp.py <image> <tissue\_type> <tile\_size>

Requires: python 3.8+, numpy, tensorflow, opencv-python, scikit-image,
          imagecodecs

Authors: Colin Greeley and Larry Holder, Washington State University

DTP identifies pathology in a given tissue image. The arguments to the script
are as follows:
* <image> is the image to be processed. See "Image Preprocessing" below.
* <model> is the model to be used to classify the image.
* <tile\_size> is the tile size used to train the model and the size used
  to tile the input image. This parameter is optional. The default is 256.

DTP first tiles the image according to the <tile\_size> into non-overlapping
tiles and then classifies each tile as diseased or not. DTP outputs two files.
One file <image>\_<tile\_size>\_tile.tif is the original image with diseased tiles
highlighted. The highlighted tif image can be very large, so the DTP does not
generate it by default (see gHighlightImage in dtp.py). You can also use the
highlight.py program to generate the image. Also, see "Image Postprocessing"
for compressing the large highlighted tif image. The second file
<image>\_<tile\_size>\_tiles.csv is a list of the diseased tiles' bounding boxes
using coordinates from the original image. Each line contains x,y,w,h where x,y
is the upper-left corner and w,h is the width and height of the tile.

Image Preprocessing:

If the image to be processed is in NDPI format, then you need to extract the
first image from the file. One option is to use the ndpi2tiff tool available
from https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/. NDPI
images may contain multiple versions of the image with different sizes. The
first image is usally the largest. To extract just the first image, use the
",0" suffix. So, the command looks like: ndpi2tiff <image>,0

Image Postprocessing:

The highlighted tif image can be very large. The vips program can reduce the
size to some extent, and make the image easier to load into some bio-image
tools, by converting to a pyramid format. One way to do this with vips is shown
below.

  vips tiffsave highlighted\_image.tif highlighted\_image\_pyramid
    --tile --pyramid --compression deflate --tile-width 256 --tile-height 256

Files included:

dtp.py: DTP tool

highlight.py: Generates image with DTP tiles highlighted.

