# Deep Tissue Pathology (DTP) Tool

Usage: `python3 dtp.py <image> <tissue_type> <pathology> <gTileSize> [<highlighting>]`

Requires: python 3.8+, numpy, tensorflow, opencv-python, scikit-image, imagecodecs
Recommendation: running tensorflow-gpu with anaconda for faster runtime

DTP identifies pathology in a given tissue image. The arguments to the script
are as follows:
* `<image>` path to the image to be processed.
* `<tissue_type>` testis, prostate, or kidney.
* `<pathology>` individual pathology that will be predicted by the classifier.
* `<gTileSize>` is the tile size used to train the model and the size used
   to tile the input image. This parameter is optional. The default is 256.
* `<highlighting>` optional argument for the highlighting type. Default is
  drawing boxes around diseased tiles.
  For heatmap options, type in any of the colormap names from the following link:
  <https://matplotlib.org/stable/tutorials/colors/colormaps.html>
  Recommendations: plasma, gray, cividis

DTP first tiles the image according to the `<tile_size>` into non-overlapping
tiles and then classifies each tile as diseased or not. DTP outputs two files.
One file `<image>_<tile_size>\_tile.tif` is the original image with diseased tiles
highlighted. The highlighted tif image can be very large, so the DTP does not
generate it by default (see `gHighlightImage` in dtp.py). You can also use the
highlight.py program to generate the image. Also, see "Image Postprocessing"
for compressing the large highlighted tif image. The second file
`<image>_<tile_size>_tiles.csv` is a list of the diseased tiles' bounding boxes
using coordinates from the original image. Each line contains x,y,w,h where x,y
is the upper-left corner and w,h is the width and height of the tile.

## Image Preprocessing

If the image to be processed is in NDPI format, then you need to extract the
first image from the file. One option is to use the ndpi2tiff tool available
from <https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/>. NDPI
images may contain multiple versions of the image with different sizes. The
first image is usally the largest. To extract just the first image, use the
",0" suffix. So, the command looks like: `ndpi2tiff <image>,0`

## Image Postprocessing

The highlighted tif image can be very large. The vips program can reduce the
size to some extent, and make the image easier to load into some bio-image
tools, by converting to a pyramid format. One way to do this with vips is shown
below.

    vips tiffsave highlighted_image.tif highlighted_image_pyramid --tile --pyramid --compression deflate --tile-width 256 --tile-height 256

## Files included

* dtp.py: DTP tool.
* highlight.py: Generates image with DTP tiles highlighted.
* run.sh: automatically run DTP and highlighting tool for all pathologies
  given tissue type and an image (Linux)
* run.bat: automatically run DTP and highlighting tool for all pathologies
  given tissue type and an image (Windows)

## Authors

Colin Greeley and Larry Holder, Washington State University

