# Deep Histology Classification (DTP) Tool

Easy Usage: `bash run_dtp.sh <image> <tissue_type>` (only for known tissue types and pathologies)\
Advanced Usage: `python3 dtp.py <image> <tissue_type> <pathology> <gTileSize> [<downscale>]`\
Advanced Usage: `python3 dtp_eval.py <image> <tissue_type> <pathology> <gTileSize> <annotations> [<downscale>]`

Requires: python 3.8+, numpy, tensorflow, opencv-python, scikit-image, imagecodecs\
Recommendation: running tensorflow-gpu with anaconda for faster runtime

DTP identifies pathology in a given tissue image. The arguments to the script
are as follows:
* `<image>` path to the image to be processed. See "Image preprocessing" below.
* `<tissue_type>` testis, prostate, or kidney.
* `<pathology>` individual pathology that will be predicted by the classifier.
  For example, if your tissue type is testis, the optional pathology arguments
  would be "atrophy", "maturation_arrest", or "vacuole". Automatic pathology
  assignment is given in the run.sh and run.bat files.
* `<gTileSize>` is the tile size used to train the model and the size used
   to tile the input image. The default is 256.
* `<annotations>` path the and ndpi.annotations.json file for the respective tif
   image. This is used to overlay the "true" tiles on the image given from the
   annotations.
* `[<downscale>]` optional variable for the factor in which the output image is
  downsampled. Since the output images are very large, this optional variable is
  usually necessary. This value can be any power of 2 greater than zero. The default
  value is 4, meaning that the output heatmap image will be 0.25 times the resolution
  of the original input tif image.

DTP first tiles the image according to the `<tile_size>` into non-overlapping
tiles and then classifies each tile as diseased or not. DTP outputs one image file,
`<image>_<tile_size>\_tile.tif` is the original image with diseased tiles
highlighted.

## Run program

To automate the entire end-to-end process of using the DTP tool and highlighting
the image, you can use the "run_dtp" script (see "files included" below). All that is
required as arguments to run the program is an image and tissue type. The
rest is automatically taken care of. The output images will be written to the
same directory as the input image.

    bash run_dtp.sh ".\qupath\testis\tif\[#017] R18-964-17.tif" testis

    or

    python3 dtp.py "..\qupath\testis\tif\[#017] R18-964-17.tif" testis Atrophy 256

    or

    python3 dtp_eval.py "..\qupath\testis\tif\[#017] R18-964-17.tif" testis Atrophy 256 "..\qupath\testis\[#017] R18-964-17.ndpi.annotations.json"

## Image Preprocessing

If the image to be processed is in NDPI format, then you need to extract the
first image from the file. One option is to use the ndpi2tiff tool available
from <https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/>. NDPI
images may contain multiple versions of the image with different sizes. The
first image is usally the largest. To extract just the first image, use the
",0" suffix. So, the command looks like: `ndpi2tiff <image>,0`

    vips tiffsave highlighted_image.tif highlighted_image_pyramid --tile --pyramid --compression deflate --tile-width 256 --tile-height 256

## Files included

* run_dtp.sh: automatically run DTP and highlighting tool for all pathologies
  given tissue type and image.
* dtp.py: DTP tool. Imposes a heatmap over an input image showing
  classifier predictions.
* dtp_eval.py: DTP evaluation tool. Imposes a heatmap over an input image showing
  classifier predictions as well as drawing boxes around the "true" tiles that
  overlap the given annotations.

## Authors

Colin Greeley and Larry Holder, Washington State University

