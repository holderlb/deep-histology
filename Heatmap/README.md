# Heatmap Generation

Usage: `python3 generate_geatmap.py --image --pathology_colors --tissue_type --pathologies --[tile_size] --[downscale]`

Requires: python 3.8+, numpy, tensorflow, opencv-python, scikit-image, imagecodecs\
Recommendation: running tensorflow-gpu with anaconda for faster runtime

generate_geatmap identifies pathology locations in a given tissue image and highlights the 
predicted areas in the output heatmap iamge. The arguments to the script are as follows:
* `--image` Path to the image to be processed. See "Image preprocessing" below.
* `--pathology_colors` Path to colors file created from "generatetiles.py". Color information for tiles is stored here.
* `--tissue_type` Name of the tissue type. i.e., "breast" tissue
* `--pathologies` Names of the pathologies you want to classify. Same as pathologies used for training.
* `--tile_size` Optional. Is the tile size used to train the model and the size used
   to tile the input image. The default is 256.
* `--downscale` Optional. Integer ariable for the factor in which the output image is
  downsampled. Since the output images are very large, this optional variable is
  usually necessary. This value can be any power of 2 greater than zero. The default
  value is 4, meaning that the output heatmap image will be 0.25 times the resolution
  of the original input tif image.

The heatmap generator first tiles the image according to the `<tile_size>` into overlapping
tiles and then classifies each tile as diseased or not. DTP outputs one image file,
`<image>_<tile_size>\_tile.tif` is the original image with diseased tiles
highlighted.

## Run program

To automate the entire end-to-end process of using the heatmap generator tool, 
you can use the "run_heatmap_generator" script (see "files included" below). All that 
is required as arguments to run the program is an image and tissue type. The
rest is automatically taken care of. The output images will be written to the
same directory as the input image.

    python .\generate_heatmapV2.py --image "..\qupath\testis\tif\[#017] R18-964-17.tif" --pathology_colors ..\qupath\testis\tiles\colors.json --tissue_type testis --pathologies Ignore Atrophy Maturation_arrest Vacuole


## Image Preprocessing

If the image to be processed is in NDPI format, then you need to extract the
first image from the file. One option is to use the ndpi2tiff tool available
from <https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/>. NDPI
images may contain multiple versions of the image with different sizes. The
first image is usally the largest. To extract just the first image, use the
",0" suffix. So, the command looks like: `ndpi2tiff <image>,0`

    vips tiffsave highlighted_image.tif highlighted_image_pyramid --tile --pyramid --compression deflate --tile-width 256 --tile-height 256

## Files included

* generate_heatmap.py: Heatmap creation tool. Imposes a heatmap over an input image showing
  classifier predictions.

## Authors

Colin Greeley and Larry Holder, Washington State University

