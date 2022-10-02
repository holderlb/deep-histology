# Heatmap Generation

Usage: `python3 evaluate.py --image --pathology_colors --tissue_type --pathologies --[tile_size] --[downscale] --[generate_heatmap] --[output_dir]`

Requires: python 3.8+, numpy, tensorflow, opencv-python, scikit-image, imagecodecs\
Recommendation: running tensorflow-gpu with anaconda for faster runtime

evaluate.py identifies pathology locations in a given tissue image and outputs information about the neural network predictions
as well as predictions in the form of qupath annotations and a heatmap image. The arguments to the script are as follows:
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
* `--generetate_heatmap` Optional. Boolean variable that determines whether a heatmap image will be created
   and saved for each input image. (Resource intensive).
* `--output_dir` Optional. File path to specify where all program outputs will go to

## Run program

To run the evaluation for a singe image, the `evaluate.py` program needs to be run the arguments seen above. An example of this is shown below:

    python3 .\evaluate.py --image "..\qupath\testis\tif\[#017] R18-964-17.tif" --pathology_colors ..\qupath\testis\tiles\colors.json --tissue_type testis --pathologies Ignore Atrophy Maturation_arrest Vacuole

To automatically run the evaluation for a directory of tif images, you can use the `auto_run_evaluations.py` program. This program simply requires a source directory as program arguments. If the tissue type(s) are not apart of the file path to any tif file, the tissue type(s) can be specified through the second program argument. All files in the source directory folder tree must be tif files. The folder tree can be arbitrarily wide or deep.

    python3 auto_run_evaluation.py <src> [<tissue_type>]

## Image Preprocessing

See `convert_to_tiff.py` in the Tiling directory of this GitHub repository.

## Files included

* evaluate.py: Evaluation statistics tool. Outputs statistics in a text file,
overlay annotations in a json file, tile information
in a csv file, and an optional heatmap overlap saved in a jpg file.
* auto_run_evaluation.py: Used to automatically run evaluate.py on a directory 
tree on tif images. Make sure tissue data is in the "data_map"
dictionary inside the auto_run_evaluation.py program.
* upload_annotations.groovy: Used to upload output annotations from evaluate.py 
into qupath to see neural network predictions as annotations.


## Authors

Colin Greeley and Larry Holder, Washington State University


