# Model Training

Usage: `python3 train.py --tile_dir --tissue_type --pathologies --[tile_size] --[batch_size] --[ensemble_size]`\
Usage: `python3 test.py --tile_dir --tissue_type --pathologies --[tile_size] --[batch_size] --[ensemble_size]`

Requires: python 3.8+, numpy, tensorflow 2.6+, opencv-python, scikit-image, imagecodecs, matplotlib\
Recommendation: running tensorflow-gpu with anaconda for faster runtime

Used to train tile classifying models (train.py) or evaluate their performance (test.py). 
The arguments to the script are described as follows:
* `<tile_dir>` Path to directory of tiles for training
* `<tissue_type>` help='Name of the tissue type. i.e., "breast" tissue
* `<pathology>` Name of the pathology you want to classify. It will be the positive class for 
  new binary classification model. Every other class will be treated as the negative class
* `[<tile_size>]` (Optional) Resolution of tiles used for neural network input. (Default: 256)
* `[<batch_size>]` (Optional) Batch size for training neural networks, preferably devisible by ensemble size. (Default: 90)
* `[<ensemble_size>]` (Optional) Number of classifiers in ensemble, should be an odd number. (Default: 9)

The output of train program is a trained model in the "models" directory with the following path
"./models/`<tissue_type>`/`<pathology><tile_size>`.h5".

The output of the test program is the evaluation results for the out-of-the-bag samples used
for the bootstrapped classifier with ensembe_size=9

## Examples

    python3 train.py --tile_dir data/qupath/testis/tiles/ --tissue_type testis --pathologies Ignore Atrophy Maturation_arrest Vacuole
    
    python3 test.py --tile_dir data/qupath/testis/tiles/ --tissue_type testis --pathologies Ignore Atrophy Maturation_arrest Vacuole
    
Note: the `pathology` argument does not actually have to be a pathology. It is just the positive 
class for classification. It can be any class that you want to classify.

## Authors

Colin Greeley and Larry Holder, Washington State University
