# Deep Histology Classification

Research on deep learning for histology. The full pipeline is used to decompose large tif histology slides into smaller images (tiles) that can be used for training a deep convolutional neural network which are then used to generate heatmaps of pathology hot spots on input images.

<p align="center">
  <img src="figures/histology_pipeline.png" width="550" title="Deep Histology Pipeline">
</p>

## Tiling

Tool that is used for preprocessing ndpi images into tiff images for entire directories and also  
generates training tiles from images annotated using QuPath.
See `Tiling` directory.

## Training

Tool that is used to train deep convolutional nerual networks on a tile dataset
generated from the Tiling tool. Trained models are then used as input for 
the Evaluation tool. See `Training` directory.

## Evaluation

Tool that tiles a tissue image and uses a pre-trained deep learning
model to classify the tiles as well as show statistics between control and experimental groups. See `Evaluation` directory.

### Heatmap sample of model predictions
<p align="center">
  <img src="figures/Heatmap.png" width="800" title="Heatmap Sample">
</p>

### Class specific sample of model predictions
<p align="center">
  <img src="figures/prostate_example.png" width="800" title="Heatmap Sample">
</p>

## Acknowledgements

This work was supported by the John Templeton Foundation grant number 61174.
