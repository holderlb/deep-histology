# Deep Histology

Research on deep learning for histology.

## Deep Histology Classification (DHC) Tool

Tool that tiles a tissue image and uses a pre-trained deep learning
model to classify the tiles. See `DHC` directory.

## Training

Tool that is used to train deep convolutional nerual networks on a tile dataset
generated from the Tiling tool. Trained models are then used as input for 
the DHC tool. See `Training` directory.

## Tiling

Tool that generates training tiles from images annotated using QuPath.
See `Tiling` directory.
