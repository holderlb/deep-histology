# Deep Histology

Research on deep learning for histology.

## Deep Tissue Pathology (DTP) Tool

Tool that tiles a tissue image and uses a pre-trained deep learning
model to classify the tiles. See `DTP` directory.

## Training

Tool that is used to train deep convolutional nerual networks on a tile dataset
generated from the Tiling tool. Trained models are then used as input for 
the DTP tool. See `Training` directory.

## Tiling

Tool that generates training tiles from images annotated using QuPath.
See `Tiling` directory.
