#!/bin/bash

# Usage: bash run.sh <image> <tissue_type>
# System: Linux
#
# Script to automatically run each pathology classifier on an input image given the tissue type.
# Creates a highlighted image for each pathology corresponding to a tissue type.
# Note: Append the character "&" on each python line to run in parallel (requires lots of memory).
#
# arg1: image location
# arg2: tissue type     (testis, prostate, kidney)


if [ $2 == "testis" ]
then
python3 .\DTP\dtp.py $1 testis Atrophy 256 
python3 .\DTP\dtp.py $1 testis Maturation_arrest 256 
python3 .\DTP\dtp.py $1 testis Vacuole 256 
fi


if [ $2 == "prostate" ]
then
python3 .\DTP\dtp.py $1 prostate Atrophy 256 
python3 .\DTP\dtp.py $1 prostate Collapsed_Prost 256 
python3 .\DTP\dtp.py $1 prostate Hyperplasia 256 
python3 .\DTP\dtp.py $1 prostate VSacuole 256 
fi


if [ $2 == "kidney" ]
then
python3 .\DTP\dtp.py $1 kidney Cyst 256 
python3 .\DTP\dtp.py $1 kidney Reduced_Glomeruli 256 
python3 .\DTP\dtp.py $1 kidney "Thickened_Bowman's_Capsule" 256 
fi