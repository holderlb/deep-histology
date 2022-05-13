#!/bin/bash

# Usage: bash run.sh <image> <tissue_type>
# System: Linux/MacOS
#
# Script to automatically run all pathology classifiers for a known tissue type 
# with pretrained models on an input image given.
# Creates a highlighted image for each pathology corresponding to a tissue type.
# This file can be added to by including your own additional tissue types with
# corresponding pathologies.
#
# arg1: image location
# arg2: tissue type     (testis, prostate, kidney)
#
# Authors: Colin Greeley and Larry Holder, Washington State University


if [ $2 == "testis" ]
then
python3 dtp.py $1 testis Atrophy 256 
python3 dtp.py $1 testis Maturation_arrest 256 
python3 dtp.py $1 testis Vacuole 256 
fi


if [ $2 == "prostate" ]
then
python3 dtp.py $1 prostate Atrophy 256 
python3 dtp.py $1 prostate Collapsed_Prost 256 
python3 dtp.py $1 prostate Hyperplasia 256 
python3 dtp.py $1 prostate VSacuole 256 
fi


if [ $2 == "kidney-female" ]
then
python3 python dtp_eval.py $1 kidney-female Cyst 256 
python3 python dtp_eval.py $1 kidney-female Reduced_Glomeruli 256 
python3 python dtp_eval.py $1 kidney-female Thickened_Bowmans_Capsule 256 
fi

if [ $2 == "kidney-male" ]
then
python3 python dtp_eval.py $1 kidney-male Cyst 256 
python3 python dtp_eval.py $1 kidney-male Reduced_Glomeruli 256 
python3 python dtp_eval.py $1 kidney-male Thickened_Bowmans_Capsule 256 
fi

if [ $2 == "ovary" ]
then
python3 python dtp_eval.py $1 ovary Antral_follicle_wo_oocyte_ov 256 
python3 python dtp_eval.py $1 ovary Antral_follicle 256 
python3 python dtp_eval.py $1 ovary Large_cyst_ov 256 
python3 python dtp_eval.py $1 ovary Preantral_follicle_ov 256 
python3 python dtp_eval.py $1 ovary Primordial_follicle 256 
python3 python dtp_eval.py $1 ovary Small_cyst_ov 256 
fi
