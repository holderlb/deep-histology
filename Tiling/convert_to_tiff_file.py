# Convert ndpi image to tiff
#
# Usage: python convert_tiff.py <src_dir> <dst_dir>
#
#
# Copies folder tree and replaces all ndpi images in the tree with the correct tiff format for deep learning
# - <src_dir> source directory
# - <dst_dir> destination directory
#
# Authors: Colin Greeley and Larry Holder, Washington State University

import os
import sys
import shutil
import subprocess


src_dir = sys.argv[1] # /home/data/epi/MG/Control/Prostate/B3 (#0650) R16-262-13.ndpi
dst_dir = sys.argv[2] # "./deployment/MG/Control/Prostate"

shutil.copy(src_dir, dst_dir)

def DFS_image_converter(dst_dir):
    print(dst_dir)
    ndpi_image = dst_dir
    dst_image_path = os.path.splitext(ndpi_image)[0] + ".tif"
    bashCommand = ["ndpi2tiff", "{},0".format(ndpi_image)]
    process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    shutil.move("{},0.tif".format(ndpi_image), dst_image_path)
    os.remove(ndpi_image)
            
DFS_image_converter(os.path.join(dst_dir, src_dir.split('/')[-1]))