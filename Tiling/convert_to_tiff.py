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


src_dir = sys.argv[1] # example: "/home/data/epi/MG/"
dst_dir = sys.argv[2] # example: "./deployment/MG/"


shutil.copytree(src_dir, dst_dir)

def DFS_image_converter(path):
    dir_list = os.listdir(path)
    for dir in dir_list:
        if os.path.isfile(os.path.join(path, dir)):
            ndpi_image = dir
            dst_image_path = os.path.splitext(os.path.join(path, ndpi_image))[0] + ".tif"
            bashCommand = ["ndpi2tiff", "{},0".format(ndpi_image)]
            process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait()
            shutil.move("{},0.tif".format(ndpi_image), dst_image_path)
            os.remove(ndpi_image)
        else:
            DFS_image_converter(os.path.join(path, dir))
            
DFS_image_converter(dst_dir)
