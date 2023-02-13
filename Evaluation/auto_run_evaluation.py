# Run AFTER using convert_to_tiff.py
#
# Usage: python3 auto_run_heatmap_gen.py <src_dir>
#
#
# Runs the evaluate.py on all images in the folder tree createcreated by convert_tiff.py
# - <src_dir> source directory, path to the directory that contains the Control folder
#
# Authors: Colin Greeley and Larry Holder, Washington State University


import os
import sys
import subprocess


src_dir = sys.argv[1] # "../deployment/MG/"
if len(sys.argv) > 2:
    tissue_type = sys.argv[2]
else:
    tissue_type = None

data_map = {"testis": {"colors": "../qupath/testis/tiles/colors.json", "pathologies": ["Ignore", "Atrophy", "Vacuole", "Normal_Tissue"], "tissue_type": "testis"},
            "prostate": {"colors": "../qupath/prostate/tiles/colors.json", "pathologies": ["Ignore", "Atrophy", "Collapsed_Prost", "Hyperplasia_Prost", "Vacuole", "Normal_Tissue"], "tissue_type": "prostate"},
            "female kidney": {"colors": "../qupath/kidney/female/tiles/colors.json", "pathologies": ["Ignore", "Cyst", "Reduced_Glomeruli", "Thickened_Bowmans_Capsule", "Normal_Tissue"], "tissue_type": "kidney/female"},
            "male kidney": {"colors": "../qupath/kidney/male/tiles/colors.json", "pathologies": ["Ignore", "Cyst", "Reduced_Glomeruli", "Thickened_Bowmans_Capsule", "Normal_Tissue"], "tissue_type": "kidney/male"},
            "ovary": {"colors": "../qupath/ovary/tiles/colors.json", "pathologies": ["Ignore", "Antral_follicle", "Large_cyst_ov", "Preantral_follicle_ov", "Primordial_follicle", "Small_cyst_ov", "Normal_Tissue"], "tissue_type": "ovary"}}


def DFS_generate_heatmap(path, tissue_type):
    dir_list = os.listdir(path)
    for dir in dir_list:
        if os.path.isfile(os.path.join(path, dir)) and os.path.splitext(dir)[1] == '.tif':
            src_image_path = os.path.join(path, dir)
            print(src_image_path, tissue_type)
            bashCommand = ["python3", "evaluate.py", 
                            "--image", src_image_path, 
                            "--pathology_colors", data_map[tissue_type.lower()]["colors"],
                            "--tissue_type", data_map[tissue_type.lower()]["tissue_type"],
                            "--pathologies", *data_map[tissue_type.lower()]["pathologies"],
                            "--output_dir", path]
            process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            while process.poll() is None:
                line = process.stdout.readline()
                if line != '':
                    print(line, end =" ")
            #if os.path.exists(os.path.split(src_image_path)[-1]) and os.path.isdir(os.path.split(src_image_path)[-1]):
            os.remove(src_image_path)
        elif os.path.isdir(os.path.join(path, dir)):
            if dir.lower() in data_map.keys():
                tissue_type = dir.lower()
            DFS_generate_heatmap(os.path.join(path, dir), tissue_type)
            
DFS_generate_heatmap(src_dir, tissue_type)
