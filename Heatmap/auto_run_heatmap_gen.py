# Run AFTER using convert_to_tiff.py
#
# Usage: python3 auto_run_heatmap_gen.py <src_dir>
#
#
# Runs the generate_heatmap.py on all images in the folder tree createcreated by convert_tiff.py
# - <src_dir> source directory, path to the directory that contains the Control folder
#
# Authors: Colin Greeley and Larry Holder, Washington State University


import os
import sys
import subprocess
import shutil


src_dir = sys.argv[1] 
tissue_type = sys.argv[2] 

data_map = {"testis": {"colors": "./colors/testis/colors.json", "pathologies": ["Ignore", "Atrophy", "Maturation_arrest", "Vacuole"], "tissue_type": "testis"},
            "prostate": {"colors": "./colors/prostate/colors.json", "pathologies": ["Ignore", "Atrophy", "Collapsed_Prost", "Hyperplasia_Prost", "Vacuole"], "tissue_type": "prostate"},
            "female kidney": {"colors": "./colors/kidney/female/colors.json", "pathologies": ["Ignore", "Cyst", "Reduced_Glomeruli", "Thickened_Bowmans_Capsule"], "tissue_type": "kidney/female"},
            "male kidney": {"colors": "./colors/kidney/male/colors.json", "pathologies": ["Ignore", "Cyst", "Reduced_Glomeruli", "Thickened_Bowmans_Capsule"], "tissue_type": "kidney/male"},
            "ovary": {"colors": "./colors/ovary/colors.json", "pathologies": ["Ignore", "Antral_follicle", "Antral_follicle_wo_oocyte_ov", "Large_cyst_ov", "Preantral_follicle_ov", "Primordial_follicle", "Small_cyst_ov"], "tissue_type": "ovary"}
            # any other pathologies can be added to this data map
            }

def DFS_generate_heatmap(path):
    dir_list = os.listdir(path)
    for dir in dir_list:
        if os.path.isfile(os.path.join(path, dir)):
            bashCommand = ["python3", "generate_heatmap.py", 
                                   "--image", os.path.join(path, dir), 
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
            os.remove(os.path.join(path, dir))
        else:
            DFS_generate_heatmap(os.path.join(path, dir))
            
DFS_generate_heatmap(src_dir)
