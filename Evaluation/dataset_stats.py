# dataset_stats
#
# Usage: python3 dataset_stats.py [src_dir]
#
# - <image> path to the dataset to be processed.
#
# Authors: Colin Greeley and Larry Holder, Washington State University


import os
import sys
import numpy as np
import csv


def get_data_map():
    data_map = {"testis": {"name": [], "atrophy": {}, "vacuole": {}, "normal_tissue": {}},
                "prostate": {"name": [], "atrophy": {}, "collapsed_prost": {}, "hyperplasia_prost": {}, "vacuole": {}, "normal_tissue": {}},
                "female kidney": {"name": [], "cyst": {}, "reduced_glomeruli": {}, "thickened_bowmans_capsule": {}, "normal_tissue": {}},
                "male kidney": {"name": [], "cyst": {}, "reduced_glomeruli": {}, "thickened_bowmans_capsule": {}, "normal_tissue": {}},
                "ovary": {"name": [], "antral_follicle": {}, "large_cyst_ov": {}, "preantral_follicle_ov": {}, "primordial_follicle": {}, "small_cyst_ov": {}, "normal_tissue": {}}}
    for tissue in data_map.keys():
        for pathology in data_map[tissue].keys():
            if pathology != "name":
                #data_map[tissue][pathology] = {"95%_confidence_tile_count": [], "50%_confidence_tile_count": [], "instances": []}
                data_map[tissue][pathology] = {"95%_confidence_tile_count": [], "50%_confidence_tile_count": [], "95%_confidence_tile_ratio": [], "50%_confidence_tile_ratio": [], "instances": []}
    return data_map

def DFS_generate_heatmap(path, data_map, tissue_type=None):
    #print(path, tissue_type)
    dir_list = os.listdir(path)
    for dir in dir_list:
        if os.path.isfile(os.path.join(path, dir)):
            if os.path.splitext(os.path.join(path, dir))[-1] == ".txt":
                name = os.path.split(os.path.join(path, dir))[-1].split("_results")[0]
                data_map[tissue_type]["name"].append(name)
                with open(os.path.join(path, dir)) as f:
                    lines = f.readlines()
                t = 0
                for line in lines:
                    if t == 0:
                        pathology = line.split(" ")[1].lower()
                        pathology = "normal_tissue" if pathology == "normal_structure" else pathology
                        data_map[tissue_type][pathology]["95%_confidence_tile_count"].append(int(line.split(": ")[-1].split('\n')[0]))
                    if t == 1:
                        data_map[tissue_type][pathology]["50%_confidence_tile_count"].append(int(line.split(": ")[-1].split('\n')[0]))
                    if t == 2:
                        data_map[tissue_type][pathology]["instances"].append(int(line.split(": ")[-1].split('\n')[0]))
                    if t == 3:
                        t = -1
                    t += 1

        elif os.path.isdir(os.path.join(path, dir)):
            if dir.lower() in data_map.keys():
                tissue_type = dir.lower()
            DFS_generate_heatmap(os.path.join(path, dir), data_map, tissue_type)
            
def calculate_ratios(data_map):
    for tissue_type in data_map.keys():
        for pathology in data_map[tissue_type].keys():
            if not (pathology == "normal_structure" or pathology == "normal_tissue" or pathology == "name"):
                for metric in data_map[tissue_type][pathology].keys():
                    if "ratio" in metric and "95" in metric:
                        data_map[tissue_type][pathology][metric] = [x/y for x,y in zip(data_map[tissue_type][pathology]["95%_confidence_tile_count"], np.sum([data_map[tissue_type][pt]["95%_confidence_tile_count"] for pt in data_map[tissue_type].keys() if pt != "name"], axis=0))]
                    if "ratio" in metric and "50" in metric:
                        data_map[tissue_type][pathology][metric] = [x/y for x,y in zip(data_map[tissue_type][pathology]["50%_confidence_tile_count"], np.sum([data_map[tissue_type][pt]["50%_confidence_tile_count"] for pt in data_map[tissue_type].keys() if pt != "name"], axis=0))]
                    
def write_tissue_stats(src_dir, control_data_map, disease_data_map):
    metric = "95%_confidence_tile_count"
    for tissue_type in disease_data_map.keys():
        out_dir = os.path.join(src_dir, os.path.split(src_dir)[-1] + '_' + tissue_type + '_experimental_group_stats.csv')
        row_list = [["image_name"] + list(disease_data_map[tissue_type].keys())[1:-1]]
        for i in range(len(disease_data_map[tissue_type]["name"])):
            new_row = [disease_data_map[tissue_type]["name"][i]] + [round((disease_data_map[tissue_type][pa][metric][i] - np.mean(control_data_map[tissue_type][pa][metric],axis=0))/np.std(control_data_map[tissue_type][pa][metric],axis=0), 2) for pa in list(disease_data_map[tissue_type].keys())[1:-1]]
            row_list.append(new_row)
        #out_dir = os.path.join(src_dir, os.path.split(src_dir)[-1] + '_' + tissue_type + '_stats.csv')
        with open(out_dir, 'w', newline='') as file:
            writer = csv.writer(file, dialect='excel')
            writer.writerows(row_list)
            
            
        out_dir = os.path.join(src_dir, os.path.split(src_dir)[-1] + '_' + tissue_type + '_control_group_stats.csv')
        row_list = [["image_name"] + list(control_data_map[tissue_type].keys())[1:-1]]
        for i in range(len(control_data_map[tissue_type]["name"])):
            new_row = [control_data_map[tissue_type]["name"][i]] + [round((control_data_map[tissue_type][pa][metric][i] - np.mean(control_data_map[tissue_type][pa][metric],axis=0))/np.std(control_data_map[tissue_type][pa][metric],axis=0), 2) for pa in list(control_data_map[tissue_type].keys())[1:-1]]
            row_list.append(new_row)
        #out_dir = os.path.join(src_dir, os.path.split(src_dir)[-1] + '_' + tissue_type + '_stats.csv')
        with open(out_dir, 'w', newline='') as file:
            writer = csv.writer(file, dialect='excel')
            writer.writerows(row_list)
            
def write_dataset_stats(src_dir, control_data_map, disease_data_map):
    out_dir = os.path.join(src_dir, os.path.split(src_dir)[-1] + '_stats.csv')
    row_list = [["tissue/class/metric", os.listdir(src_dir)[0] + " mean", os.listdir(src_dir)[1] + " mean", os.listdir(src_dir)[0] + " std. dev.", os.listdir(src_dir)[1] + " std. dev."]]
    for tissue_type in control_data_map.keys():
        for pathology in control_data_map[tissue_type].keys():
            if not (pathology == "normal_structure" or pathology == "normal_tissue" or pathology == "name"):
                for metric in control_data_map[tissue_type][pathology].keys():
                    new_row = [tissue_type + ' ' + pathology + ' ' + metric, 
                            np.mean(control_data_map[tissue_type][pathology][metric]),
                            np.mean(disease_data_map[tissue_type][pathology][metric]),
                            np.std(control_data_map[tissue_type][pathology][metric]),
                            np.std(disease_data_map[tissue_type][pathology][metric])]
                    row_list.append(new_row)
        row_list.append(["", "", "", "", ""])
    #print(os.path.split(src_dir))
    with open(out_dir, 'w', newline='') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerows(row_list)

def clear_files(src_dir):
    for file in os.listdir(src_dir):
        if "csv" in file:
            os.remove(os.path.join(src_dir, file))


if __name__ == "__main__":
    
    src_dir = sys.argv[1] # "../deployment/MG/"
    if src_dir[-1] == '\\' or src_dir[-1] == '/':
        src_dir = src_dir[:-1]
    clear_files(src_dir)
    
    control_dir = os.path.join(src_dir, os.listdir(src_dir)[0])
    disease_dir = os.path.join(src_dir, os.listdir(src_dir)[1])
    
    control_data_map = get_data_map()
    disease_data_map = get_data_map()
    DFS_generate_heatmap(control_dir, control_data_map)
    DFS_generate_heatmap(disease_dir, disease_data_map)
    calculate_ratios(control_data_map)
    calculate_ratios(disease_data_map)
    
    write_dataset_stats(src_dir, control_data_map, disease_data_map)
    write_tissue_stats(src_dir, control_data_map, disease_data_map)
    
    
        