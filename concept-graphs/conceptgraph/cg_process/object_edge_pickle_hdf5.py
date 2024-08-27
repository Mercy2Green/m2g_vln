import h5py
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import gzip
import pickle as pkl
import pandas as pd

def edges_pkls_to_hdf5(
    data_root = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R",
    obj_file_name="cfslam_scenegraph_nodes.pkl.gz",
    edges_file_name="cfslam_scenegraph_edges.pkl",
    data_edges_hdf5 = "/data0/vln_datasets/preprocessed_data/edges_hdf5"
    ):
    
    if not os.path.exists(data_edges_hdf5):
        os.mkdir(data_edges_hdf5)
    
    edges_dict = get_all_edges_from_file(
        data_root, 
        obj_file_name,
        edges_file_name, 
        scans_allocation_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/scans_allocation")
    
    # create a hdf5 dataset to store edges_dict in data_deges_hdf5
    # check if the edges.hdf5 is exist, if it is, delet it  
    if os.path.exists(os.path.join(data_edges_hdf5, "edges.hdf5")):
        os.remove(os.path.join(data_edges_hdf5, "edges.hdf5"))
        
    with h5py.File(os.path.join(data_edges_hdf5, "edges.hdf5"), 'w') as f:
        for scan in edges_dict.keys():
            scan_group = f.create_group(scan)
            # Assuming edges_dict[scan][0] is not mixed and doesn't cause issues
            objs = scan_group.create_dataset("objs", data=edges_dict[scan][0])
            
            # Splitting the mixed list into integers and strings
            edges_integers = np.array([item[:2] for item in edges_dict[scan][1]], dtype=np.int32)
            edges_strings = np.array([item[2] for item in edges_dict[scan][1]], dtype=h5py.special_dtype(vlen=str))
            
            # Creating separate datasets for integers and strings
            edges_ints_ds = scan_group.create_dataset("edges_integers", data=edges_integers)
            edges_strs_ds = scan_group.create_dataset("edges_strings", data=edges_strings)
            
    if os.path.exists(os.path.join(data_edges_hdf5, "edges.hdf5")):
        print("The hdf5 dataset of edges is created successfully")


def get_edges_from_hdf5(hdf5_path="/data0/vln_datasets/preprocessed_data/edges_hdf5", hdf5_file_name="edges.hdf5"):
    edges_dict = {}
    hdf5_path = os.path.join(hdf5_path, hdf5_file_name)
    
    with h5py.File(hdf5_path, 'r') as f:
        for scan in f.keys():
            # Initialize an empty list to store combined edges for the current scan
            combined_edges = []
            
            # Load the 'objs' dataset as is
            objs = f[scan]['objs'][:]
            
            # Check if the 'edges_integers' and 'edges_strings' datasets exist
            if 'edges_integers' in f[scan] and 'edges_strings' in f[scan]:
                edges_integers = f[scan]['edges_integers'][:]
                edges_strings = f[scan]['edges_strings'][:]
                
                # Combine integers and strings back into the original mixed structure
                for i in range(len(edges_integers)):
                    combined_edge = list(edges_integers[i]) + [edges_strings[i].decode('utf-8')]
                    combined_edges.append(combined_edge)
            else:
                # Fallback if the original 'edges' dataset exists without splitting
                combined_edges = f[scan]['edges'][:]
            
            # Store the loaded data in the dictionary
            edges_dict[scan] = (objs, combined_edges)
    return edges_dict

def get_all_edges_from_file(
    data_root = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R", 
    obj_file_name="cfslam_scenegraph_nodes.pkl.gz",
    edges_file_name="cfslam_scenegraph_edges.pkl", 
    scans_allocation_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/scans_allocation"):

    with open(os.path.join(scans_allocation_dir, 'all_scans.txt')) as f:
    # with open(os.path.join(scans_allocation_dir, 'test_scan.txt')) as f:  
        scans = [x.strip() for x in f]    

    with open(os.path.join(scans_allocation_dir, 'eval_scans.txt')) as f:
        eval_scans = [x.strip() for x in f] 
    
    train_scans = list(set(scans) - set(eval_scans))
    
    dict_scan_edges = {}
    for scan in tqdm(train_scans):
        print("Loading scan: ", scan)
        with gzip.open(f"{data_root}/{scan}/sg_cache/{obj_file_name}", 'rb') as f:
            objs_gz = pkl.load(f)
        objs = [obj["bbox_np"] for obj in objs_gz['objects']]

        with open(f"{data_root}/{scan}/sg_cache/{edges_file_name}", 'rb') as f:
            edges = pkl.load(f)
        dict_scan_edges[scan] = (objs, edges)
    return dict_scan_edges

def main():
    
    edges_pkls_to_hdf5()
    edges_dict = get_edges_from_hdf5("/data0/vln_datasets/preprocessed_data/edges_hdf5")
    
    print("Test ok")

if __name__ == "__main__":
    main()