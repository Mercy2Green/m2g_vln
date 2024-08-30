import os
import random
import numpy as np
import h5py
import json
from os import path
from numpy import save
from requests import get
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, List

import gzip, pickle

import argparse
from pathlib import Path
import re
from PIL import Image
import cv2

import open_clip

import distinctipy

import torch
import torchvision
import supervision as sv

from conceptgraph.dataset.datasets_common import GradSLAMDataset, as_intrinsics_matrix

from conceptgraph.utils.model_utils import compute_clip_features
import torch.nn.functional as F

from gradslam.datasets import datautils
from conceptgraph.slam.utils import gobs_to_detection_list
from conceptgraph.slam.cfslam_pipeline_batch import BG_CLASSES

# Local application/library specific imports
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import OnlineObjectRenderer, vis_result_fast, vis_result_slow_caption
from conceptgraph.utils.ious import (
    compute_2d_box_contained_batch
)
from conceptgraph.utils.general_utils import to_tensor

from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
from conceptgraph.slam.utils import (
    create_or_load_colors,
    merge_obj2_into_obj1, 
    denoise_objects,
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
    get_classes_colors
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects
)

import gc
import open3d

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from habitat.core.vector_env import VectorEnv

from conceptgraph.dataset.datasets_common import load_dataset_config

try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

# Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    
import sys
if "TAG2TEXT_PATH" in os.environ:
    TAG2TEXT_PATH = os.environ["TAG2TEXT_PATH"]
    
EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
sys.path.append(EFFICIENTSAM_PATH)

import torchvision.transforms as TS
try:
    from ram.models import ram
    from ram.models import tag2text
    from ram import inference_tag2text, inference_ram
except ImportError as e:
    print("RAM sub-package not found. Please check your GSA_PATH. ")
    raise e

# Disable torch gradient computation
# torch.set_grad_enabled(False)
# Don't set it in global, just set it in the function that needs it.
# Using with torch.set_grad_enabled(False): is better.
# Or using with torch.no_grad(): is also good.
    
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

FOREGROUND_GENERIC_CLASSES = [
    "item", "furniture", "object", "electronics", "wall decoration", "door"
]

FOREGROUND_MINIMAL_CLASSES = [
    "item"
]



class ObjEdgeProcessor():
    def __init__(
        self,
        objs_hdf5_save_dir="/data0/vln_datasets/preprocessed_data/finetune_cg_hdf5",
        objs_hdf5_save_file_name="finetune_cg_data.hdf5",
        edges_hdf5_save_dir="/data0/vln_datasets/preprocessed_data/edges_hdf5",
        edges_hdf5_save_file_name="edges.hdf5",
        connectivity_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity",
        connectivity_file_name="scans.txt",
        exclude_scans_file_name="eval_scans.txt",
        obj_feature_name="clip",
        obj_pose_name="bbox_np",
        allobjs_list=[],
        alledges_dict={},
        allvps_pos_dict={}
        ):
        
        # vps
        
        self.connectivity_dir = connectivity_dir
        self.connectivity_file_name = connectivity_file_name
        self.exclude_scans_file_name = exclude_scans_file_name
        
        self.allvps_pos_dict = allvps_pos_dict # a dict of dict, scan dict.  usage: allvps[scan]
        
        # objs
        self.objs_hdf5_save_dir = objs_hdf5_save_dir
        self.objs_hdf5_save_file_name = objs_hdf5_save_file_name
        self.obj_feature_name = obj_feature_name
        self.obj_pose_name = obj_pose_name
        
        self.allobjs_list_list = allobjs_list # a dict of dict, scan dict.  usage: allobjs[scan]
        
        # edges
        self.edges_hdf5_save_dir = edges_hdf5_save_dir
        self.edges_hdf5_save_file_name = edges_hdf5_save_file_name
        
        self.alledges_dict = alledges_dict
        
        
    def get_merge_feature_edges_relationship(self, scan, path):
        
        if self.allobjs_list == []:
            self.load_allobjs_from_hdf5()
            Warning("Loading objs from hdf5 file can be very slow! Shouldn't be used in the mergeing process.")
        all_objects = self.allobjs_list[scan]

        filtered_objs, filtered_objs_feature, filtered_objs_pose = self.merge_feature(all_objects, path, self.obj_feature_name, self.obj_pose_name)
        edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend = self.get_obj_edges_finetune(scan, filtered_objs_pose, filtered_objs, self.alledges_dict)
        
        # return filtered_objs, objs_feature, objs_pose, edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend
        return filtered_objs, filtered_objs_feature, filtered_objs_pose, edges_objects_extend, edges_relationship_extend
    
    def process_obj(self, all_objs, path):
        
        # The all_objs is a list of objs, each obj is a dict, the key is the feature name, the value is the feature value.
        # The all_edges is a list of edges, each edge is a list of two obj index and the relationship.
        # The path is a list of vp names, the function return the objs and edges in the path.
        bbox_center_str = "bbox_center_np"
        feature_str = "clip"
        
        for obj in all_objs:
                obj["bbox_center_np"] = obj["bbox"].center

        
        # Merge the feature and pose of the objs in the path
        filtered_objs, filtered_objs_feature, filtered_objs_pose = self.merge_feature(
            all_objs,
            path,
            feature_str,
            bbox_center_str)
        
        return filtered_objs, filtered_objs_feature, filtered_objs_pose
        # return _objs, _objs_feature, _objs_pose, _edges_objects, _edges_relationship 
    
    def process_edge(self, filtered_objs, all_objs, all_edges):
        
        # The objs is a list of objs, each obj is a dict, the key is the feature name, the value is the feature value.
        # The all_edges is a list of edges, each edge is a list of two obj index and the relationship.
        # The function return the edges in the path.
        
        edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend = self.match_obj_edge(filtered_objs, all_objs, all_edges)
        
        
        return edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend
    
    def load_obj_edge_vp(self):
        self.load_allobjs_from_hdf5()
        self.get_edges_from_hdf5()
        self.load_allvps_pos_from_connectivity_json()
        return self.allobjs_list, self.alledges_dict, self.allvps_pos_dict
    
    def match_obj_edge(self, filtered_objs, all_objs, all_edges):
        
        bbox_center_str = "bbox_center_np"
        
        correspondence_dict = self.find_correspondence_similarity_center2center(all_objs, filtered_objs, bbox_center_name=bbox_center_str)
    
        edges = []  
        for relationship in all_edges:
            edge = []
            if relationship[0] in correspondence_dict and relationship[1] in correspondence_dict and relationship[2] != "none of these":
                edge.append(correspondence_dict[relationship[0]])
                edge.append(correspondence_dict[relationship[1]])
                edge.append(relationship[2])
                edges.append(edge)
            else:
                continue
            
        # edges_objs is edges first two columns
        edges_objects = [edge[:2] for edge in edges]
        # edges_relationship is the third column
        edges_relationship = [self.edge_encode(edge[2], False) for edge in edges]
        
        # exntend
        _edges_objects_extend = [[edge[1], edge[0]] for edge in edges]
        _edges_relationship_extend =  [self.edge_encode(edge[2], True) for edge in edges]
        
        edges_objects_extend = []
        edges_relationship_extend = []
        edges_objects_extend.extend(edges_objects)
        edges_objects_extend.extend(_edges_objects_extend)
        edges_relationship_extend.extend(edges_relationship)
        edges_relationship_extend.extend(_edges_relationship_extend)
        
        # return edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend
        return np.array(edges_objects), np.array(edges_relationship), np.array(edges_objects_extend), np.array(edges_relationship_extend)
    
    def merge_feature(self, all_objects, path, valname_feature="clip", valname_pose="bbox_np"):

        filtered_objs = []
        objs_feature = []
        objs_pose = []
        
        for obj in all_objects:
            filterd_obj = {}
            vp_in_path = []
            obj_observation = obj.get("observed_info")
            obj_observation_vps = list(obj_observation.keys())
            for vp in obj_observation_vps:
                if vp in path:
                    vp_in_path.append(vp)
                    #if filterd_obj[valname_feature] is None: ######!!!! This is a wrong way to judge the key is exist or not.
                    if valname_feature not in filterd_obj: # This is the right way.
                        filterd_obj[valname_feature] = obj_observation[vp]
                    else:
                        filterd_obj[valname_feature] += obj_observation[vp]
                    
            if filterd_obj != {}:
                filterd_obj[valname_feature] = filterd_obj[valname_feature] / len(vp_in_path)
                filterd_obj[valname_pose] = obj[valname_pose]
                filtered_objs.append(filterd_obj)
                
                # obj_feature_tensor = torch.tensor(filterd_obj[valname_feature])
                # obj_feature_normalize = F.normalize(obj_feature_tensor, p=2, dim=0)
                # objs_feature.append(obj_feature_normalize.numpy())
                objs_feature.append(filterd_obj[valname_feature]) # The feature is already normalized.
                
                objs_pose.append(filterd_obj[valname_pose])
        
        # obj_clip_fts = F.normalize(obj_clip_fts, p=2, dim=-1)
        
        return filtered_objs, objs_feature, objs_pose
        
    def get_obj_edges_finetune(self, scan, objs_pose, objs, edges_dict):
        edges_objects = None
        edges_relationship = None
        edges_relationship_extend = None
        
        _, edges_relationship = self.get_edges_from_dict(edges_dict, scan)
        
        correspondence_dict = self.find_correspondece_similarity_bboxcenter(objs, objs_pose)
        # Correspondec_dict is a dict, the key is the index of the objs, and the value is the index of obj_pos.
        
        edges = []  
        for relationship in edges_relationship:
            edge = []
            if relationship[0] in correspondence_dict and relationship[1] in correspondence_dict and relationship[2] != "none of these":
                edge.append(correspondence_dict[relationship[0]])
                edge.append(correspondence_dict[relationship[1]])
                edge.append(relationship[2])
                edges.append(edge)
            else:
                continue
            
        # edges_objs is edges first two columns
        edges_objects = [edge[:2] for edge in edges]
        # edges_relationship is the third column
        edges_relationship = [self.edge_encode(edge[2], False) for edge in edges]
        
        # exntend
        _edges_objects_extend = [[edge[1], edge[0]] for edge in edges]
        _edges_relationship_extend =  [self.edge_encode(edge[2], True) for edge in edges]
        
        edges_objects_extend = []
        edges_relationship_extend = []
        edges_objects_extend.extend(edges_objects)
        edges_objects_extend.extend(_edges_objects_extend)
        edges_relationship_extend.extend(edges_relationship)
        edges_relationship_extend.extend(_edges_relationship_extend)
        
        # return edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend
        return np.array(edges_objects), np.array(edges_relationship), np.array(edges_objects_extend), np.array(edges_relationship_extend)
        
    def find_similar_vp(self, scan, cur_pos, similarity_threshold=0.2, decimals=5):
        import numpy as np
        """Find the most similar viewpoint based on the similarity of vp position."""
        
        vps_pos_dict = self.allvps_pos_dict[scan]
        # Key is the vp name, value is the vp position
        
        distance_list = []
        min_distance = None
        
        # Iterate over each center in allvps
        for vp_name, vp_pos in vps_pos_dict.items():
            # Calculate Euclidean distance to determine similarity.
            distance = np.linalg.norm(np.around(cur_pos, decimals=decimals) - np.around(vp_pos, decimals=decimals))
            distance_list.append(distance)
            if min_distance is None or distance <= min_distance:
                if distance == min_distance:
                    Warning("There are two vp have the same distance to the current vp. Should deal with this situation.")
                min_distance = distance
                closest_vp = vp_name
        
        if min_distance > similarity_threshold:
            Warning("Compromising. The most similar vp is not similar enough. The distance is larger than the threshold.")

        return closest_vp
    
    def load_allobjs_from_hdf5(self):
        hdf5_filename = os.path.join(self.objs_hdf5_save_dir, self.objs_hdf5_save_file_name)
        save_objs = {}
        with h5py.File(hdf5_filename, 'r') as hdf5_file:
            for scan in hdf5_file:
                scan_group = hdf5_file[scan]
                save_objs[scan] = []
                for obj_idx in scan_group:
                    obj_group = scan_group[obj_idx]
                    observed_info_group = obj_group["observed_info"]
                    observed_info = {}
                    for key in observed_info_group:
                        np_array = observed_info_group[key][()]
                        observed_info[key] = np_array  # Assuming direct usage of numpy arrays; convert as needed
                    bbox_np = obj_group["bbox_np"][()]
                    obj_data = {"observed_info": observed_info, "bbox_np": bbox_np}
                    save_objs[scan].append(obj_data)
        self.allobjs_list = save_objs
        return save_objs 
    
    def get_edges_from_hdf5(self):
        import h5py
        
        hdf5_path = self.edges_hdf5_save_dir
        hdf5_file_name = self.edges_hdf5_save_file_name
        
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
        self.alledges_dict = edges_dict
        return edges_dict
    
    def load_allvps_pos_from_connectivity_json(self):
        
        allvps_pos_dict = {}    
        
        connectivity_dir = self.connectivity_dir
        connectivity_file_name = self.connectivity_file_name
        exclude_scans_file_name = self.exclude_scans_file_name
        
        if exclude_scans_file_name is None:
            eval_scans = []
        else:
            with open(os.path.join(connectivity_dir, exclude_scans_file_name)) as f:
                eval_scans = [x.strip() for x in f] 
        
        with open(os.path.join(connectivity_dir, connectivity_file_name)) as f:
            scans = [x.strip() for x in f]      # load all scans
            
        filtered_scans = list(set(scans) - set(eval_scans))
        
        for scan in filtered_scans:
            with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
                data = json.load(f)
                allvps_pos_dict[scan] = {}
                for item in data:
                    if item['included']:
                        allvps_pos_dict[scan][item['image_id']] = np.array([item['pose'][3], item['pose'][7], item['pose'][11]])
                        # This is form the "def load_nav_graphs(connectivity_dir):" of the common.py 
        
        self.allvps_pos_dict = allvps_pos_dict
        
        return allvps_pos_dict
    
    def load_viewpoint_ids(self, connectivity_dir, connectivity_file_name, exclude_scans_file_name="eval_scans.txt"):
        viewpoint_ids = []
        
        if exclude_scans_file_name is None:
            eval_scans = []
        else:
            with open(os.path.join(connectivity_dir, exclude_scans_file_name)) as f:
                eval_scans = [x.strip() for x in f] 
        
        with open(os.path.join(connectivity_dir, connectivity_file_name)) as f:
            scans = [x.strip() for x in f]      # load all scans
            
        filtered_scans = list(set(scans) - set(eval_scans))
        
        for scan in filtered_scans:
            with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
                data = json.load(f)
                viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
        print('Loaded %d viewpoints' % len(viewpoint_ids))
        return viewpoint_ids
    
    def find_correspondence_similarity_center2center(self, all_objs, filtered_objs, similarity_threshold=0.2, decimals=5, bbox_center_name="bbox_np"):
        import numpy as np
        
        correspondence_dict = {}
        
        all_objs_centers = [obj[bbox_center_name] for obj in all_objs]
        filtered_objs_centers = [fobj[bbox_center_name] for fobj in filtered_objs]
        
        for i, pos_center in enumerate(filtered_objs_centers):
            for j, obj_center in enumerate(all_objs_centers):
                distance = np.linalg.norm(np.around(pos_center, decimals=decimals) - np.around(obj_center, decimals=decimals))
                if distance < similarity_threshold:
                    correspondence_dict[j] = i
                    break
                
        return correspondence_dict
        
    def find_correspondece_similarity_bboxcenter(self, objs, obj_pos, similarity_threshold=0.2, decimals=5, bbox_dict_name="bbox_np"):
        import numpy as np
        """Find correspondence based on the similarity of bbox centers."""
        correspondence_dict = {}
        
        # Calculate centers for each bbox in objs
        objs_centers = [self.calculate_center(obj[bbox_dict_name]) for obj in objs]
        
        # Iterate over each center in obj_pos
        for i, pos_center in enumerate(obj_pos):
            # Compare with each center in objs
            for j, obj_center in enumerate(objs_centers):
                # Calculate Euclidean distance to determine similarity
                distance = np.linalg.norm(np.around(pos_center, decimals=decimals) - np.around(obj_center, decimals=decimals))
                # If distance is within the threshold, add to correspondence_dict
                if distance < similarity_threshold:
                    correspondence_dict[j] = i
                    break  # Assuming one-to-one correspondence, stop after the first match
        return correspondence_dict
    
    def calculate_center(self, bbox_corners):
        """Calculate the center of a bbox given its eight corners."""
        if len(bbox_corners) == 3:
            bbox_center = bbox_corners
            return bbox_center
        
        return np.mean(bbox_corners, axis=0)
    
    def get_edges_from_dict(self, dict_scan_edges, scan):
        objs_bbox_list, edges_relationship = dict_scan_edges[scan]
        objs = [{"bbox_np": obj} for obj in objs_bbox_list]
        return objs, edges_relationship
    
    def edge_encode(self, edge_relationship, revert_flag = False):
    
        if revert_flag == False:
            if edge_relationship == "a on b":
                encoded_relationship = 0
            elif edge_relationship == "a in b":
                encoded_relationship = 1
            elif edge_relationship == "b on a":
                encoded_relationship = 2
            elif edge_relationship == "b in a":
                encoded_relationship = 3
            elif edge_relationship == "none of these":
                encoded_relationship = 4
        elif revert_flag == True:
            if edge_relationship =="a on b":
                encoded_relationship = 2
            elif edge_relationship == "a in b":
                encoded_relationship = 3
            elif edge_relationship == "b on a":
                encoded_relationship = 0
            elif edge_relationship == "b in a":
                encoded_relationship = 1
            elif edge_relationship == "none of these":
                encoded_relationship = 4
        
        return encoded_relationship
   
    def generate_object_edges_by_rules(self, objs, cfg):
        from conceptgraph.slam.slam_classes import MapObjectList
        from conceptgraph.slam.utils import compute_overlap_matrix
        import gc

        # Load the scene map
        # scene_map = MapObjectList()
        # load_scene_map(args, scene_map)
        
        args = cfg
        scene_map = objs

        # Also remove segments that do not have a minimum number of observations
        indices_to_remove = set([])
        for obj_idx in range(len(scene_map)):
            conf = scene_map[obj_idx]["conf"]
            # Remove objects with less than args.min_views_per_object observations
            if len(conf) < args.min_views_per_object:
                indices_to_remove.add(obj_idx)
        indices_to_remove = list(indices_to_remove)
        # combine with also_indices_to_remove and sort the list
        # indices_to_remove = list(set(indices_to_remove + also_indices_to_remove))
        indices_to_remove = list(set(indices_to_remove))
        
        # List of tags in original scene map that are in the pruned scene map
        segment_ids_to_retain = [i for i in range(len(scene_map)) if i not in indices_to_remove]
        # with open(Path(args.cachedir) / "cfslam_scenegraph_invalid_indices.pkl", "wb") as f:
        #     pkl.dump(indices_to_remove, f)
        # print(f"Removed {len(indices_to_remove)} segments")
        
        pruned_scene_map = []
        # pruned_object_tags = []
        for _idx, segmentidx in enumerate(segment_ids_to_retain):
            pruned_scene_map.append(scene_map[segmentidx])
            # pruned_object_tags.append(object_tags[_idx])
        scene_map = MapObjectList(pruned_scene_map)
        # object_tags = pruned_object_tags
        del pruned_scene_map
        # del pruned_object_tags
        gc.collect()
        num_segments = len(scene_map)

        # for i in range(num_segments):
        #     scene_map[i]["caption_dict"] = responses[i]
        #     # scene_map[i]["object_tag"] = object_tags[i]

        # # Save the pruned scene map (create the directory if needed)
        # if not (Path(args.cachedir) / "map").exists():
        #     (Path(args.cachedir) / "map").mkdir(parents=True, exist_ok=True)
        # with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        #     pkl.dump(scene_map.to_serializable(), f)

        print("Computing bounding box overlaps...")
        bbox_overlaps = compute_overlap_matrix(args, scene_map)

        # Construct a weighted adjacency matrix based on similarity scores
        weights = []
        rows = []
        cols = []
        for i in range(num_segments):
            for j in range(i + 1, num_segments):
                if i == j:
                    continue
                if bbox_overlaps[i, j] > 0.01:
                    weights.append(bbox_overlaps[i, j])
                    rows.append(i)
                    cols.append(j)
                    weights.append(bbox_overlaps[i, j])
                    rows.append(j)
                    cols.append(i)

        adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_segments, num_segments))

        # Find the minimum spanning tree of the weighted adjacency matrix
        mst = minimum_spanning_tree(adjacency_matrix)

        # Find connected components in the minimum spanning tree
        _, labels = connected_components(mst)

        components = []
        _total = 0
        if len(labels) != 0:
            for label in range(labels.max() + 1):
                indices = np.where(labels == label)[0]
                _total += len(indices.tolist())
                components.append(indices.tolist())

        # with open(Path(args.cachedir) / "cfslam_scenegraph_components.pkl", "wb") as f:
        #     pkl.dump(components, f)

        # Initialize a list to store the minimum spanning trees of connected components
        minimum_spanning_trees = []
        relations = []
        if len(labels) != 0:
            # Iterate over each connected component
            for label in range(labels.max() + 1):
                component_indices = np.where(labels == label)[0]
                # Extract the subgraph for the connected component
                subgraph = adjacency_matrix[component_indices][:, component_indices]
                # Find the minimum spanning tree of the connected component subgraph
                _mst = minimum_spanning_tree(subgraph)
                # Add the minimum spanning tree to the list
                minimum_spanning_trees.append(_mst)

            relation_queries = []
            for componentidx, component in enumerate(components):
                if len(component) <= 1:
                    continue
                for u, v in zip(
                    minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
                ):
                    segmentidx1 = component[u]
                    segmentidx2 = component[v]
                    _bbox1 = scene_map[segmentidx1]["bbox"]
                    _bbox2 = scene_map[segmentidx2]["bbox"]

                    input_dict = {
                        "object1": {
                            "id": segmentidx1,
                            "bbox_extent": np.round(_bbox1.extent, 4).tolist(),
                            "bbox_center": np.round(_bbox1.center, 4).tolist(),
                            "object_volume": np.round(_bbox1.volume(), 4),
                            # "object_tag": object_tags[segmentidx1],
                        },
                        "object2": {
                            "id": segmentidx2,
                            "bbox_extent": np.round(_bbox2.extent, 4).tolist(),
                            "bbox_center": np.round(_bbox2.center, 4).tolist(),
                            "object_volume": np.round(_bbox2.volume(), 4),
                            # "object_tag": object_tags[segmentidx2],
                        },
                    }
                    # print(f"{input_dict['object1']['object_volume']}, {input_dict['object2']['object_volume']}")

                    relation_queries.append(input_dict)

                    # input_json_str = json.dumps(input_dict)
                    
                    output_dict = input_dict
                    # Example usage within the provided code snippet
                    # Assuming _bbox1 and _bbox2 are the bounding boxes from the code excerpt
                    if self.is_inside(_bbox1, _bbox2) and self.is_bigger(_bbox2, _bbox1):
                        # print(f"a in b")
                        output_dict["object_relation"] = "a in b"
                    elif self.is_inside(_bbox2, _bbox1) and self.is_bigger(_bbox1, _bbox2):
                        # print(f"b in a")     
                        output_dict["object_relation"] = "b in a"          
                    elif self.is_attached(_bbox1, _bbox2) and self.is_bigger(_bbox2, _bbox1):
                        # print("a on b")
                        output_dict["object_relation"] = "a on b"
                    elif self.is_attached(_bbox2, _bbox1) and self.is_bigger(_bbox1, _bbox2):
                        # print("b on a")
                        output_dict["object_relation"] = "b on a"
                    else:
                        # print("none of these")
                        output_dict["object_relation"] = "none of these"

                    relations.append(output_dict)

            # # Save the query JSON to file
            # print("Saving query JSON to file...")
            # with open(Path(args.cachedir) / "cfslam_object_relation_queries.json", "w") as f:
            #     json.dump(relation_queries, f, indent=4)

            # # Saving the output
            # print("Saving object relations to file...")
            # with open(Path(args.cachedir) / "cfslam_object_relations.json", "w") as f:
            #     json.dump(relations, f, indent=4)

        scenegraph_edges = []

        _idx = 0
        for componentidx, component in enumerate(components):
            if len(component) <= 1:
                continue
            for u, v in zip(
                minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
            ):
                segmentidx1 = component[u]
                segmentidx2 = component[v]
                
                # if _idx % 10 == 0: # for debug
                #     save_edges_image_with_relation(args, scene_map, segmentidx1, segmentidx2, relations[_idx]["object_relation"])
                # # print(f"{segmentidx1}, {segmentidx2}, {relations[_idx]['object_relation']}")
                if relations[_idx]["object_relation"] != "none of these":
                    scenegraph_edges.append((segmentidx1, segmentidx2, relations[_idx]["object_relation"]))
                _idx += 1
        # print(f"Created 3D scenegraph with {num_segments} nodes and {len(scenegraph_edges)} edges")

        # with open(Path(args.cachedir) / "cfslam_scenegraph_edges.pkl", "wb") as f:
        #     pkl.dump(scenegraph_edges, f)
            
        # results = {'objects': scene_map.to_serializable()}
        # scene_map_save_path = Path(args.cachedir) / "cfslam_scenegraph_nodes.pkl.gz"
        # with gzip.open(scene_map_save_path, "wb") as f:
        #     pkl.dump(results, f)
        
        return scenegraph_edges, scene_map # scene_map is objects (objs after filtered)
            
    def is_inside(self, bbox1, bbox2):
        """
        Check if bbox1 is inside bbox2.
        """
        # Calculate corners of bbox1
        corners1 = np.array([
            bbox1.center + np.array([dx * bbox1.extent[0] / 2, dy * bbox1.extent[1] / 2, dz * bbox1.extent[2] / 2])
            for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)
        ])
        
        # Transform corners1 to bbox2's coordinate system
        relative_corners = corners1 - bbox2.center
        
        # Check if all corners are within bbox2's extent
        inside = np.all(np.abs(relative_corners) <= bbox2.extent / 2, axis=1)
        
        return np.all(inside)

    def is_on_top(self, bboxA, bboxB):
        # Step 1: Check vertical alignment
        vertical_distance = (bboxA.center[1] - bboxA.extent[1] / 2) - (bboxB.center[1] + bboxB.extent[1] / 2)
        if vertical_distance > 0 and vertical_distance < 0.5:  # Assuming a small threshold for "on"
            # Step 2: Check horizontal overlap
            for axis in [0, 2]:  # Assuming 0:x-axis, 1:y-axis(vertical), 2:z-axis for a y-up system
                distance = abs(bboxA.center[axis] - bboxB.center[axis])
                max_extent = (bboxA.extent[axis] + bboxB.extent[axis]) / 2
                if distance >= max_extent:
                    return False  # No horizontal overlap in this axis
            return True  # Passed both checks
        return False

    def is_attached(self, bbox1, bbox2, threshold=0.1):
        """
        Check if bbox1 is attached to bbox2 within a certain threshold.
        The threshold determines how close the front faces of the bounding boxes need to be.
        """
        # Calculate the distance between the each faces of the two bounding boxes
        distance_x = np.abs(bbox1.center[0] - bbox2.center[0]) - (bbox1.extent[0]/2 + bbox2.extent[0]/2)
        distance_y = np.abs(bbox1.center[1] - bbox2.center[1]) - (bbox1.extent[1]/2 + bbox2.extent[1]/2)
        distance_z = np.abs(bbox1.center[2] - bbox2.center[2]) - (bbox1.extent[2]/2 + bbox2.extent[2]/2)

        if distance_x <= threshold or distance_y <= threshold or distance_z <= threshold:
            return True
        else:
            return False
        
    def is_attached_old(self, bbox1, bbox2, threshold=0.1):
        """
        Check if bbox1 is attached to bbox2 within a certain threshold.
        The threshold determines how close the front faces of the bounding boxes need to be.
        """
        # Calculate the distance between the each faces of the two bounding boxes
        distance_x = np.abs(bbox1.center[0] - bbox2.center[0]) - (bbox1.extent[0]/2 + bbox2.extent[0]/2)
        distance_y = np.abs(bbox1.center[1] - bbox2.center[1]) - (bbox1.extent[1]/2 + bbox2.extent[1]/2)
        distance_z = np.abs(bbox1.center[2] - bbox2.center[2]) - (bbox1.extent[2]/2 + bbox2.extent[2]/2)
        

        # Check if the bounding boxes are aligned vertically and horizontally
        aligned_x = np.abs(bbox1.center[0] - bbox2.center[0]) <= (bbox1.extent[0]/2 + bbox2.extent[0]/2)
        aligned_y = np.abs(bbox1.center[1] - bbox2.center[1]) <= (bbox1.extent[1]/2 + bbox2.extent[1]/2)
        aligned_z = np.abs(bbox1.center[2] - bbox2.center[2]) <= (bbox1.extent[2]/2 + bbox2.extent[2]/2)
        
        if aligned_x or aligned_y or aligned_z:
            if distance_x <= threshold or distance_y <= threshold or distance_z <= threshold:
                return True
            else:
                return False
        else:
            return False

    def is_bigger(self, bbox1, bbox2):
        """
        Check if bbox1 is bigger than bbox2.
        """
        return bbox1.volume() > bbox2.volume()

        
# The SAM based on automatic mask generation, without bbox prompting
def get_sam_segmentation_dense(
    variant:str, model: Any, image: np.ndarray, _device="cuda"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    The SAM based on automatic mask generation, without bbox prompting
    
    Args:
        model: The mask generator or the YOLO model
        image: )H, W, 3), in RGB color space, in range [0, 255]
        
    Returns:
        mask: (N, H, W)
        xyxy: (N, 4)
        conf: (N,)
    '''
    if variant == "sam":
        results = model.generate(image)
        mask = []
        xyxy = []
        conf = []
        for r in results:
            mask.append(r["segmentation"])
            r_xyxy = r["bbox"].copy()
            # Convert from xyhw format to xyxy format
            r_xyxy[2] += r_xyxy[0]
            r_xyxy[3] += r_xyxy[1]
            xyxy.append(r_xyxy)
            conf.append(r["predicted_iou"])
        mask = np.array(mask)
        xyxy = np.array(xyxy)
        conf = np.array(conf)
        return mask, xyxy, conf
    elif variant == "fastsam":
        # The arguments are directly copied from the GSA repo
        results = model(
            image,
            imgsz=1024,
            device=_device,
            retina_masks=True,
            iou=0.9,
            conf=0.4,
            max_det=100,
        )
        raise NotImplementedError
    else:
        raise NotImplementedError


def get_sam_mask_generator(variant:str, device: str | int) -> SamAutomaticMaskGenerator:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=12,
            points_per_batch=144,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=100,
        )
        return mask_generator
    elif variant == "fastsam":
        raise NotImplementedError
        # from ultralytics import YOLO
        # from FastSAM.tools import *
        # FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/FastSAM-x.pt")
        # model = YOLO(args.model_path)
        # return model
    else:
        raise NotImplementedError


def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> list[str]:
    '''
    Convert a text prompt from Tag2Text to a list of classes. 
    '''
    classes = text_prompt.split(',')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes


def process_ai2thor_classes(classes: List[str], add_classes:List[str]=[], remove_classes:List[str]=[]) -> List[str]:
    '''
    Some pre-processing on AI2Thor objectTypes in a scene
    '''
    classes = list(set(classes))
    
    for c in add_classes:
        classes.append(c)
        
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]

    # Split the element in classes by captical letters
    classes = [obj_class.replace("TV", "Tv") for obj_class in classes]
    classes = [re.findall('[A-Z][^A-Z]*', obj_class) for obj_class in classes]
    # Join the elements in classes by space
    classes = [" ".join(obj_class) for obj_class in classes]
    
    return classes

# Prompting SAM with detected boxes
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    
    # if variant == "mobilesam":
    #     from MobileSAM.setup_mobile_sam import setup_model
    #     MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
    #     checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
    #     mobile_sam = setup_model()
    #     mobile_sam.load_state_dict(checkpoint, strict=True)
    #     mobile_sam.to(device=device)
        
    #     sam_predictor = SamPredictor(mobile_sam)
    #     return sam_predictor

    # elif variant == "lighthqsam":
    #     from LightHQSAM.setup_light_hqsam import setup_model
    #     HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
    #     checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
    #     light_hqsam = setup_model()
    #     light_hqsam.load_state_dict(checkpoint, strict=True)
    #     light_hqsam.to(device=device)
        
    #     sam_predictor = SamPredictor(light_hqsam)
    #     return sam_predictor
        
    elif variant == "fastsam":
        raise NotImplementedError
    else:
        raise NotImplementedError
    
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", type=Path, required=True,
    )
    parser.add_argument(
        "--dataset_config", type=str, required=True,
        help="This path may need to be changed depending on where you run this script. "
    )
    
    parser.add_argument("--scene_id", type=str, default="train_3")
    
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--desired-height", type=int, default=480)
    parser.add_argument("--desired-width", type=int, default=640)

    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.5)

    parser.add_argument("--class_set", type=str, default="scene", 
                        choices=["scene", "generic", "minimal", "tag2text", "ram", "none"], 
                        help="If none, no tagging and detection will be used and the SAM will be run in dense sampling mode. ")
    parser.add_argument("--detector", type=str, default="dino", 
                        choices=["yolo", "dino"], 
                        help="When given classes, whether to use YOLO-World or GroundingDINO to detect objects. ")
    parser.add_argument("--add_bg_classes", action="store_true", 
                        help="If set, add background classes (wall, floor, ceiling) to the class set. ")
    parser.add_argument("--accumu_classes", action="store_true",
                        help="if set, the class set will be accumulated over frames")

    parser.add_argument("--sam_variant", type=str, default="sam",
                        choices=['fastsam', 'mobilesam', "lighthqsam"])
    
    parser.add_argument("--save_video", action="store_true")
    
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--use_slow_vis", action="store_true", 
                        help="If set, use vis_result_slow_caption. Only effective when using ram/tag2text. ")
    
    parser.add_argument("--exp_suffix", type=str, default=None,
                        help="The suffix of the folder that the results will be saved to. ")
    
    return parser



# def get_pos_ori(self):
#         pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
#         pos = [x[0] for x in pos_ori]
#         ori = [x[1] for x in pos_ori]
#         return pos, ori
#     # get_agent_state
    
class ConfigDict():

    def __init__(
        self,
        dataset_config: str = "/home/lg1/lujia/VLN_HGT/vlnce_baselines/m2g_config_files/dataset_r2r_finetune.yaml",
        detection_config: str = "/home/lg1/lujia/VLN_HGT/vlnce_baselines/m2g_config_files/detection_r2r_finetune.yaml",
        merge_config: str = "/home/lg1/lujia/VLN_HGT/vlnce_baselines/m2g_config_files/merge_r2r_finetune.yaml",
        edge_config: str = "/home/lg1/lujia/VLN_HGT/vlnce_baselines/m2g_config_files/edge_r2r_finetune.yaml",
    ):
        
        self.dataset_config = load_dataset_config(dataset_config)
        self.detection_config = self.load_yaml_as_dictconfig(detection_config)
        self.merge_config = self.load_yaml_as_dictconfig(merge_config)
        self.edge_config = self.load_yaml_as_dictconfig(edge_config)
        
    def load_yaml_as_dictconfig(self, yaml_file_path: str) -> Dict:
        # Load the YAML file as a DictConfig
        cfg = OmegaConf.load(yaml_file_path)
        return cfg

    def update_cfg(cfg: DictConfig, cfg_update: Dict) -> DictConfig: 
        #Usage:
            # cfg = load_yaml_as_dictconfig(dataset_dictconfig)
            # _cfg = {
            #     "dataset_root": dataset_root,
            #     "dataset_config": dataset_config,
            #     "scene_id": scan,
            #     "spatial_sim_type": "overlap",
            #     "mask_conf_threshold": 0.25,
            #     "match_method": "sim_sum",
            #     "sim_threshold": 1.2,
            #     "dbscan_eps": 0.1,
            #     "gsa_variant": "ram_withbg_allclasses",
            #     "skip_bg": False,
            #     "max_bbox_area_ratio": 0.5,
            #     "save_suffix": "overlap_maskconf0.25_simsum1.2_dbscan.1",
            #     "path": path,
            #     "save_pcd":True,
            # }

            # cfg = update_cfg(cfg, _cfg)
        
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cfg_update))
        return cfg



class FeatureMergeDataset(GradSLAMDataset):
    
    # color_paths is a list of str, The str is the key of the image dict.
    # In the GradSLAMDatset, you need to load image form files.
    # But in the realtime finetune, the iamges are generate by the habitat env.
    # So the image is already in the memory.
    # So the color_paths is the key of the image_rgb(or bgr) dict.
    # The depth_paths is the key of the image_depth dict.
    
    def __init__(
        self,
        config_dict: Dict,
        observation: List[Dict],
        # _scan_list: List[str],
        # _vp_list: List[str],
        sensor_state,
        HFOV = 90, # Our R2R dataset HFOV is 60.
        WIDTH = 224, 
        HEIGHT = 224,
        _rgb_keys = ["rgb", "rgb_30", "rgb_60", "rgb_90", "rgb_120", "rgb_150", "rgb_180", "rgb_210", "rgb_240", "rgb_270", "rgb_300", "rgb_330"],
        _depth_keys = ["depth", "depth_30", "depth_60", "depth_90", "depth_120", "depth_150", "depth_180", "depth_210", "depth_240", "depth_270", "depth_300", "depth_330"]
    ):
        
        self.rgb_keys = _rgb_keys
        self.depth_keys = _depth_keys
        
        self.image_rgb_dict = {} # The rgb dict
        self.image_depth_dict = {} # The depth dict

        self.observation_2_image_dict(observation)
        
        
        # self.poses_raw = _poses_raw
        self.sensor_state = sensor_state
        self.load_poses()
                
        self.color_paths = []
        self.depth_paths = []
         
        self.fx, self.fy, self.cx, self.cy = self.calculate_fx_fy_cx_cy(HFOV, WIDTH, HEIGHT)
        self.camera_intrinsics = self.load_camera_intrinsics(HFOV, WIDTH, HEIGHT)
         
        self.config_dict = config_dict
        super().__init__(
            config_dict)        
        
        ### Need to judge the
        
        # self.scan_dict = {}
        # self.vp_dict = {}
        # self.scan_dict, self.vp_dict = self.observation_2_scan_vp_dict(_scan_list, _vp_list)
        

        # This should have env, scan, vp.
    
    def observation_2_scan_vp_dict(self, scan_list, vp_list):
        
        _scan_dict = {}
        _vp_dict = {}
        
        rgb_keys = self.rgb_keys
        
        for i, key in enumerate(rgb_keys):
            _scan_dict[key] = scan_list[i]
            _vp_dict[key] = vp_list[i]
            
        return _scan_dict, _vp_dict
            
    
    def observation_2_image_dict(self, observation):
        
        image_rgb_dict = {}
        image_depth_dict = {}

        # rgb_keys = ["rgb", "rgb_30", "rgb_60", "rgb_90", "rgb_120", "rgb_150", "rgb_180", "rgb_210", "rgb_240", "rgb_270", "rgb_300", "rgb_330"]
        # depth_keys = ["depth", "depth_30", "depth_60", "depth_90", "depth_120", "depth_150", "depth_180", "depth_210", "depth_240", "depth_270", "depth_300", "depth_330"]
        
        rgb_keys = self.rgb_keys
        depth_keys = self.depth_keys
        
        for rgb_key, depth_key in zip(rgb_keys, depth_keys):
            if rgb_key in observation:
                image_rgb_dict[rgb_key] = observation[rgb_key]
            if depth_key in observation:
                image_depth_dict[depth_key] = observation[depth_key]
                    
        self.image_rgb_dict = image_rgb_dict
        self.image_depth_dict = image_depth_dict
        
        return self.image_rgb_dict, self.image_depth_dict
    
    def get_filepaths(self):
        # get_the key list of the image_rgb_dict and image_depth_dict
        
        color_paths = list(self.image_rgb_dict.keys())
        depth_paths = list(self.image_depth_dict.keys())
        
        # self.color_paths = color_paths
        # self.depth_paths = depth_paths
        embeddings_paths = []
        
        return color_paths, depth_paths, embeddings_paths
        
    def calculate_camera_intrinsic(self, HFOV, WIDTH, HEIGHT):
        import math 
        camera_intrinsic = np.array([
            [((1 / np.tan(math.radians(HFOV) / 2.))*WIDTH) / 2, 0., WIDTH/2, 0.],
            [0., ((1 / np.tan(math.radians(HFOV) / 2.))*HEIGHT) / 2, HEIGHT/2, 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]])
        
        return camera_intrinsic
    
    def load_camera_intrinsics(self, HFOV, WIDTH, HEIGHT):
        
        camera_intrinsic = self.calculate_camera_intrinsic(HFOV, WIDTH, HEIGHT)
        
        camera_intrinsics = []
        
        for key in self.image_rgb_dict.keys():
            camera_intrinsics.append(camera_intrinsic)
            
        return camera_intrinsics
    
    def calculate_fx_fy_cx_cy(self, HFOV, WIDTH, HEIGHT):
        import math
        
        camera_intrinsic = np.array([
            [((1 / np.tan(math.radians(HFOV) / 2.))*WIDTH) / 2, 0., WIDTH/2, 0.],
            [0., ((1 / np.tan(math.radians(HFOV) / 2.))*HEIGHT) / 2, HEIGHT/2, 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]])
        
        fx, fy ,cx, cy = camera_intrinsic[0, 0], camera_intrinsic[1, 1], camera_intrinsic[0, 2], camera_intrinsic[1, 2]
        
        return fx, fy, cx, cy
    
    def quaternion_to_rotation_matrix(self, quat):
        qw, qx, qy, qz = quat.w, quat.x, quat.y, quat.z
        return np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, 0],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw, 0],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2, 0],
            [0, 0, 0, 1]
        ])

    def transformation_matrix(self, pos, quat):
        x, y, z = pos
        T = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
        
        R = self.quaternion_to_rotation_matrix(quat)
        
        M = np.dot(T, R)
        return M
    
    def load_poses(self):
        
        poses = []
        
        for key in self.rgb_keys:
            sensor = self.sensor_state[key]
            pos = sensor.position
            quat = sensor.rotation
            pose = self.transformation_matrix(pos, quat)
            poses.append(torch.tensor(pose))
        # The poses has the same length with the self.image_dict.
        # They are corresponding to each other.

        return poses
    
    def __getitem__(self, index):
        
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        
        color = self.image_rgb_dict[color_path]
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        
        depth = self.image_depth_dict[depth_path]
        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)
        
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        K = datautils.scale_intrinsics(
            K, self.height_downsample_ratio, self.width_downsample_ratio
        )
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )
        
   

class ObjFeatureGenerator():
    def __init__(
        self,
        generator_device = "cuda:0",
        # cg_config= "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/configs/slam_pipeline/base.yaml"
        ):
        
        self.detection_config = {}
        self.merge_config = {}
        
        self.grounding_dino_model = None
        
        self.sam_predictor = None
        self.mask_generator = None
        
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_model = None
        self.clip_tokenizer = None
        
        self.tagging_model = None
        self.tagging_transform = None
        self.classes = None
        self.specified_tags = None
        self.global_classes = set()
        
        self.use_bg = False
        
        # Device part
        # The self.device is the original device, while the self.generator_device is the device for the generator
        self.generator_device = generator_device
        # The data processed by the generator will be stored in the self.device
        
        ## For save and debug
        self.step = 0
    
    def set_step(self, step):
        # For debug
        self.step = step
    
    def get_detection_config(self, config_dict):
        self.detection_config = config_dict

    def get_merge_config(self, config_dict):
        self.merge_config = config_dict
    
    def get_config(self, detection_dict, merge_dict):
        
        self.get_detection_config(detection_dict)
        self.get_merge_config(merge_dict)

            
    def obj_feature_generate(self, dataset):
        
        # image_rgb is a list of np.array [env * 12 * 224 * 224 * 3]
        # change for into batch process
        
        # set device
        _device = self.get_device()
        
        # _image_rgb = []
        # _image_bgr = []
        detections_list = []
        classes_list = []
        
        with torch.set_grad_enabled(False):
            
            for idx in range(len(dataset)):
                
                image_key = dataset.rgb_keys[idx]
                _image_rgb = dataset.image_rgb_dict[image_key]

                _classes, _text_prompt, _caption = self.tag2text_inference(
                    cfg=self.detection_config,
                    image_rgb=_image_rgb,
                    tagging_model=self.tagging_model,
                    tagging_transform=self.tagging_transform,
                    specified_tags=self.specified_tags, 
                    device = _device
                )
                
                _detections = self.detection_inference(
                    cfg=self.detection_config,
                    image_rgb = _image_rgb, 
                    classes = _classes, 
                    sam_variant = self.detection_config.sam_variant, 
                    mask_generator = self.mask_generator,
                    grounding_dino_model = self.grounding_dino_model,
                    device = _device
                )
                
                _image_crops, _image_feats, _text_feats = self.segementation_inference(
                    cfg=self.detection_config,
                    image_rgb = _image_rgb,
                    detections = _detections,
                    text_prompt = _text_prompt,
                    caption = _caption,
                    sam_predictor = self.sam_predictor,
                    clip_model = self.clip_model,
                    clip_preprocess = self.clip_preprocess,
                    clip_tokenizer = self.clip_tokenizer,
                    classes = _classes,
                    device = _device
                )

                # Convert the detections to a dict. The elements are in np.array
                detections = {
                    "xyxy": _detections.xyxy,
                    "confidence": _detections.confidence,
                    "class_id": _detections.class_id,
                    "mask": _detections.mask,
                    "classes": _classes,
                    "image_crops": _image_crops,
                    "image_feats": _image_feats,
                    "text_feats": _text_feats,
                }
                
                if self.detection_config.class_set in ["ram", "tag2text"]:
                    detections["tagging_caption"] = _caption
                    detections["tagging_text_prompt"] = _text_prompt
                    
                detections_list.append(detections)
                classes_list.append(_classes)
            
        return detections_list, classes_list
        
        # # save the detections using pickle
        # # Here we use gzip to compress the file, which could reduce the file size by 500x
        # with gzip.open(detections_save_path, "wb") as f:
        #     pickle.dump(results, f)
    
        # # save global classes
        # with open(self.dataset_root / self.scene_id / f"gsa_classes_{save_name}.json", "w") as f:
        #     json.dump(list(global_classes), f)
        
        # if self.save_video:
        #     frames.append(annotated_image)   
        # if self.save_video:
        #     imageio.mimsave(video_save_path, frames, fps=10)
        #     print(f"Video saved to {video_save_path}")
        
        
    
    def init_model(self):
        
        # # Define the device for training and preprocessing
        # training_device = torch.device('cuda:0')  # GPU A for training
        # preprocessing_device = torch.device(f'cuda:{local_rank}')  # Other GPUs for preprocessing

        # # Create and move the training model to the training device
        # training_model = MyTrainingModel().to(training_device)

        # # Create and move the preprocessing model to the preprocessing device
        # preprocessing_model = MyPreprocessingModel().to(preprocessing_device)

        if self.detection_config == {}:
            raise ValueError("The detection config is not loaded. ")
                 
        if self.detection_config.device == self.generator_device:
            Warning("Using the same deivce for the generator and VLN training or other process. ")
        
        self._init_GroudingDINO(self.generator_device)
        self._init_SAM(self.detection_config, self.generator_device)
        self._init_CLIP(self.generator_device)
        self._init_tagger(self.detection_config, self.generator_device)

    def _init_GroudingDINO(self, _device):
        
        ### Initialize the Grounding DINO model ###
        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
            device=_device
        )
        
        self.grounding_dino_model = grounding_dino_model
        
    def _init_SAM(self, cfg, _device):
        
        # cfg.sam_variant is in the parser. The default is "sam".
        
        ### Initialize the SAM model ###
        if cfg.class_set == "none":
            mask_generator = get_sam_mask_generator(cfg.sam_variant, _device)
            self.mask_generator = mask_generator
        else:
            sam_predictor = get_sam_predictor(cfg.sam_variant, _device)
            self.sam_predictor = sam_predictor
        
    def _init_CLIP(self, _device):
        
        ### Initialize the CLIP model
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(_device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
    
    def _init_tagger(self, cfg, _device):
        
        tagging_model = None
        tagging_transform = None
        classes = None
        specified_tags = None
        
        if cfg.class_set == "scene":
            # Load the object meta information
            obj_meta_path = cfg.dataset_root / cfg.scene_id / "obj_meta.json"
            with open(obj_meta_path, "r") as f:
                obj_meta = json.load(f)
            # Get a list of object classes in the scene
            classes = process_ai2thor_classes(
                [obj["objectType"] for obj in obj_meta],
                add_classes=[],
                remove_classes=['wall', 'floor', 'room', 'ceiling']
            )
        elif cfg.class_set == "generic":
            classes = FOREGROUND_GENERIC_CLASSES
        elif cfg.class_set == "minimal":
            classes = FOREGROUND_MINIMAL_CLASSES
        elif cfg.class_set in ["tag2text", "ram"]:
            ### Initialize the Tag2Text or RAM model ###
            
            if cfg.class_set == "tag2text":
                # The class set will be computed by tag2text on each image
                # filter out attributes and action categories which are difficult to grounding
                delete_tag_index = []
                for i in range(3012, 3429):
                    delete_tag_index.append(i)

                specified_tags='None'
                # load model
                tagging_model = tag2text.tag2text_caption(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                                                        image_size=384,
                                                        vit='swin_b',
                                                        delete_tag_index=delete_tag_index)
                # threshold for tagging
                # we reduce the threshold to obtain more tags
                tagging_model.threshold = 0.64 
            elif cfg.class_set == "ram":
                tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                                            image_size=384,
                                            vit='swin_l')
                
            tagging_model = tagging_model.eval().to(_device)
            
            # initialize Tag2Text
            tagging_transform = TS.Compose([
                TS.Resize((384, 384)),
                TS.ToTensor(), 
                TS.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            
            classes = None
        elif cfg.class_set == "none":
            classes = ['item']
        else:
            raise ValueError("Unknown cfg.class_set: ", cfg.class_set)
        
        self.tagging_model = tagging_model
        self.tagging_transform = tagging_transform
        self.classes = classes
        self.specified_tags = specified_tags
        
        if cfg.class_set not in ["ram", "tag2text"]:
            print("There are total", len(classes), "classes to detect. ")
        elif cfg.class_set == "none":
            print("Skipping tagging and detection models. ")
        else:
            print(f"{cfg.class_set} will be used to detect classes. ")
            
    def observations_2_imagetensor(self, observation, _device):
                
        # The input is the form the vlnce_baseline ->extract_instruction_tokens ->observation.
        # The observation is a list of dict. The list dim is num_process(the env). 
        # In each dict, the key are "rgb", "rgb_30", ......, "rgb_330".
        # There are 12 rgb np.array, for each the shape is [224*224*3]
        
        rgb_keys = ["rgb", "rgb_30", "rgb_60", "rgb_90", "rgb_120", "rgb_150", "rgb_180", "rgb_210", "rgb_240", "rgb_270", "rgb_300", "rgb_330"]
        image_list = []
        
        for env in observation:
            # Extract the RGB arrays and stack them
            rgb_arrays = [env[key] for key in rgb_keys]
            stacked_rgb = np.stack(rgb_arrays, axis=0)  # Shape: [12, 224, 224, 3]
            image_list.append(stacked_rgb)

        # Convert the list of stacked arrays to a single tensor
        image_tensor = torch.tensor(np.stack(image_list, axis=0))  # Shape: [env, 12, 224, 224, 3]
        image_tensor = image_tensor.to(_device)
        
        return image_tensor
    
    def get_device(self):
        if self.detection_config.device == self.generator_device:
            return self.device
        elif self.detection_config.device != self.generator_device:
            return self.generator_device
        else:
            raise NameError("The device setting has error.")
            
            
    def tag2text_inference(
        self,
        cfg, 
        image_rgb, 
        tagging_model, 
        tagging_transform, 
        specified_tags, 
        device):
        
        image_pil = Image.fromarray(image_rgb)
        
        ### Tag2Text ###
        if cfg.class_set in ["ram", "tag2text"]:
            raw_image = image_pil.resize((384, 384))
            raw_image = tagging_transform(raw_image).unsqueeze(0).to(device)
            
            if cfg.class_set == "ram":
                res = inference_ram(raw_image , tagging_model)
                caption="NA"
            elif cfg.class_set == "tag2text":
                res = inference_tag2text.inference(raw_image , tagging_model, specified_tags)
                caption=res[2]

            # Currently ", " is better for detecting single tags
            # while ". " is a little worse in some case
            text_prompt=res[0].replace(' |', ',')
            
            # Add "other item" to capture objects not in the tag2text captions. 
            # Remove "xxx room", otherwise it will simply include the entire image
            # Also hide "wall" and "floor" for now...
            add_classes = ["other item"]
            remove_classes = [
                "room", "kitchen", "office", "house", "home", "building", "corner",
                "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
                "apartment", "image", "city", "blue", "skylight", "hallway", 
                "bureau", "modern", "salon", "doorway", "wall lamp"
            ]
            bg_classes = ["wall", "floor", "ceiling"]

            if cfg.add_bg_classes:
                add_classes += bg_classes
            else:
                remove_classes += bg_classes

            classes = process_tag_classes(
                text_prompt, 
                add_classes = add_classes,
                remove_classes = remove_classes,
            )
            
        # add classes to global classes
        self.global_classes.update(classes)
        
        if cfg.accumu_classes:
            # Use all the classes that have been seen so far
            classes = list(self.global_classes)
            
        return classes, text_prompt, caption
    
    def detection_inference(
        self, 
        cfg, 
        image_rgb, 
        classes,
        sam_variant, 
        mask_generator,
        grounding_dino_model,
        device
        ):
        
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        ### Detection and segmentation ###
        if cfg.class_set == "none":
            # Directly use SAM in dense sampling mode to get segmentation
            mask, xyxy, conf = get_sam_segmentation_dense(
                sam_variant, mask_generator, image_rgb, device)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            # image_crops, image_feats, text_feats = compute_clip_features(
            #     image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device)

            # ### Visualize results ###
            # annotated_image, labels = vis_result_fast(
            #     image, detections, classes, instance_random_color=True)
            
            # cv2.imwrite(vis_save_path, annotated_image)
        else:
            if cfg.detector == "dino":
                # Using GroundingDINO to detect and SAM to segment
                detections = grounding_dino_model.predict_with_classes(
                    image=image_bgr, # This function expects a BGR image...
                    classes=classes,
                    box_threshold=cfg.box_threshold,
                    text_threshold=cfg.text_threshold,
                )
            
                if len(detections.class_id) > 0:
                    ### Non-maximum suppression ###
                    # print(f"Before NMS: {len(detections.xyxy)} boxes")
                    nms_idx = torchvision.ops.nms(
                        torch.from_numpy(detections.xyxy), 
                        torch.from_numpy(detections.confidence), 
                        cfg.nms_threshold
                    ).numpy().tolist()
                    # print(f"After NMS: {len(detections.xyxy)} boxes")

                    detections.xyxy = detections.xyxy[nms_idx]
                    detections.confidence = detections.confidence[nms_idx]
                    detections.class_id = detections.class_id[nms_idx]
                    
                    # Somehow some detections will have class_id=-1, remove them
                    valid_idx = detections.class_id != -1
                    detections.xyxy = detections.xyxy[valid_idx]
                    detections.confidence = detections.confidence[valid_idx]
                    detections.class_id = detections.class_id[valid_idx]

                    # # Somehow some detections will have class_id=-None, remove them
                    # valid_idx = [i for i, val in enumerate(detections.class_id) if val is not None]
                    # detections.xyxy = detections.xyxy[valid_idx]
                    # detections.confidence = detections.confidence[valid_idx]
                    # detections.class_id = [detections.class_id[i] for i in valid_idx]
        
        return detections
    
    def segementation_inference(
        self,
        cfg, 
        image_rgb, 
        detections,
        text_prompt,
        caption,
        sam_predictor,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
        classes,
        device
        ):

        if len(detections.class_id) > 0:
            ### Segment Anything ###
            detections.mask = get_sam_segmentation_from_xyxy(
                sam_predictor=sam_predictor,
                image=image_rgb,
                xyxy=detections.xyxy
            )
            # Compute and save the clip features of detections  
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device)
        else:
            image_crops, image_feats, text_feats = [], [], []
        
        ### Visualize results ###
        if cfg.save_seg:
            
            # random a save name using random number, don't using step
            # vis_save_path = cfg.debug_save_path + "/" + "seg_saves" +"/"+ f"{random.randint(0, 100000):06d}"
            save_image_id = random.randint(0, 100)
            vis_save_path = cfg.debug_save_path + "/" + "seg_saves" 
            original_image_save_path = vis_save_path + "/" + f"step{self.step}_ori_{save_image_id:06d}" + ".jpg"
            seg_image_save_path = vis_save_path + "/" + f"step{self.step}_seg_{save_image_id:06d}" + ".jpg"
            
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            annotated_image, labels = vis_result_fast(image_bgr, detections, classes)
            
            # save the annotated grounded-sam image
            if cfg.class_set in ["ram", "tag2text"] and cfg.use_slow_vis:
                annotated_image_caption = vis_result_slow_caption(
                    image_rgb, detections.mask, detections.xyxy, labels, caption, text_prompt)
                Image.fromarray(annotated_image_caption).save(seg_image_save_path)
                cv2.imwrite(original_image_save_path, image_bgr)
            else:
                cv2.imwrite(seg_image_save_path, annotated_image)
                cv2.imwrite(original_image_save_path, image_bgr)
        
        return image_crops, image_feats, text_feats

    def process_detections_for_merge(self, detecitions_gobs, classes_gobs, dataset:FeatureMergeDataset, cur_step_name:str):
        
        if self.merge_config == {}:
            raise ValueError("The merge config is not loaded. ")
        cfg = self.merge_config
        
        # _device = self.get_device()
        # # objects = MapObjectList(device=_device)
        
        fg_detections_list = DetectionList()
        bg_detections_list = DetectionList()
        
        for idx in range(len(dataset)):
            # get color image
            color_path = dataset.color_paths[idx]
            # image_original_pil = dataset.image_rgb_dict[color_path]
            ## for VLN
            vp_idx = cur_step_name
            # The vp_idx = None may have trouble in the future.

            # image/detection have the vp_id

            color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]
            # image_rgb = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
            # Get the RGB image
            color_np = color_tensor.cpu().numpy() # (H, W, 3)
            image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
            # assert image_rgb.max() > 1, "Image is not in range [0, 255]"
            
            # Get the depth image
            depth_tensor = depth_tensor[..., 0]
            depth_array = depth_tensor.cpu().numpy()

            # Get the intrinsics matrix
            cam_K = intrinsics.cpu().numpy()[:3, :3]
            
            # load grounded SAM detections
            gobs = None # stands for grounded SAM observations
            
            gobs = detecitions_gobs[idx]
            
            classes = classes_gobs[idx]

            # color_path = Path(color_path)
            # detections_path = color_path.parent.parent / cfg.detection_folder_name / color_path.name
            # detections_path = detections_path.with_suffix(".pkl.gz")
            # color_path = str(color_path)
            # detections_path = str(detections_path)

            # get pose, this is the untrasformed pose.
            unt_pose = dataset.poses[idx]
            unt_pose = unt_pose.cpu().numpy()
            
            # Don't apply any transformation otherwise
            adjusted_pose = unt_pose
            
            fg_detection_list, bg_detection_list = gobs_to_detection_list(
                cfg = cfg,
                image = image_rgb,
                depth_array = depth_array,
                cam_K = cam_K,
                idx = idx,
                gobs = gobs,
                vp_idx = vp_idx,
                trans_pose = adjusted_pose, # This is pose
                class_names = classes,
                BG_CLASSES = BG_CLASSES,
                color_path = color_path,
            )
                
            fg_detections_list.extend(fg_detection_list)
            bg_detections_list.extend(bg_detection_list)
                
            # if cfg.save_objects_all_frames:
            #     save_all_path = save_all_folder / f"{idx:06d}.pkl.gz"
            #     objects_to_save = MapObjectList([
            #         _ for _ in objects if _['num_detections'] >= cfg.obj_min_detections
            #     ])
                
            #     objects_to_save = prepare_objects_save_vis(objects_to_save)  #We want to save this!!!!!!
                
            #     if not cfg.skip_bg:
            #         bg_objects_to_save = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
            #         bg_objects_to_save = prepare_objects_save_vis(bg_objects_to_save)
            #     else:
            #         bg_objects_to_save = None
                
            #     result = {
            #         "camera_pose": adjusted_pose,
            #         "objects": objects_to_save,
            #         "bg_objects": bg_objects_to_save,
            #     }
            #     with gzip.open(save_all_path, 'wb') as f:
            #         pickle.dump(result, f)
            
            # if cfg.vis_render:
            #     objects_vis = MapObjectList([
            #         copy.deepcopy(_) for _ in objects if _['num_detections'] >= cfg.obj_min_detections
            #     ])
                
            #     if cfg.class_agnostic:
            #         objects_vis.color_by_instance()
            #     else:
            #         objects_vis.color_by_most_common_classes(class_colors)
                
                # rendered_image, vis = obj_renderer.step(
                #     image = image_original_pil,
                #     gt_pose = adjusted_pose,
                #     new_objects = objects_vis,
                #     paint_new_objects=False,
                #     return_vis_handle = cfg.debug_render,
                # )

                # if cfg.debug_render:
                #     vis.run()
                #     del vis
                
                # # Convert to uint8
                # if rendered_image is not None:
                #     rendered_image = (rendered_image * 255).astype(np.uint8)
                #     frames.append(rendered_image)
                
            # print(
            #     f"Finished image {idx} of {len(dataset)}", 
            #     f"Now we have {len(objects)} objects.",
            #     f"Effective objects {len([_ for _ in objects if _['num_detections'] >= cfg.obj_min_detections])}"
            # )
    
        return fg_detections_list, bg_detection_list
    
    def merge_objs_detections(self, objects, fg_detections_list, bg_detections_list, cfg):
        
        if not cfg.skip_bg:
            # Handle the background detection separately 
            # Each class of them are fused into the map as a single object
            bg_objects = {
                c: None for c in BG_CLASSES
            }
        else:
            bg_objects = None
        
        if len(bg_detections_list) > 0:
            for detected_object in bg_detections_list:
                class_name = detected_object['class_name'][0]
                if bg_objects[class_name] is None:
                    bg_objects[class_name] = detected_object
                else:
                    matched_obj = bg_objects[class_name]
                    matched_det = detected_object
                    bg_objects[class_name] = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)

        if cfg.use_contain_number:
            xyxy = fg_detections_list.get_stacked_values_torch('xyxy', 0)
            contain_numbers = compute_2d_box_contained_batch(xyxy, cfg.contain_area_thresh)
            for i in range(len(fg_detections_list)):
                fg_detections_list[i]['contain_number'] = [contain_numbers[i]]
            
        if len(objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detections_list)):
                objects.append(fg_detections_list[i])
                
        spatial_sim = compute_spatial_similarities(cfg, fg_detections_list, objects)
        visual_sim = compute_visual_similarities(cfg, fg_detections_list, objects)
        agg_sim = aggregate_similarities(cfg, spatial_sim, visual_sim)
        
        # Compute the contain numbers for each detection
        if cfg.use_contain_number:
            # Get the contain numbers for all objects
            contain_numbers_objects = torch.Tensor([obj['contain_number'][0] for obj in objects])
            detection_contained = contain_numbers > 0 # (M,)
            object_contained = contain_numbers_objects > 0 # (N,)
            detection_contained = detection_contained.unsqueeze(1) # (M, 1)
            object_contained = object_contained.unsqueeze(0) # (1, N)                

            # Get the non-matching entries, penalize their similarities
            xor = detection_contained ^ object_contained
            agg_sim[xor] = agg_sim[xor] - cfg.contain_mismatch_penalty
        
        # Threshold sims according to cfg. Set to negative infinity if below threshold
        agg_sim[agg_sim < cfg.sim_threshold] = float('-inf')
        
        objects = merge_detections_to_objects(cfg, fg_detections_list, objects, agg_sim) # 
        
        # # Perform post-processing periodically if told so
        # if cfg.denoise_interval > 0 and (idx+1) % cfg.denoise_interval == 0:
        #     objects = denoise_objects(cfg, objects)
        # if cfg.filter_interval > 0 and (idx+1) % cfg.filter_interval == 0:
        #     objects = filter_objects(cfg, objects)
        # if cfg.merge_interval > 0 and (idx+1) % cfg.merge_interval == 0:
        #     objects = merge_objects(cfg, objects)
        
        if bg_objects is not None:
            bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
            bg_objects = denoise_objects(cfg, bg_objects)
            
        objects = denoise_objects(cfg, objects)  
        
        # Save the full point cloud before post-processing
        if cfg.save_pcd:
            # ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            class_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)
            class_colors = {str(i): c for i, c in enumerate(class_colors)}
            
            results = {
                'objects': objects.to_serializable(),
                'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
                'cfg': cfg,
                # 'class_names': classes,
                'class_colors': class_colors,
            }

            # pcd_save_path = cfg.dataset_root / \
            #     cfg.scene_id / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_{cfg.save_suffix}.pkl.gz"
            # # make the directory if it doesn't exist
            # pcd_save_path.parent.mkdir(parents=True, exist_ok=True)
            # pcd_save_path = str(pcd_save_path)
            
            # pcd_save_path = cfg.debug_save_path +'/'+ 'pcd_saves' +'/'+ f"full_pcd_{cfg.gsa_variant}_{cfg.save_suffix}_step{self.step}.pkl.gz"
            pcd_save_path = cfg.debug_save_path +'/'+ 'pcd_saves' +'/'+ f"full_pcd_step{self.step}.pkl.gz"
            
        #     with gzip.open(pcd_save_path, "wb") as f:
        #         pickle.dump(results, f)
        #     print(f"Saved full point cloud to {pcd_save_path}")
        
        objects = filter_objects(cfg, objects)
        objects = merge_objects(cfg, objects)
        
        # Save again the full point cloud after the post-processing
        if cfg.save_pcd:
            results['objects'] = objects.to_serializable()
            pcd_save_path = pcd_save_path[:-7] + "_post.pkl.gz"
            with gzip.open(pcd_save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved full point cloud after post-processing to {pcd_save_path}")
            
        # if cfg.save_objects_all_frames:
        #     save_meta_path = save_all_folder / f"meta.pkl.gz"
        #     with gzip.open(save_meta_path, "wb") as f:
        #         pickle.dump({
        #             'cfg': cfg,
        #             'class_names': classes,
        #             'class_colors': class_colors,
        #         }, f)
            
        # if cfg.vis_render:
        #     # Still render a frame after the post-processing
        #     objects_vis = MapObjectList([
        #         _ for _ in objects if _['num_detections'] >= cfg.obj_min_detections
        #     ])

        #     if cfg.class_agnostic:
        #         objects_vis.color_by_instance()
        #     else:
        #         objects_vis.color_by_most_common_classes(class_colors)
            
        #     rendered_image, vis = obj_renderer.step(
        #         image = image_original_pil,
        #         gt_pose = adjusted_pose,
        #         new_objects = objects_vis,
        #         paint_new_objects=False,
        #         return_vis_handle = False,
        #     )
            
        #     # Convert to uint8
        #     rendered_image = (rendered_image * 255).astype(np.uint8)
        #     frames.append(rendered_image)
            
        #     # Save frames as a mp4 video
        #     frames = np.stack(frames)
        #     video_save_path = (
        #         cfg.dataset_root
        #         / cfg.scene_id
        #         / ("objects_mapping-%s-%s.mp4" % (cfg.gsa_variant, cfg.save_suffix))
        #     )
        #     imageio.mimwrite(video_save_path, frames, fps=10)
        #     print("Save video to %s" % video_save_path)
        # print(scene_map)
        
        # The code below is very same with filer objs
        # Just more a oriented_bouding_box convert.
        # pcd_filter_number = cfg.obj_min_points
        # print("Before pcd filtering", len(objects))
        # for obj in objects:
        #     if isinstance(obj["bbox"], open3d.geometry.AxisAlignedBoundingBox):
        #         if len(obj["pcd"].points) >= pcd_filter_number:
        #             obj["bbox"] = obj["bbox"].get_oriented_bounding_box()
        #         else:
        #             objects.remove(obj)
        # print("After pcd filtering", len(objects))

        # if self.use_bg:
        #     objects.extend(bg_objects)
        
        for obj in objects:
            if isinstance(obj["bbox"], open3d.geometry.AxisAlignedBoundingBox):
                obj["bbox"] = obj["bbox"].get_oriented_bounding_box()
        return objects, bg_objects
       
    def get_class_colors(self, classes, cfg):

        class_colors = get_classes_colors(classes)
        class_colors = {str(k): v for k, v in class_colors.items()}
        return class_colors
    
    def as_intrinsics_matrix(self, intrinsics):
        """
        Get matrix representation of intrinsics.

        """
        K = np.eye(3)
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = intrinsics[2]
        K[1, 2] = intrinsics[3]
        return K

    def load_yaml_as_dictconfig(self, yaml_file_path: str) -> Dict:
        # Load the YAML file as a DictConfig
        cfg = OmegaConf.load(yaml_file_path)
        return cfg
            
 
    
class ObjFeatureGeneratorBatch(ObjFeatureGenerator):
    def __init__(
        self,):
        super().__init__()
    
    def tag2text_inference(
        self,
        cfg, 
        images_rgb,  # Changed to a batch of images
        tagging_model, 
        tagging_transform, 
        specified_tags, 
        device):
        
        # Convert list of images to a batch tensor
        raw_images = []
        for image_rgb in images_rgb:
            image_pil = Image.fromarray(image_rgb)
            resized_image = image_pil.resize((384, 384))
            transformed_image = tagging_transform(resized_image).unsqueeze(0)
            raw_images.append(transformed_image)
        raw_images = torch.cat(raw_images).to(device)  # Create a batch tensor

        classes_batch = []
        text_prompts_batch = []
        captions_batch = []

        ### Tag2Text ###
        if cfg.class_set in ["ram", "tag2text"]:
            if cfg.class_set == "ram":
                res_batch = [inference_ram(raw_image.unsqueeze(0), tagging_model) for raw_image in raw_images]
                captions = ["NA"] * len(images_rgb)
            elif cfg.class_set == "tag2text":
                res_batch = [inference_tag2text.inference(raw_image.unsqueeze(0), tagging_model, specified_tags) for raw_image in raw_images]
                captions = [res[2] for res in res_batch]

            for res, caption in zip(res_batch, captions):
                # Currently ", " is better for detecting single tags
                # while ". " is a little worse in some case
                text_prompt = res[0].replace(' |', ',')
                
                # Add "other item" to capture objects not in the tag2text captions. 
                # Remove "xxx room", otherwise it will simply include the entire image
                # Also hide "wall" and "floor" for now...
                add_classes = ["other item"]
                remove_classes = [
                    "room", "kitchen", "office", "house", "home", "building", "corner",
                    "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
                    "apartment", "image", "city", "blue", "skylight", "hallway", 
                    "bureau", "modern", "salon", "doorway", "wall lamp"
                ]
                bg_classes = ["wall", "floor", "ceiling"]

                if cfg.add_bg_classes:
                    add_classes += bg_classes
                else:
                    remove_classes += bg_classes

                classes = process_tag_classes(
                    text_prompt, 
                    add_classes=add_classes,
                    remove_classes=remove_classes,
                )
                
                # add classes to global classes
                self.global_classes.update(classes)
                
                if cfg.accumu_classes:
                    # Use all the classes that have been seen so far
                    classes = list(self.global_classes)
                
                classes_batch.append(classes)
                text_prompts_batch.append(text_prompt)
                captions_batch.append(caption)
            
        return classes_batch, text_prompts_batch, captions_batch