import os
import numpy as np
import h5py
import json

from os import path
import time

from numpy import save
from requests import get
# from conceptgraph.slam.cfslam_pipeline_batch import *
from omegaconf import OmegaConf, DictConfig
from typing import Dict

import h5py
import json

import sys

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

        filtered_objs, filtered_objs_feature, filtered_objs_pose = self.merge_feature(scan, path, self.obj_feature_name, self.obj_pose_name)
        edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend = self.get_obj_edges_finetune(scan, filtered_objs_pose, filtered_objs, self.alledges_dict)
        
        # return filtered_objs, objs_feature, objs_pose, edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend
        return filtered_objs, filtered_objs_feature, filtered_objs_pose, edges_objects_extend, edges_relationship_extend
    
    def load_obj_edge_vp(self):
        self.load_allobjs_from_hdf5()
        self.get_edges_from_hdf5()
        self.load_allvps_pos_from_connectivity_json()
        return self.allobjs_list, self.alledges_dict, self.allvps_pos_dict
    
    def merge_feature(self, scan, path, valname_feature="clip", valname_pose="bbox_np"):
        
        import torch
        import torch.nn.functional as F
        
        if self.allobjs_list == []:
            self.load_allobjs_from_hdf5()
            Warning("Loading objs from hdf5 file can be very slow! Shouldn't be used in the mergeing process.")
        all_objects = self.allobjs_list[scan]
    
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
        #key is the vp name, value is the vp position
        
        distance_list = []
        min_distance = None
        
        # Iterate over each center in allvps
        for vp_name, vp_pos in vps_pos_dict.items():
            # Calculate Euclidean distance to determine similarity
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
        
    def find_correspondece_similarity_bboxcenter(self, objs, obj_pos, similarity_threshold=0.2, decimals=5):
        import numpy as np
        """Find correspondence based on the similarity of bbox centers."""
        correspondence_dict = {}
        
        # Calculate centers for each bbox in objs
        objs_centers = [self.calculate_center(obj["bbox_np"]) for obj in objs]
        
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
        
        
class ObjEdegGenerator():
    def __init__(self) -> None:
        pass
    
    def obj_feature_generate(self, rgb, depth):
        
        objs, objs_feature, objs_pose = [], [], []
        
        
        
        
        
        return  objs, objs_feature, objs_pose
    
    def obj_edge_generate(self, objs):
        pass
    
    def CG_processing(
        self,
        scan, 
        vp_list,
        dataset_dictconfig = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/configs/slam_pipeline/base.yaml",
        dataset_root = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R",
        dataset_config = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml"
        ):
        
        path = vp_list
        
        cfg = load_yaml_as_dictconfig(dataset_dictconfig)
        _cfg = {
            "dataset_root": dataset_root,
            "dataset_config": dataset_config,
            "scene_id": scan,
            "spatial_sim_type": "overlap",
            "mask_conf_threshold": 0.25,
            "match_method": "sim_sum",
            "sim_threshold": 1.2,
            "dbscan_eps": 0.1,
            "gsa_variant": "ram_withbg_allclasses",
            "skip_bg": False,
            "max_bbox_area_ratio": 0.5,
            "save_suffix": "overlap_maskconf0.25_simsum1.2_dbscan.1",
            "path": path,
            "save_pcd":False,
        }

        cfg = update_cfg(cfg, _cfg)

        scene_map = MapObjectList(device=cfg.device)

        cfg = process_cfg(cfg)

        # Initialize the dataset
        dataset = get_dataset(
            dataconfig=cfg.dataset_config,
            start=cfg.start,
            end=cfg.end,
            stride=cfg.stride,
            basedir=cfg.dataset_root,
            sequence=cfg.scene_id,
            desired_height=cfg.image_height,
            desired_width=cfg.image_width,
            device="cpu",
            dtype=torch.float,
            trajectory=cfg.path,
        )
        classes, class_colors = create_or_load_colors(cfg, cfg.color_file_name)
        objects = MapObjectList(device=cfg.device)
        
        if not cfg.skip_bg:
            # Handle the background detection separately 
            # Each class of them are fused into the map as a single object
            bg_objects = {
                c: None for c in BG_CLASSES
            }
        else:
            bg_objects = None
            
        # For visualization
        if cfg.vis_render:
            view_param = o3d.io.read_pinhole_camera_parameters(cfg.render_camera_path)
                
            obj_renderer = OnlineObjectRenderer(
                view_param = view_param,
                base_objects = None, 
                gray_map = False,
            )
            frames = []
            
        if cfg.save_objects_all_frames:
            save_all_folder = cfg.dataset_root \
                / cfg.scene_id / "objects_all_frames" / f"{cfg.gsa_variant}_{cfg.save_suffix}"
            os.makedirs(save_all_folder, exist_ok=True)

        for idx in trange(len(dataset)): #trange is meaning of tqdm(range()), tqdm is meaning of progress bar
            # get color image
            color_path = dataset.color_paths[idx]
            image_original_pil = Image.open(color_path)
            ## for VLN
            vp_idx = color_path.split('/')[-1].split('_')[0] ##image name

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

            color_path = Path(color_path)
            detections_path = color_path.parent.parent / cfg.detection_folder_name / color_path.name
            detections_path = detections_path.with_suffix(".pkl.gz")
            color_path = str(color_path)
            detections_path = str(detections_path)
            
            with gzip.open(detections_path, "rb") as f:
                gobs = pickle.load(f)

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
            # Detection is not object 
            
            if len(bg_detection_list) > 0:
                for detected_object in bg_detection_list:
                    class_name = detected_object['class_name'][0]
                    if bg_objects[class_name] is None:
                        bg_objects[class_name] = detected_object
                    else:
                        matched_obj = bg_objects[class_name]
                        matched_det = detected_object
                        bg_objects[class_name] = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
                
            if len(fg_detection_list) == 0:
                continue
                
            if cfg.use_contain_number:
                xyxy = fg_detection_list.get_stacked_values_torch('xyxy', 0)
                contain_numbers = compute_2d_box_contained_batch(xyxy, cfg.contain_area_thresh)
                for i in range(len(fg_detection_list)):
                    fg_detection_list[i]['contain_number'] = [contain_numbers[i]]
                
            if len(objects) == 0:
                # Add all detections to the map
                for i in range(len(fg_detection_list)):
                    objects.append(fg_detection_list[i])

                # Skip the similarity computation 
                continue
                    
            spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
            visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
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
            
            objects = merge_detections_to_objects(cfg, fg_detection_list, objects, agg_sim) # 
            
            # Perform post-processing periodically if told so
            if cfg.denoise_interval > 0 and (idx+1) % cfg.denoise_interval == 0:
                objects = denoise_objects(cfg, objects)
            if cfg.filter_interval > 0 and (idx+1) % cfg.filter_interval == 0:
                objects = filter_objects(cfg, objects)
            if cfg.merge_interval > 0 and (idx+1) % cfg.merge_interval == 0:
                objects = merge_objects(cfg, objects)

            # scenegraph_edges = []
                
            if cfg.save_objects_all_frames:
                save_all_path = save_all_folder / f"{idx:06d}.pkl.gz"
                objects_to_save = MapObjectList([
                    _ for _ in objects if _['num_detections'] >= cfg.obj_min_detections
                ])
                
                objects_to_save = prepare_objects_save_vis(objects_to_save)  #We want to save this!!!!!!
                
                if not cfg.skip_bg:
                    bg_objects_to_save = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
                    bg_objects_to_save = prepare_objects_save_vis(bg_objects_to_save)
                else:
                    bg_objects_to_save = None
                
                result = {
                    "camera_pose": adjusted_pose,
                    "objects": objects_to_save,
                    "bg_objects": bg_objects_to_save,
                }
                with gzip.open(save_all_path, 'wb') as f:
                    pickle.dump(result, f)
            
            if cfg.vis_render:
                objects_vis = MapObjectList([
                    copy.deepcopy(_) for _ in objects if _['num_detections'] >= cfg.obj_min_detections
                ])
                
                if cfg.class_agnostic:
                    objects_vis.color_by_instance()
                else:
                    objects_vis.color_by_most_common_classes(class_colors)
                
                rendered_image, vis = obj_renderer.step(
                    image = image_original_pil,
                    gt_pose = adjusted_pose,
                    new_objects = objects_vis,
                    paint_new_objects=False,
                    return_vis_handle = cfg.debug_render,
                )

                if cfg.debug_render:
                    vis.run()
                    del vis
                
                # Convert to uint8
                if rendered_image is not None:
                    rendered_image = (rendered_image * 255).astype(np.uint8)
                    frames.append(rendered_image)
                
            print(
                f"Finished image {idx} of {len(dataset)}", 
                f"Now we have {len(objects)} objects.",
                f"Effective objects {len([_ for _ in objects if _['num_detections'] >= cfg.obj_min_detections])}"
            )

            
        if bg_objects is not None:
            bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
            bg_objects = denoise_objects(cfg, bg_objects)
            
        objects = denoise_objects(cfg, objects)  
        
        # Save the full point cloud before post-processing
        if cfg.save_pcd:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results = {
                'objects': objects.to_serializable(),
                'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
                'cfg': cfg,
                'class_names': classes,
                'class_colors': class_colors,
            }

            pcd_save_path = cfg.dataset_root / \
                cfg.scene_id / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_{cfg.save_suffix}.pkl.gz"
            # make the directory if it doesn't exist
            pcd_save_path.parent.mkdir(parents=True, exist_ok=True)
            pcd_save_path = str(pcd_save_path)
            
            with gzip.open(pcd_save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved full point cloud to {pcd_save_path}")
        
        objects = filter_objects(cfg, objects)
        objects = merge_objects(cfg, objects)

        scene_map = objects
        
        # Save again the full point cloud after the post-processing
        if cfg.save_pcd:
            results['objects'] = objects.to_serializable()
            pcd_save_path = pcd_save_path[:-7] + "_post.pkl.gz"
            with gzip.open(pcd_save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved full point cloud after post-processing to {pcd_save_path}")
            
        if cfg.save_objects_all_frames:
            save_meta_path = save_all_folder / f"meta.pkl.gz"
            with gzip.open(save_meta_path, "wb") as f:
                pickle.dump({
                    'cfg': cfg,
                    'class_names': classes,
                    'class_colors': class_colors,
                }, f)
            
        if cfg.vis_render:
            # Still render a frame after the post-processing
            objects_vis = MapObjectList([
                _ for _ in objects if _['num_detections'] >= cfg.obj_min_detections
            ])

            if cfg.class_agnostic:
                objects_vis.color_by_instance()
            else:
                objects_vis.color_by_most_common_classes(class_colors)
            
            rendered_image, vis = obj_renderer.step(
                image = image_original_pil,
                gt_pose = adjusted_pose,
                new_objects = objects_vis,
                paint_new_objects=False,
                return_vis_handle = False,
            )
            
            # Convert to uint8
            rendered_image = (rendered_image * 255).astype(np.uint8)
            frames.append(rendered_image)
            
            # Save frames as a mp4 video
            frames = np.stack(frames)
            video_save_path = (
                cfg.dataset_root
                / cfg.scene_id
                / ("objects_mapping-%s-%s.mp4" % (cfg.gsa_variant, cfg.save_suffix))
            )
            imageio.mimwrite(video_save_path, frames, fps=10)
            print("Save video to %s" % video_save_path)
        # print(scene_map)
        return scene_map
        


def load_yaml_as_dictconfig(yaml_file_path: str) -> Dict:
    # Load the YAML file as a DictConfig
    cfg = OmegaConf.load(yaml_file_path)
    return cfg
        