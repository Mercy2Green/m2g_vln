from os import path
from re import S
import time

from numpy import save
from pytest import Parser
from requests import get
from conceptgraph.slam.cfslam_pipeline_batch import *
from omegaconf import OmegaConf, DictConfig
from typing import Dict

import h5py
import json

import sys

def get_scan_list_from_folder(folder_path):
    scan_list = []
    for filename in os.listdir(folder_path):
        # 分割文件名
        parts = filename.split('_')
        if parts:  # 确保列表不为空
            scan = parts[0]
            scan_list.append(scan)
    return scan_list

def get_scene_map(batch_obj_clip_np, batch_obj_pos_np, path_map_index):

    obj_indices = np.arange(batch_obj_clip_np.shape[0])
    valid_obs_combination = np.where(path_map_index >= 0, path_map_index, 0).astype(int)
    selected_obj_clip = batch_obj_clip_np[obj_indices, valid_obs_combination,:]
    mask = path_map_index > 0

    obj_clip = selected_obj_clip[mask]
    obj_pos = batch_obj_pos_np[mask]
    return obj_clip, obj_pos


# test the saved file can be used correctely
def test_saving_data(scan, path, hdf5_path, json_path):
    # read the obj_clip from hdf5
    with h5py.File(hdf5_path, 'r') as f:
        batch_obj_clips = f[f'{scan}_obj_clip_ft'][...].astype(np.float32)
        paths_map_indices = f[f'{scan}_paths_indices'][...].astype(np.float32)
        batch_obj_pos = f[f'{scan}_obj_pos'][...].astype(np.float32)
    
    # read the paths_indices from json
    with open(json_path, 'r') as f:
        paths_indices = json.load(f)

    # get path index from a list by input the element
    path_indx = paths_indices[scan].index(path)
    # get the path_map_index
    path_map_index = paths_map_indices[path_indx]
    # get the obj_clip_np
    obj_clip, obj_pos = get_scene_map(batch_obj_clips, batch_obj_pos, path_map_index)

    print("#############################################")
    print(f"obj_clip shape: {obj_clip.shape}")
    print(f"obj_pos shape: {obj_pos.shape}")

# get all paths info from json files
import json

def get_all_paths4all_scans(json_list):
    all_paths_info = []
    # read each json file in json_list
    for json_file in json_list:
        with open(json_file, 'r') as f:
            for line in f:
            # data = json.load(f)
                data = json.loads(line)  # 解析每一行为JSON对象
            # for item in data:
            #     if "instructions" in item:
            #         print(" there is instructions in the json file !!!!!!!!!!!!!!")
                all_paths_info.append(data)  # 将解析后的数据添加到列表中
    return all_paths_info

def get_all_paths4test_scans(json_list):
    all_paths_info = []
    # Read the entire JSON file
    for json_file in json_list:
        with open(json_file, 'r') as f:
            data = json.load(f)  # Load the entire file content as a JSON object
            for item in data:
                all_paths_info.append(item)  # Add each item to the list
    return all_paths_info

# get paths info given the scan
def get_paths4scan(scan, all_paths_info):
    paths = []
    for path_info in all_paths_info:
        if path_info.get('scan') == scan:
            paths.append(path_info.get('path'))
    return paths

def find_paths_by_scan(scan_id, file_path):
    paths = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if data['scan'] == scan_id:
                    paths.append(data['path'])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Could not decode JSON from file: {file_path}")
        return None

    return paths

def generate_subpaths(paths):
    # 创建一个新列表以避免修改原始列表
    # 简单版本，顺序循环
    all_paths = list(paths)
    
    # 遍历原始路径列表
    for path in paths:
        # 生成并添加所有可能的子路径
        for i in range(1, len(path)):
            subpath = path[:i]
            if subpath not in all_paths:  # 避免添加重复的子路径
                all_paths.append(subpath)

    return all_paths


def get_unique_viewpoints(paths):
    # 使用集合来去重并获取所有独特的视点
    unique_viewpoints = set()
    for path in paths:
        unique_viewpoints.update(path)  # 将每个path中的视点添加到集合中，自动去重

    # convert unique_viewpoints to list
    unique_viewpoints = list(unique_viewpoints)
    
    return unique_viewpoints

### for VLN
# save path list into a json file
import os
import json

def save_path_list2json(scan, paths, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = {}
    # 尝试读取现有数据
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as f:
            data = json.load(f)
    
    # 更新或添加新的scan数据
    data[scan] = paths
    
    # 写回文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Save all paths for {scan} to {file_path} successfully!")

# save hdf5 file
def save_path_ObjList2hdf5(scan, obj_clips, paths_indices, batch_obj_pos,file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, 'w') as f:
        clip_np = f.create_dataset(f'{scan}_obj_clip_ft', data=obj_clips)
        #save the paths_indices
        paths_indices_np = f.create_dataset(f'{scan}_paths_indices', data=paths_indices)
        # save the pos infomation
        obj_pos_np = f.create_dataset(f'{scan}_obj_pos', data=batch_obj_pos)
    
    print(f"Save obj_clips and paths_indices to {file_path} successfully!")

# get path indices
def get_best_obs_combination(batch_obj_obs_mask, paths_mask):
    '''
    batch_obj_obs_mask: tensor shape (n_objects, n_obs_combinations, n_vp)
    paths_mask: tensor shape (n_paths, n_vp)

    Returns:
        A tensor of shape (n_paths, n_objects) indicating the index of the observation
        combination with the maximum observations that is a subset of each path.
    '''
    # Expand dimensions to match for broadcasting
    batch_obj_obs_mask = batch_obj_obs_mask.unsqueeze(0)  # (1, n_objects, n_obs_combinations, n_vp)
    paths_mask = paths_mask.unsqueeze(1).unsqueeze(1)  # (n_paths, 1, 1, n_vp)

    # Check if each observation combination is a subset of each path
    is_subset = (batch_obj_obs_mask <= paths_mask).all(dim=3)  # Shape: (n_paths, n_objects, n_obs_combinations)
    # is_subset_1 = torch.all((batch_obj_obs_mask & paths_mask) == batch_obj_obs_mask, dim=3)
    # print(is_subset)
    # print(is_subset_1)

    # Calculate the number of observations in each combination
    obs_counts = batch_obj_obs_mask.sum(dim=3)  # Shape: (1, n_objects, n_obs_combinations)

    # Use the subset mask to mask out combinations that are not subsets
    valid_obs_counts = obs_counts * is_subset.float()  # Shape: (n_paths, n_objects, n_obs_combinations)


    valid_obs_counts[~is_subset] = -1

    # # Find the index of the combination with the maximum observations for each path and object
    # best_combination_indices = valid_obs_counts.argmax(dim=2).squeeze(dim=-1)  # Shape: (n_paths, n_objects)
    # 找到最大观测组合的索引
    best_combination_indices = valid_obs_counts.argmax(dim=2)

    # 检查并调整全为 -1 的情况
    all_invalid = (valid_obs_counts == -1).all(dim=2)
    best_combination_indices[all_invalid] = -1

    #
    return best_combination_indices

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def merger_clip4all_combinations(objects: MapObjectList, paths:list, all_vp_ids:list, device='cuda'):
    '''
    given all possible paths, which provide all obserbation combination for all object
    '''
    batch_masks = []
    batch_clips = []
    
    max_combinations = 0
    max_vps = 0
    
    observed_combinations_list = []  # Store the results of find_union_of_viewpoints
    
    batch_clip_masks = []
    batch_obj_pos = []
    for obj in objects:
        observed_info = obj.get('observed_info')
        if observed_info is None:
            continue
        
        vp_ids = list(observed_info.keys())
        
        # This is the list of all possible combinations of viewpoints
        observed_combinations = find_union_of_viewpoints(vp_ids, paths)
        # observed_combinations = get_all_length_combinations(vp_ids)
        
        # observed_combinations_list.append(observed_combinations) 
        combination_masks = create_merge_masks(vp_ids, observed_combinations)
        if combination_masks is None:
            continue
        vp_clips = [observed_info[vp_id] for vp_id in vp_ids]
        stacked_vp_clips = torch.stack(vp_clips)
        
        batch_masks.append(combination_masks)
        batch_clips.append(stacked_vp_clips)
        
        max_combinations = max(max_combinations, len(observed_combinations))
        max_vps = max(max_vps, len(vp_ids))
        ## create masks for saving the whole information
        clip_masks = create_merge_masks(all_vp_ids, observed_combinations)
        batch_clip_masks.append(clip_masks)

        # create the position information
        obj_pos = np.round(obj['bbox'].center,5)
        batch_obj_pos.append(obj_pos)
        #for now
        # clips is 1*m tensor list,
        # masks is n*m tensor list,
        # n is path(combination) number
        # m is vp number
    masks_paded = batch_masks
    clips_paded = batch_clips
    # transfer obj_pos from list to np
    # obj_pos_np = np.array(batch_obj_pos)
    
    # Pad the masks and clips to have the same shape
    batch_clip_masks = pad_tensors(batch_clip_masks)
    paths_masks = create_merge_masks(all_vp_ids, paths)
    # for i in range(len(batch_masks)):
    #     max_combinations = max(max_combinations, batch_masks[i].shape[0])
    #     max_vps = max(max_vps, batch_clips[i].shape[0])

    # Pad the masks and clips to have the same shape
    for i in range(len(batch_masks)):
        combination_pad_size = max_combinations - batch_masks[i].shape[0]
        vp_pad_size = max_vps - batch_clips[i].shape[0]
        masks_paded[i] = F.pad(batch_masks[i], (0, 0, 0, vp_pad_size, 0, combination_pad_size))
        # masks_paded[i] = F.pad(masks_paded[i], (0, 0, 0, 0, 0, vp_pad_size))
        clips_paded[i] = F.pad(batch_clips[i], (0, 0, 0, vp_pad_size))
        # print(masks_paded[i].shape)
    
    
    tensor_masks_paded = torch.stack(masks_paded)
    tensor_clips_paded = torch.stack(clips_paded)
    
    # To device
    tensor_masks_paded = tensor_masks_paded.to(device)
    tensor_clips_paded = tensor_clips_paded.to(device)
    # tensor_masks_paded is objnum * path(combination)num * vpnum * 1
    # tensor_clips_paded is objnum * vpnum * feature_dim
    # For example :
    # tensor_masks_paded is 16 5 3 1
    # tensor_clips_paded is 16 3 1024

    tensor_masks_paded = tensor_masks_paded.squeeze(-1).unsqueeze(2).permute(0,3,2,1)
    #tensor_masks_paded is objnum * vpnum * 1 * path(combination)num
    tensor_clips_paded = tensor_clips_paded.unsqueeze(3)
    #tensor_clips_paded is objnum * vpnum * feature_dim * 1
    tensor_merged_clips = tensor_masks_paded * tensor_clips_paded
    # tensor_merged_clips is objnum * vpnum * feature_dim * path(combination)num

    # obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
    #                    obj2['clip_ft'] * n_obj2_det) / (
    #                    n_obj1_det + n_obj2_det)
    # obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)
    
    # Merge the clips
    merged_clips = torch.mean(tensor_merged_clips, dim=1)
    merged_clips = merged_clips.permute(0, 2, 1)
    merged_clips = F.normalize(merged_clips, dim=2)
    # merged_clips is objnum * path(combination)num * feature_dim
    # Every Vp have one object view. So we don't need the weight.

    ## change merged_clips into numpy
    merged_clips_np = merged_clips.cpu().numpy()
    # print the space of merged_clips
    # print("#############################################")
    # print(f"Space of merged_clips for the whole scence: {merged_clips.nbytes}")
    # print("#############################################")
    # # Store the merged clips in the MapObjectList
    # for i, obj in enumerate(objects):
    #     obj['multi_clip'] = {}
    #     observed_info = obj.get('observed_info')
    #     if observed_info is None:
    #         continue
    #     vp_ids = list(observed_info.keys())
    #     observed_combinations = observed_combinations_list[i]  # Retrieve the results
    #     for j, combination in enumerate(observed_combinations):
    #         obj['multi_clip'][combination] = merged_clips[i, j]

    return merged_clips_np, batch_clip_masks, paths_masks, batch_obj_pos

def save_multi_clip_to_hdf5(objects: MapObjectList, filename: str):
    with h5py.File(filename, 'w') as f:
        for i, obj in enumerate(objects):
            multi_clip = obj.get('multi_clip')
            if multi_clip is None:
                continue
            for combination, clip in multi_clip.items():
                # Create a group for each object and combination
                group = f.create_group(f'object_{i}/combination_{combination}')
                # Save the clip to the group
                group.create_dataset('clip', data=clip.cpu().numpy())

def create_merge_masks(vp_ids, observed_combinations):

    combination_masks = []
    for combination in observed_combinations:
        vp_masks = []
        for vp_id in vp_ids:
            if vp_id in combination:
                # 将当前视点的掩码添加到列表中
                vp_masks.append(torch.ones(1))
            else:
                # 否则，将一个全零的掩码添加到列表中
                vp_masks.append(torch.zeros(1))
        # 将当前组合的掩码堆叠起来
        combination_mask = torch.stack(vp_masks)
        # 将当前组合的掩码添加到列表中
        combination_masks.append(combination_mask)
    # 将所有组合的掩码堆叠起来，返回一个张量
    # checkout if combination_masks is empty
    if len(combination_masks) == 0:
        return None
    return torch.stack(combination_masks)


def find_union_of_viewpoints(viewpoints_a, paths):
    union_result = set()
    intersections = [] 
    for path in paths:
        intersection = find_subsequences(viewpoints_a, path)
        if list(intersection) != []:
            intersections.append(intersection)
    # # get the union of all intersections
    for intersection in intersections:
        union_result.update(intersection)
    return union_result

def get_all_sublists(lst):
    sublists = []
    for i in range(len(lst)):
        for j in range(i+1, len(lst)+1):
            sublists.append(lst[i:j])
    return sublists

def get_all_length_combinations(lst):
    from itertools import permutations
    def has_consecutive_duplicates(sequence):
        return any(x == y for x, y in zip(sequence, sequence[1:]))

    all_combinations = set()
    for r in range(1, len(lst) + 1):
        for subset in permutations(lst, r):
            if not has_consecutive_duplicates(subset):
                all_combinations.add(tuple(subset))
    return all_combinations

def find_subsequences(lst, target_lst):
    # lst is the list to search in the target_lst. 
    sublists = get_all_sublists(lst)
    subsequences = set()

    for subl in sublists:
        length = len(subl)
        for i in range(len(target_lst)-length+1):
            sub_target_lst = tuple(target_lst[i:i+length])
            if sub_target_lst == tuple(subl):
                subsequences.add(sub_target_lst)
    return subsequences

def load_yaml_as_dictconfig(yaml_file_path: str) -> Dict:
    # Load the YAML file as a DictConfig
    cfg = OmegaConf.load(yaml_file_path)
    return cfg


def update_cfg(cfg: DictConfig, cfg_update: Dict) -> DictConfig:
    # Update cfg with cfg_test
    cfg = OmegaConf.merge(cfg, OmegaConf.create(cfg_update))
    return cfg


# @hydra.main(version_base=None, config_path="../configs/slam_pipeline", config_name="base")
def CG_processing(
    scan,
    path,
    dataset_dictconfig = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/configs/slam_pipeline/base.yaml",
    dataset_root = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R",
    dataset_config = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml"
    ):

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
        "save_pcd":True,
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
def get_scans_in_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    scans = data.keys()
    print(f"all processed scans: {scans}")
    return scans
def load_result(result_path):
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    
    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])
        
        if results['bg_objects'] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])

        class_colors = results['class_colors']
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)

        bg_objects = None
        class_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)
        class_colors = {str(i): c for i, c in enumerate(class_colors)}
    else:
        raise ValueError("Unknown results type: ", type(results))
        
    return objects, bg_objects, class_colors

def get_vis_scene_map(whole_scene_map, vis_scene_map, batch_obj_pos):
    for pos in batch_obj_pos:
        found = False  # 初始化标志变量

        for obj in whole_scene_map:
            obj_pos = np.round(obj['bbox'].center, 5)
            dist = np.linalg.norm(obj_pos - pos)
            if dist < 0.1:
                vis_scene_map.append(obj)
                found = True  # 找到对象，设置标志变量为True
                # print when find the object
                print(f"find the object in front ground with distance {dist}!!!!!!!!!!!!!!!!!")
                break  # 找到对象后跳出内层循环
        if not found:
            # 如果在whole_scene_map中没有找到对象
            print(f"not found in fg with distance with pose {pos}!!!!!!!!!!!!")
    
    
def load_scene_map_here(args, scene_map):
    """
    Loads a scene map from a gzip-compressed pickle file. This is a function because depending whether the mapfile was made using cfslam_pipeline_batch.py or merge_duplicate_objects.py, the file format is different (see below). So this function handles that case.
    
    The function checks the structure of the deserialized object to determine
    the correct way to load it into the `scene_map` object. There are two
    expected formats:
    1. A dictionary containing an "objects" key.
    2. A list or a dictionary (replace with your expected type).
    """
    import pickle as pkl
    with gzip.open(Path(args.scene_map_file), "rb") as f:
        loaded_data = pkl.load(f)
        
        # Check the type of the loaded data to decide how to proceed
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            scene_map.load_serializable(loaded_data["objects"])
        elif isinstance(loaded_data, list) or isinstance(loaded_data, dict):  # Replace with your expected type
            scene_map.load_serializable(loaded_data)
        else:
            raise ValueError("Unexpected data format in map file.")
        print(f"Loaded {len(scene_map)} objects")
def transfer_vp_pos(pos):
    """
    pos ins a  4x4 matrix in row major order in [float x 16]
    we only want to get the coordinates in [x, y, z]
    """
    # original (x,y,z)
    # x = pos[3]
    # y = pos[7]
    # z =  pos[11]
    # conceptgraph
    x = pos[3]
    y = - (pos[11] - 1.25)
    z = pos[7]
    return [x,y,z]
    
def get_vp_coordinates(path, scan):
    """
    path is a list of viewpoints in the path
    """
    connection_files = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity"
    with open(connection_files + f"/{scan}_connectivity.json") as f:
        data = json.load(f)
    vps_pos_list = []
    for vp in path:
        for vp_info in data:
            if vp_info["image_id"] == vp:
                vp_pos_mat = vp_info["pose"]
                vp_pos = transfer_vp_pos(vp_pos_mat)
                vps_pos_list.append(vp_pos)
                break
    return vps_pos_list
def transfrom3D(xyzhe):
    '''
    Return (N, 4, 4) transformation matrices from (N,5) x,y,z,heading,elevation 
    '''
    theta_x = xyzhe[:,4] # elevation
    cx = np.cos(theta_x)
    sx = np.sin(theta_x)

    theta_y = xyzhe[:,3] # heading
    cy = np.cos(theta_y)
    sy = np.sin(theta_y)

    T = np.zeros([xyzhe.shape[0], 4, 4])
    T[:,0,0] =  cy
    T[:,0,1] =  sx*sy
    T[:,0,2] =  cx*sy 
    T[:,0,3] =  xyzhe[:,0] # x

    T[:,1,0] =  0
    T[:,1,1] =  cx
    T[:,1,2] =  -sx
    T[:,1,3] =  xyzhe[:,1] # y

    T[:,2,0] =  -sy
    T[:,2,1] =  cy*sx
    T[:,2,2] =  cy*cx
    T[:,2,3] =  xyzhe[:,2] # z

    T[:,3,3] =  1
    return T.astype(np.float32)

def transform_pcd_bbox(filtered_objs, vps_pos, last_vp_pos, scan, path):
    '''
    this function try to make some transformation test 
    '''
    import math
    # get S_w2c
    S_w2c = np.zeros(3).astype(np.float32)
    # S_w2c[0] = last_vp_pos[0]
    # S_w2c[1] = -last_vp_pos[1] + 1.25
    # S_w2c[2] = last_vp_pos[1]
    #
    S_w2c[0] = last_vp_pos[0]
    S_w2c[1] = last_vp_pos[1]
    S_w2c[2] = last_vp_pos[2]
    # get T_w2c
    scanvp_cands_file = "/home/lg1/lujia/VLN_HGT/pretrain_src/datasets/R2R/annotations/scanvp_candview_relangles.json"
    scanvp_cands = json.load(open(scanvp_cands_file))
    viewidx = scanvp_cands['%s_%s'%(scan, path[-2])][path[-1]][0]
    cur_heading = (viewidx % 12) * math.radians(30)
    xyzhe = np.zeros([1,5]).astype(np.float32)
    xyzhe[:,3] =  - cur_heading
    xyzhe[:,4] = 0
    T_w2c = transfrom3D(xyzhe)
    T_w2c = np.squeeze(T_w2c, axis=0)
    # R_1 = np.array([[1,0,0],
    #                 [0,0,1],
    #                 [0,1,0]]).astype(np.float32)
    # S_2 = np.array([0,-1.25,0]).astype(np.float32)
    # R_3 = np.array([[1,0,0],
    #                 [0,-1,0],
    #                 [0,0,1]]).astype(np.float32)
    # transform the pcd and bbox
    for obj in filtered_objs:
        pcd = obj['pcd']
        bbox = obj['bbox']
        # minor the S_w2c
        pcd.translate(-S_w2c)
        bbox.translate(-S_w2c)
        # transform the pcd and bbox with T_w2c
        # pcd.rotate(T_w2c[:3,:3])
        # bbox.rotate(T_w2c[:3,:3])
        
        # CG TO MAT
        # pcd.rotate(R_1)
        # pcd.translate(S_2)
        # pcd.rotate(R_3)
        # bbox.rotate(R_1)
        # bbox.translate(S_2)
        # bbox.rotate(R_3)
    # print("transform the pcd and bbox successfully!")
    # transform for vps
    # S_w2c_vp = np.zeros([10,10,10]).astype(np.float32)
    # S_w2c_vp[0] = last_vp_pos[0]
    # S_w2c_vp[1] = last_vp_pos[1]
    # S_w2c_vp[2] = last_vp_pos[2]
    for i, vp_pos in enumerate(vps_pos):
        vp_pos_array = np.array(vp_pos)
        # vp_pos[0] = vp_pos[0] - S_w2c_vp
        vp_pos_array = vp_pos_array - S_w2c
        vps_pos[i] = vp_pos_array.tolist()
    print("transform the vps successfully!")
    return filtered_objs, vps_pos
    
    
    
def save_scene_map(scan, path, all_vps):
    # get the object list from a whole scene
    scene_map_folder = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R"
    scene_map_file = scene_map_folder + f"/{scan}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz"
    save_folder = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data"
    save_path = save_folder + f"/{scan}_visual_cg.pkl.gz"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_map_file", type=str, default=scene_map_file)
    parser.add_argument("--savefile", type=str, default=save_path)
    args = parser.parse_args()
    # from concept-graphs.conceptgraph.scenegraph.build_scenegraph_cfslam import load_scene_map
    # whole_scene_map = MapObjectList()
    print(f"Loading scene map {scan} ...")
    # load_scene_map_here(args,whole_scene_map)
    whole_scene_map, bg_objects, class_colors = load_result(args.scene_map_file)
    print(f"Loaded {len(whole_scene_map)} objects")
    searching_path = [path]
    # processing the object searching 
    merged_clips_np, batch_clip_masks, paths_masks, batch_obj_pos = merger_clip4all_combinations(whole_scene_map, searching_path, all_vps)
    ## 
    vis_fg = MapObjectList()
    vis_bg = MapObjectList()
    print("Start searching the object in front ground...")
    get_vis_scene_map(whole_scene_map, vis_fg, batch_obj_pos)
    # print("Start searching the object in back ground...")
    # get_vis_scene_map(bg_objects, vis_bg, batch_obj_pos)
    # get viewpoint coordinates
    vps_coordinates = get_vp_coordinates(path, scan)
    # transform the pcd and bbox of objects
    # vis_fg, vps_coordinates = transform_pcd_bbox(vis_fg, vps_coordinates, vps_coordinates[-1],scan,path)
    # concate the result
    vis_result = {
        "objects": vis_fg.to_serializable(),
        "bg_objects": None if vis_bg is None else vis_bg.to_serializable(),
        "class_colors": class_colors,
        "viewpoints": vps_coordinates,
    }
    
    # save the vis_scene_map

    if not os.path.exists(os.path.dirname(args.savefile)):
        os.makedirs(os.path.dirname(args.savefile), exist_ok=True)
    with gzip.open(args.savefile, "wb") as f:
        pickle.dump(vis_result, f)
        print(f"Save vis_scene_map to {args.savefile} successfully!")
    assert batch_obj_pos is not None, "batch_obj_pos should be None"
    return batch_obj_pos
    
def print_instruction(json_dir, path, scan):
    data = []
    for file in json_dir:
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        for path_info in data:
            if path_info["scan"] == scan:
                if path == path_info["path"]:
                    instructions = path_info["instr_id"]
                    print(f" the instruction is {instructions}")
                    
                    
def cg_process_and_save_files(save_data_root, test_flag = False):

    test_scans_txt_path = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity/eval_scans.txt"
    scans_txt_path = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity/scans.txt"
    
    if test_flag:
        scans_txt_path = test_scans_txt_path
    
    with open(scans_txt_path, 'r') as file:
        scans_list = [line.strip() for line in file]

    # get all paths data
    json_dir =["/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_train_enc.jsonl",
            "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_prevalent_aug_train_enc.jsonl",
            "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_seen_enc.jsonl",
            "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_unseen_enc.jsonl"]
    
    test_json_dir = ["/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/annotations/R2R_test_enc.json"]

    paths = []
    
    # read all the paths json files
    if test_flag:
        all_trajectory_info = get_all_paths4test_scans(test_json_dir)
    else:
        all_trajectory_info = get_all_paths4all_scans(json_dir)
    
    

    # Save file names
    test_hdf5_save_name = "cgobj_clip_18scans.hdf5"
    test_json_save_name = "all_paths_18scans.json"
    
    if test_flag:
        hdf5_path = save_data_root + '/' + test_hdf5_save_name
        json_path = save_data_root + '/' + test_json_save_name
    else:  
        hdf5_path = save_data_root + '/' + "cgobj_clip_90scans.hdf5"
        json_path = save_data_root + '/' + "all_paths_90scans.json"

    for scan in tqdm(scans_list):

        # get paths for a given scan
        paths = get_paths4scan(scan, all_trajectory_info)
        # get unique viewpoints in paths
        unique_viewpoints = get_unique_viewpoints(paths)

        # generate all possible subpaths
        all_paths = generate_subpaths(paths)

        # statistic running time for cg_processing
        object_list = CG_processing(scan, unique_viewpoints)
        
        # merge all observed combinations
        merged_clips_np, batch_clip_masks, paths_masks, batch_obj_pos = merger_clip4all_combinations(object_list, all_paths,unique_viewpoints)

        # get best observed combination
        best_combination_indices = get_best_obs_combination(batch_clip_masks, paths_masks,)

        # save merged_clips_np and best_combination_indices
        save_path_ObjList2hdf5(scan, merged_clips_np, best_combination_indices, batch_obj_pos,hdf5_path)
        # save_path_list2json(all_paths, json_path)
        save_path_list2json(scan, all_paths, json_path)

    # python scripts/visualize_cfslam_results.py --result_path /media/m2g/Data/Datasets/m2g_vln_server/m2g_vln/VLN-BEVBert/img_features/17DRP5sb8fy/pcd_saves

    
def main_old():
    print("Start processing...")
    # scans_path = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity"
    # scans_list = get_scan_list_from_folder(scans_path)
    
    scans_txt_path = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity/scans.txt"
    
    with open(scans_txt_path, 'r') as file:
        scans_list = [line.strip() for line in file]

    # get all paths data
    json_dir =["/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_train_enc.jsonl",
            "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_prevalent_aug_train_enc.jsonl",
            "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_seen_enc.jsonl",
            "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_unseen_enc.jsonl"]

    paths = []
    
    # read all the paths json files
    all_trajectory_info = get_all_paths4all_scans(json_dir)
    
    # # For test
    # scan='ZMojNkEp431'
    # paths = get_paths4scan(scan, all_trajectory_info)
    # print(f" get all the paths for scan {scan}, the length of paths is {len(paths)}")
    # # all_paths = generate_subpaths(paths)
    # unique_viewpoints = get_unique_viewpoints(paths)
    # searaching_path = paths[21]
    ### 
    
    # # get the visualization data
    # save_scene_map(scan, searaching_path, unique_viewpoints)
    # print_instruction(json_dir, searaching_path, scan)
    

    save_data_root = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/test"
    
    # read pkl file into MapObjectList
    
    
    # This is the 90 scans save path.
    hdf5_path = save_data_root + "cgobj_clip_90scans.hdf5"
    json_path = save_data_root + "all_paths_90scans.json"
    
    

    # test_saving_data(scan, all_paths[0], hdf5_path, json_path)

    # paths = find_paths_by_scan(scan, json_dir[0])
    # for file in json_dir:
    #     _path = find_paths_by_scan(scan, file)
    #     paths.extend(_path)

    empty_scans = []
    time_stats = {
        'total_start': time.time(),
        'loop_times': []
    }

    start = time.time()
    i = 0
    # get all the scan in cureent json
    # all_paths_json_path = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/" + "all_paths_90scans.json"
    # processed_scans = get_scans_in_json(all_paths_json_path)
    # scans_list = ['zsNo4HB9uLZ']
    for scan in scans_list:
        # scan='zsNo4HB9uLZ'
        print(f"Processing scan {scan}...")
        # if scan is in the processed scans, skip it
        # if scan in processed_scans:
        #     print(f"{scan} has been processed! or having not data")
        #     i = i+1
        #     continue
        # print(f"these scans {scan} need to be processed")
        loop_start = time.time()  # 循环开始时间
        loop_stats = {'start': loop_start}

        # get paths for a given scan
        paths = get_paths4scan(scan, all_trajectory_info)
        # get unique viewpoints in paths
        unique_viewpoints = get_unique_viewpoints(paths)
        
        #This is ONLY FOR TEST
        ###FOR TESTTTTTTTT!!!!!!!!
        # unique_viewpoints = unique_viewpoints[0:5]

        # generate all possible subpaths
        all_paths = generate_subpaths(paths)
        print("#############################################")
        print(f"length of unique_viewpoints: {len(unique_viewpoints)} for scan {scan}")
        print("#############################################")
        if len(unique_viewpoints) == 0:
            print(f"{scan} is empty!")
            empty_scans.append(scan)
            continue

        # statistic running time for cg_processing
        object_list = CG_processing(scan, unique_viewpoints)
        end_time_1 = time.time()
        # print the time in seconds
        print("#############################################")
        print(f"Time for first CG_processing: {end_time_1 - start}")
        print("#############################################")

        end_time_1_1 = time.time()
        # merge all observed combinations
        merged_clips_np, batch_clip_masks, paths_masks, batch_obj_pos = merger_clip4all_combinations(object_list, all_paths,unique_viewpoints)

        # get best observed combination
        best_combination_indices = get_best_obs_combination(batch_clip_masks, paths_masks,)

        # save merged_clips_np and best_combination_indices
        save_path_ObjList2hdf5(scan, merged_clips_np, best_combination_indices, batch_obj_pos,hdf5_path)
        # save_path_list2json(all_paths, json_path)
        save_path_list2json(scan, all_paths, json_path)
        
        # 记录循环结束时间并计算总耗时
        loop_end = time.time()
        loop_stats['end'] = loop_end
        loop_stats['total'] = (loop_end - loop_start) / 3600  # 转换为小时
        time_stats['loop_times'].append(loop_stats)

        print(f"Loop {i}:")
        i = i+1
        print(f"Total time: {loop_stats['total']} hours")
        # print in minutes
        print(f"Total time: {loop_stats['total'] * 60} minutes")
        print(f"finish {scan} processing")
        break

    print(f"there are {len(empty_scans)} empty scans!")
    # 计算并打印总时间
    time_stats['total_end'] = time.time()
    time_stats['total_time'] = (time_stats['total_end'] - time_stats['total_start']) / 3600  # 转换为小时

    # 打印每次循环的时间统计
    for i, loop_time in enumerate(time_stats['loop_times']):
        print(f"Loop {i}:")
        for key, value in loop_time.items():
            if key != 'start' and key != 'end':
                print(f"  {key}: {value:.2f} hours")
        print(f"  Total: {loop_time['total']:.2f} hours")
        print("#############################################")

    # 打印总时间
    print(f"Total time: {time_stats['total_time']:.2f} hours")
    print("Done!")
    # print("#############################################")
    print("start testing the saved file")
    # test_saving_data(scan, all_paths[0], hdf5_path, json_path)
    # print scene_map all keys

    # python scripts/visualize_cfslam_results.py --result_path /media/m2g/Data/Datasets/m2g_vln_server/m2g_vln/VLN-BEVBert/img_features/17DRP5sb8fy/pcd_saves

if __name__ == "__main__":
    
    save_data_root = "/data2/vln_dataset/test_m2g_processed"
   
    cg_process_and_save_files(save_data_root, test_flag = True) # This is for generate test data.