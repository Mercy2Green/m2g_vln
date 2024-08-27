from os import path
import time
import os
from numpy import save
from requests import get
from conceptgraph.slam.cfslam_pipeline_batch import *
from omegaconf import OmegaConf, DictConfig
from typing import Dict
import h5py
import json

import torch.multiprocessing as mp

from progressbar import ProgressBar


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
    mask = path_map_index >= 0

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
    path_indx = paths_indices.index(path)
    # get the path_map_index
    path_map_index = paths_map_indices[path_indx]
    # get the obj_clip_np
    obj_clip, obj_pos = get_scene_map(batch_obj_clips, batch_obj_pos, path_map_index)

    print("#############################################")
    print(f"obj_clip shape: {obj_clip.shape}")
    print(f"obj_pos shape: {obj_pos.shape}")

def get_all_paths4all_scans(json_list):
    all_paths_info = []
    # read each json file in json_list
    for json_file in json_list:
        with open(json_file, 'r') as f:
            for line in f:
                data = json.loads(line)  # 解析每一行为JSON对象
                all_paths_info.append(data)  # 将解析后的数据添加到列表中
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
    with h5py.File(file_path, 'a') as f:
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


def merger_clip4all_combinations(objects: MapObjectList, paths:list, all_vp_ids:list, gpu_device='cuda:0'):

    device = torch.device(gpu_device)

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
        observed_combinations = find_union_of_viewpoints(vp_ids, paths)
        observed_combinations_list.append(observed_combinations) 
        combination_masks = create_merge_masks(vp_ids, observed_combinations)
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
        obj_pos = np.round(obj['bbox'].center,1)
        batch_obj_pos.append(obj_pos)
        #for now
        # clips is 1*m tensor list,
        # masks is n*m tensor list,
        # n is path(combination) number
        # m is vp number
    masks_paded = batch_masks
    clips_paded = batch_clips
    # transfer obj_pos from list to np
    obj_pos_np = np.array(batch_obj_pos)
    
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
    
    tensor_masks_paded = torch.stack(masks_paded).to(device)
    tensor_clips_paded = torch.stack(clips_paded).to(device)
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
    
    # Merge the clips
    if tensor_merged_clips.is_cuda:
        print("Running on the GPU")
    else:
        raise ValueError("The merged clips should be on the GPU")
    
    merged_clips = torch.mean(tensor_merged_clips, dim=1)
    merged_clips = merged_clips.permute(0, 2, 1)
    merged_clips = F.normalize(merged_clips, dim=2)
    # merged_clips is objnum * path(combination)num * feature_dim
    # Every Vp have one object view. So we don't need the weight.

    ## change merged_clips into numpy
    merged_clips_np = merged_clips.cpu().numpy()
    # print the space of merged_clips
    print("#############################################")
    print(f"Space of merged_clips for the whole scence: {merged_clips.nbytes}")
    print("#############################################")

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


#############################################
def load_yaml_as_dictconfig(yaml_file_path: str) -> Dict:
    # Load the YAML file as a DictConfig
    cfg = OmegaConf.load(yaml_file_path)
    return cfg


def update_cfg(cfg: DictConfig, cfg_update: Dict) -> DictConfig:
    # Update cfg with cfg_test
    cfg = OmegaConf.merge(cfg, OmegaConf.create(cfg_update))
    return cfg

def CG_processing(scan, path, device='cpu'):

    dataset_dictconfig = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/configs/slam_pipeline/base.yaml"
    dataset_root = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R"
    dataset_config = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml"
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
        "device": device,
    }
    cfg = update_cfg(cfg, _cfg)
    cfg = process_cfg(cfg)

    scene_map = MapObjectList(device=device)
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

        color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]

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

        # There is problem!!!!!!!!!!!!!!!!!!!!!!
        if cfg.use_contain_number:
            print(1)
            xyxy = fg_detection_list.get_stacked_values_torch('xyxy', 0)
            print(1)
            contain_numbers = compute_2d_box_contained_batch(xyxy, cfg.contain_area_thresh)
            print(1)
            for i in range(len(fg_detection_list)):
                print(1)
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

    return scene_map

def generate_obj_hdf5(proc_id, scans_list, all_trajectory_info, out_queue, device='cuda:0'):

    print("Start at process: ", proc_id, "with device: ", device)

    paths = []
    empty_scans = []
    start = time.time()
    i = 0
    for scan in scans_list:
        # get paths for a given scan
        paths = get_paths4scan(scan, all_trajectory_info)
        # get unique viewpoints in paths
        unique_viewpoints = get_unique_viewpoints(paths)

        # generate all possible subpaths
        all_paths = generate_subpaths(paths)
        print("#############################################")
        print(f"length of unique_viewpoints: {len(unique_viewpoints)} for scan {scan}")
        print("#############################################")
        if len(unique_viewpoints) == 0:
            print(f"{scan} is empty!")
            empty_scans.append(scan)
            continue
        object_list = CG_processing(scan, unique_viewpoints[:5], device)
        end_time_1 = time.time()
        # print the time in seconds
        print("#############################################")
        print(f"Time for first CG_processing: {end_time_1 - start}")
        print("#############################################")

        # merge all observed combinations
        merged_clips_np, batch_clip_masks, paths_masks, batch_obj_pos = merger_clip4all_combinations(object_list, all_paths, unique_viewpoints, device)

        # get best observed combination
        best_combination_indices = get_best_obs_combination(batch_clip_masks, paths_masks,)

        # save merged_clips_np and best_combination_indices
        # save_path_ObjList2hdf5(scan, merged_clips_np, best_combination_indices, batch_obj_pos, hdf5_path)
        # save_path_list2json(scan, all_paths, json_path)
        out_queue.put((scan, merged_clips_np, best_combination_indices, batch_obj_pos, all_paths))

    out_queue.put(None)


def multi_process_generate_obj_hdf5(scans_txt_path, json_dir, hdf5_path, json_path, n_gpus):

    with open(scans_txt_path, 'r') as file:
        scans_list = [line.strip() for line in file]

    all_trajectory_info = get_all_paths4all_scans(json_dir)

    num_gpus = min(n_gpus, len(scans_list))
    num_data_per_gpu = len(scans_list) // num_gpus

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_gpus):
        sidx = proc_id * num_data_per_gpu
        eidx = None if proc_id == num_gpus - 1 else sidx + num_data_per_gpu
        device = "cuda:" + str(proc_id % torch.cuda.device_count())

        process = mp.Process(
            target=generate_obj_hdf5,
            args=(proc_id, scans_list[sidx: eidx], all_trajectory_info, out_queue, device)
        )
        process.start()
        processes.append(process)
    
    num_finished_gpus = 0
    num_finished_scans = 0

    progress_bar = ProgressBar(max_value=len(scans_list))
    progress_bar.start()

    while num_finished_gpus < num_gpus:
        res = out_queue.get()
        if res is None:
            num_finished_gpus += 1
        else:
            scan, merged_clips_np, best_combination_indices, batch_obj_pos, all_paths = res
            save_path_ObjList2hdf5(scan, merged_clips_np, best_combination_indices, batch_obj_pos, hdf5_path)
            save_path_list2json(scan, all_paths, json_path)
            num_finished_scans += 1
            progress_bar.update(num_finished_scans)

    progress_bar.finish()
    for process in processes:
        process.join()

if __name__ == "__main__":
    
    scans_txt_path = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity/scans.txt"
    json_dir =["/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_train_enc.jsonl",
            "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_prevalent_aug_train_enc.jsonl",
            "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_seen_enc.jsonl",
            "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_unseen_enc.jsonl"]

    hdf5_path = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/" + "eval_cgobj_clip.hdf5"
    json_path = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/" + "eval_all_paths.json"

    multi_process_generate_obj_hdf5(scans_txt_path, json_dir, hdf5_path, json_path, n_gpus=1)
