import numpy as np
from conceptgraph.slam.cfslam_pipeline_batch import *
from torch import gt
from cg_processing import *

def get_scene_map_etp(batch_obj_clip_np, batch_obj_pos_np, path_map_index):
    obj_indices = np.arange(batch_obj_clip_np.shape[0])
    valid_obs_combination = np.where(path_map_index >= 0, path_map_index, 0).astype(int)
    valid_obs_combination = np.squeeze(valid_obs_combination)

    selected_obj_clip = batch_obj_clip_np[obj_indices, valid_obs_combination,:]

    mask = np.squeeze(path_map_index > 0)
    obj_clip = selected_obj_clip[mask]
    obj_pos = batch_obj_pos_np[mask]

    return obj_clip, obj_pos
def save_testing_data(scan, path_vps, vps_pos, gt_obj_pos):
    '''
    given the scan id and path idxs, using the machanism in etp to get the object pos
    and corresponding visual data
    '''
    assert gt_obj_pos is not None, "The gt_obj_pos is None"
    # read the cg files
    with h5py.File('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/cgobj_clip_90scans.hdf5', 'r') as f:
        batch_obj_clips = f[f'{scan}_obj_clip_ft'][...].astype(np.float32)
        paths_map_indices = f[f'{scan}_paths_indices'][...].astype(np.float32)
        batch_obj_pos = f[f'{scan}_obj_pos'][...].astype(np.float32)
    # get path_idx
    with open('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/all_paths_90scans.json', 'r') as f:
        all_paths_indices = json.load(f)
    paths_indices = all_paths_indices[scan]
    path_indx = paths_indices.index(path_vps)
      ## check the path_idx
    assert path_vps == paths_indices[path_indx], "The path idx is not correct"
    # get the map idx
    map_idx = paths_map_indices[path_indx]      #肯定是这个map数据出错了
    # get the obj pos correponding to the map
    def get_scene_map_etp(batch_obj_clip_np, batch_obj_pos_np, path_map_index):
        obj_indices = np.arange(batch_obj_clip_np.shape[0])
        valid_obs_combination = np.where(path_map_index >= 0, path_map_index, 0).astype(int)
        valid_obs_combination = np.squeeze(valid_obs_combination)

        selected_obj_clip = batch_obj_clip_np[obj_indices, valid_obs_combination,:]

        mask = np.squeeze(path_map_index > 0)
        # mask = path_map_index >= 0
        # print(f"mask.shape: {mask.shape}")
        
        obj_clip = selected_obj_clip[mask]
        obj_pos = batch_obj_pos_np[mask]

        return obj_clip, obj_pos
    _, obj_pos = get_scene_map_etp(batch_obj_clips, batch_obj_pos, map_idx)
    assert obj_pos.shape[0] == len(gt_obj_pos), f"The object pos is not correct, obj_pos size:{obj_pos.shape[0]}, gt_obj_pos size:{len(gt_obj_pos)}"
    # use the object pos to get the visual data
    save_visual_data_etp(obj_pos, vps_pos, scan)
def get_unique_vps(scan):
    json_dir =["/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_train_enc.jsonl",
                "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_prevalent_aug_train_enc.jsonl",
                "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_seen_enc.jsonl",
                "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_unseen_enc.jsonl"]
    all_trajectory_info = get_all_paths4all_scans(json_dir)
    paths = get_paths4scan(scan, all_trajectory_info)
    print(f" get all the paths for scan {scan}, the length of paths is {len(paths)}")
    unique_viewpoints = get_unique_viewpoints(paths)
    return unique_viewpoints, paths
def saving_visual_data(scan, vps_pos, object_pos, whole_scene_map,class_colors):
    # load the whole scene map
    scene_map_folder = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R"
    scene_map_file = scene_map_folder + f"/{scan}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz"
    save_folder = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data"
    save_path = save_folder + f"/{scan}_new_testing.pkl.gz"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_map_file", type=str, default=scene_map_file)
    parser.add_argument("--savefile", type=str, default=save_path)
    args = parser.parse_args()
    vis_fg = MapObjectList()
    print("Start searching the object in front ground...")
    get_vis_scene_map(whole_scene_map, vis_fg, object_pos)
    # print("Start searching the object in back ground...")
    # get_vis_scene_map(bg_objects, vis_bg, batch_obj_pos)
    # get viewpoint coordinates
    vps_coordinates = vps_pos
    # transform the pcd and bbox of objects
    # vis_fg, vps_coordinates = transform_pcd_bbox(vis_fg, vps_coordinates, vps_coordinates[-1],scan,path)
    # concate the result
    vis_result = {
        "objects": vis_fg.to_serializable(),
        "bg_objects": None,
        "class_colors": class_colors,
        "viewpoints": vps_coordinates,
    }
    # save the vis_scene_map
    if not os.path.exists(os.path.dirname(args.savefile)):
        os.makedirs(os.path.dirname(args.savefile), exist_ok=True)
    with gzip.open(args.savefile, "wb") as f:
        pickle.dump(vis_result, f)
        print(f"Save vis_scene_map to {args.savefile} successfully!")
def get_correct_obj_pos(scan, path_idxs,vps_pos):
    '''
    achieve a correct mask technique for the object retrieval
    '''
    # get the whole scene map
    whole_scene_map, class_colors = get_whole_scene_map(scan)
    # get all vp id in this scan
    unique_viewpoints, all_paths = get_unique_vps(scan)
    # testing path
    path_testing = [path_idxs]
    # generate obj_obs_mask, paths_mask
    _, _, path_testing_mask, gt_obj_pos = merger_clip4all_combinations(whole_scene_map, path_testing, unique_viewpoints)
    # generate scene map data for the whole scene
    print("start to generate the scene map data for the whole scene")
    merged_clips_np, batch_clip_masks, paths_masks, batch_obj_pos = merger_clip4all_combinations(whole_scene_map, all_paths, unique_viewpoints)
    print("end to generate the scene map data for the whole scene")
    # print space size of merged_clips_np, batch_clip_masks, paths_masks, batch_obj_pos in MB
    print(f"merged_clips_np space size: {merged_clips_np.nbytes / 1024 / 1024} MB")
    print(f"batch_clip_masks space size: {batch_clip_masks.nbytes / 1024 / 1024} MB")
    print(f"paths_masks space size: {paths_masks.nbytes / 1024 / 1024} MB")
    print(f"batch_obj_pos space size: {batch_obj_pos.nbytes / 1024 / 1024} MB")
    
    # get obj indices
    obj_indices = get_obj_indices(batch_clip_masks, paths_masks)
    # check the non-negative count is equal to the gt_obj_pos or not
    non_negative_count = (obj_indices != -1).sum().item()
    print(f"非 -1 的值的数量: {non_negative_count}")
    assert non_negative_count == len(gt_obj_pos), "The object pos is not correct"
    #
    # get obj pos
    # _, obj_pos_testing = get_scene_map_testing(scan, obj_indices)
    # assert obj_pos_testing.shape[0] == len(gt_obj_pos), f"The object pos is not correct, obj_pos size:{obj_pos_testing.shape[0]}, gt_obj_pos size:{len(gt_obj_pos)}"
    # print(" the size is the same!!! we success!!!")
    # # using the object pos to get the visual data
    # saving_visual_data(scan, vps_pos, obj_pos_testing, whole_scene_map, class_colors)
    
def get_scene_map_testing(scan, obj_indices):
    '''
    load clip and pos hdf5 file, and get the object clip and object pos corresponding to the obj_indices
    '''
    # load the hdf5 file
    with h5py.File('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/cgobj_clip_90scans.hdf5', 'r') as f:
        batch_obj_clips = f[f'{scan}_obj_clip_ft'][...].astype(np.float32)
        paths_map_indices = f[f'{scan}_paths_indices'][...].astype(np.float32)
        batch_obj_pos = f[f'{scan}_obj_pos'][...].astype(np.float32)
    _, obj_pos = get_scene_map_etp(batch_obj_clips, batch_obj_pos, obj_indices)
    return obj_pos
def get_whole_scene_map(scan):
    '''
    given scan and load the whole scene map
    '''
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
    whole_scene_map, _, class_colors = load_result(args.scene_map_file)
    return whole_scene_map, class_colors

def save_visual_data_cg(path_idxs, scan):
    '''
    this function is for saving the visual data for input from cg project
    given vp_idxs and scan id, and then generate the visual data
    '''
    
    json_dir =["/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_train_enc.jsonl",
        "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_prevalent_aug_train_enc.jsonl",
        "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_seen_enc.jsonl",
        "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_unseen_enc.jsonl"]
    all_trajectory_info = get_all_paths4all_scans(json_dir)
    paths = get_paths4scan(scan, all_trajectory_info)
    print(f" get all the paths for scan {scan}, the length of paths is {len(paths)}")
    unique_viewpoints = get_unique_viewpoints(paths)
    gt_obj_pos = save_scene_map(scan, path_idxs, unique_viewpoints)
    assert gt_obj_pos is not None, "The gt_obj_pos is None"
    
    return gt_obj_pos
    
def save_visual_data_etp(object_pos, vps_pos, scan):
    '''
    this function is for saving the visual data for input from etp project
    '''
    # load the whole scene map
    scene_map_folder = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R"
    scene_map_file = scene_map_folder + f"/{scan}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz"
    save_folder = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data"
    save_path = save_folder + f"/{scan}_visual_testing.pkl.gz"
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
    vis_fg = MapObjectList()
    vis_bg = MapObjectList()
    print("Start searching the object in front ground...")
    get_vis_scene_map(whole_scene_map, vis_fg, object_pos)
    # print("Start searching the object in back ground...")
    # get_vis_scene_map(bg_objects, vis_bg, batch_obj_pos)
    # get viewpoint coordinates
    vps_coordinates = vps_pos
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
def get_obj_indices(batch_obj_obs_mask, paths_mask):
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
    non_zero_mask = batch_obj_obs_mask.any(dim=3)
    is_subset = (batch_obj_obs_mask <= paths_mask).all(dim=3) & non_zero_mask  # Shape: (n_paths, n_objects, n_obs_combinations)
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
    
    # check if 

    return best_combination_indices
# test the combination function
def test_combination():
    # generate the test data
    obj_obs_mask = np.array([
        [[0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 1, 0]],
        [[0, 0, 1, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 0]],
        [[1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
    ])
    print(f'obj_obs_mask shape: {obj_obs_mask.shape}')
    # generate path mask
    path_mask = np.array([[0,1,0,1,0]])
    # transform the mask to tensor
    obj_obs_mask = torch.from_numpy(obj_obs_mask)
    path_mask = torch.from_numpy(path_mask)
    best_combination_indices = get_obj_indices(obj_obs_mask, path_mask)
    print(f'best_combination_indices: {best_combination_indices}')
def main():
    '''
    this function is used to save the visual data for the scene
    given input as :
        object pos
        vps pos
        scan id
    '''
    #read the object global pos for npy file
    object_pos = np.load('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data/object_global_save.npy')
    # read the vp global pos for npy file
    vps_pos = np.load('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data/path_global_save.npy')
    # scan id
    scan = 'D7N2EKCX4Sj'
    # save the visual data
    # save_visual_data_etp(object_pos, vps_pos, scan)
    # using the old but correct method to visualize
    path_idxs = ['e3c67078918d48a8a37abdbd38c61839', '1d7b7a08654f46df87604e7ae30f06b5', 
                 '61e6284b6ef541e59a87efa918514255', 'b3ea270a560d4fc784e7c7d4ca0e2248', 
                 '5bc65c559e2c4edc92ac6e9832d28ab1', '0a447b165b724cc8a73b00aafb9f8997']
    # get gt_obj_pos
    # gt_obj_pos = save_visual_data_cg(path_idxs, scan)
    # reproducing the visual data
    # save_testing_data(scan, path_idxs, vps_pos, gt_obj_pos)
    get_correct_obj_pos(scan, path_idxs, vps_pos)
if __name__ == '__main__':
    main()
    # test_combination()