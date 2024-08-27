'''
Instruction and trajectory (view and object features) dataset
'''
import os
import json
from re import T
from turtle import heading
import jsonlines
import numpy as np
import h5py
import math

from .common import load_nav_graphs
from .common import get_angle_fts, get_view_rel_angles
from .common import calculate_vp_rel_pos_fts
from .common import softmax
from model.bev_utils import transfrom3D
import pickle
import time
import sys

MAX_DIST = 30   # normalize
MAX_STEP = 10   # normalize
TRAIN_MAX_STEP = 20

class ReverieTextPathData(object):
    def __init__(
        self, anno_files, img_ft_file, dep_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, depth_feat_size=128, angle_feat_size=4,
        obj_feat_size=None, obj_prob_size=None, max_objects=20,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        val_sample_num=None,
    ):
        self.img_ft_file = img_ft_file
        self.dep_ft_file = dep_ft_file
        self.obj_ft_file = obj_ft_file

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.depth_feat_size = depth_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.act_visited_node = act_visited_node

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}
            self._feature_store_depth = {}
            self._hsg_store = {}
            
            # for edges
            self._dict_scan_edges = {}

        # {scan_vp: {vp: [viewidx, rel_angle_dist, rel_heading, rel_elevation]}}
        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir)
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in self.all_point_rel_angles]

        self.data = []
        for anno_file in anno_files:
            with jsonlines.open(anno_file, 'r') as f:
                for item in f:
                    self.data.append(item)

        if val_sample_num:
            # cannot evaluate all the samples as it takes too much time
            sel_idxs = np.random.permutation(len(self.data))[:val_sample_num]
            self.data = [self.data[sidx] for sidx in sel_idxs]

    def __len__(self):
        return len(self.data)
    def get_scene_map(self, batch_obj_clip_np, batch_obj_pos_np, path_map_index):
        # print(f"batch_obj_clip_np.shape {batch_obj_clip_np.shape}")
        # print(f"batch_obj_pos_np.shape {batch_obj_pos_np.shape}")
        # print(f"path_map_index.shape {path_map_index.shape}")
        obj_indices = np.arange(batch_obj_clip_np.shape[0])
        valid_obs_combination = np.where(path_map_index >= 0, path_map_index, 0).astype(int)
        valid_obs_combination = np.squeeze(valid_obs_combination)
        # print(f"valid_obs_combination.shape: {valid_obs_combination.shape}")
        
        selected_obj_clip = batch_obj_clip_np[obj_indices, valid_obs_combination,:]
        # print(f"selected_obj_clip.shape: {selected_obj_clip.shape}")
        # mask = path_map_index >= 0
        mask = np.squeeze(path_map_index > 0)
        # mask = path_map_index >= 0
        # print(f"mask.shape: {mask.shape}")
        
        obj_clip = selected_obj_clip[mask]
        obj_pos = batch_obj_pos_np[mask]
        # print(f"obj_clip.shape: {obj_clip.shape}")
        # print(f"obj_pos.shape: {obj_pos.shape}")
        return obj_clip, obj_pos


    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)

            obj_attrs = {}
            obj_fts = np.zeros((0, self.obj_feat_size+self.obj_prob_size), dtype=np.float32)
            if self.obj_ft_file is not None:
                with h5py.File(self.obj_ft_file, 'r') as f:
                    if key in f:
                        obj_fts = f[key][...].astype(np.float32)
                        obj_fts = obj_fts[:self.max_objects]
                        for attr_key, attr_value in f[key].attrs.items():
                            if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids']:
                                obj_attrs[attr_key] = attr_value[:self.max_objects]
            if self.in_memory:
                self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        gt_obj_id = item['instr_id'].split('_')[1]
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            # it occurs when the gt_objid is not in max_objects
            obj_label = -100 # ignore 
            # print('No groundtruth obj_id', item['instr_id'], len(obj_ids))
        return obj_label

    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                        + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k # [stop] is 0
            # local: 
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1 # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, 
        return_obj_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item.get('heading', 0)
        pos_vps = item['pos_vps']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos':
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == 'neg_in_gt_path':
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.graphs[scan].nodes.keys() if vp not in noneg_vp_set]
                end_vp = end_vps[np.random.randint(len(end_vps))]

        gt_path = self.shortest_paths[scan][start_vp][end_vp]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            # 'vp_objids': last_vp_objids,
            'vp_angles': last_vp_angles,
        }

        if return_obj_label:
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp][0]
            heading = (viewidx % 12) * math.radians(30)
            # elevation = (viewidx // 12 - 1) * math.radians(30)
            elevation = 0
        return heading, elevation

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []

        for vp in path:
            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            
            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]
                    obj_box_fts[k] = [h/self.obj_image_h, w/self.obj_image_w, (h*w)/self.obj_image_size]           
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get('obj_ids', [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        return traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
               last_vp_angles, last_vp_objids
        
    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s'%(scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        if self.act_visited_node:
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # shape=(num_gmap_vpids, 7)
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)
        
        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i+1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]] / MAX_DIST

        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists
    
    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position_cg'], 
                    self.graphs[scan].nodes[vp]['position_cg'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                    (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
        
    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)
                
        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len+1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts

        return vp_pos_fts
    def get_bev_inputs(self, scan, cur_vp, cur_heading, cur_elevation, cand_vpids):
        assert cur_elevation == 0

        x, y, z = self.graphs[scan].nodes[cur_vp]['position'][:3]
        # rgbs, depths = self.get_scanvp_grid_feature(scan, cur_vp)   # idx0: view 12

        # camera to world
        xyzhe = np.zeros([12, 5]).astype(np.float32)
        xyzhe[:, 0] = x
        xyzhe[:, 1] = z
        xyzhe[:, 2] = -y
        xyzhe[:, 3] = -np.arange(12) * np.radians(30) # counter-clock
        xyzhe[:, 4] = np.pi

        # world to camera
        xyzhe = np.zeros([1, 5]).astype(np.float32)
        xyzhe[:, 3] = cur_heading
        T_w2c = transfrom3D(xyzhe)
        return T_w2c  
       

class R2RTextPathData(ReverieTextPathData):
    def __init__(
        self, anno_files, img_ft_file, dep_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, depth_feat_size=128, angle_feat_size=4,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        val_sample_num=None
    ):
        super().__init__(
            anno_files, img_ft_file, dep_ft_file, None, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size, depth_feat_size=depth_feat_size,
            angle_feat_size=angle_feat_size, obj_feat_size=0, obj_prob_size=0, 
            max_objects=0, max_txt_len=max_txt_len, in_memory=in_memory,
            act_visited_node=act_visited_node, val_sample_num=val_sample_num
        )

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts = self._feature_store[key]
            dep_fts = self._feature_store_depth[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)
            with h5py.File(self.dep_ft_file, 'r') as f:
                dep_fts = f[key][...].astype(np.float32)
            if self.in_memory:
                self._feature_store[key] = view_fts
                self._feature_store_depth[key] = dep_fts
        return view_fts, dep_fts

    def get_act_labels(self, end_vp, end_idx, item, gmap_vpids, traj_cand_vpids):
        if end_vp == item['path'][-1]:  # stop
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            gt_next_vp = item['path'][end_idx + 1]
            for k, cand_vp in enumerate(gmap_vpids):
                if cand_vp == gt_next_vp:
                    global_act_label = k
                    break
            # local: 
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                if cand_vp == gt_next_vp:
                    local_act_label = k + 1 # [stop] is 0
                    break
        return global_act_label, local_act_label
    def get_cg_inputs(self,obj_lens, o_o_edge_index, o_o_edge_type):
        graph_info = {}
        # _, _, edges_objects, edges_relationship = self.get_obj_edges(scan, obj_clip_fts, obj_global_pos)
        # generate node_type with shape(num_nodes) with 0 value
        node_type = np.zeros(obj_lens,dtype=np.int32)
        # generate edge_index with shape(num_edges,2) with 0 value
        # edge_index = np.zeros((2, obj_lens),dtype=np.int32)
        # edge_index = edge_index.T
        # generate edge_type with shape(num_edges) with 0 value
        # edge_type = np.zeros(obj_lens,dtype=np.int32)
        # edge_time with shape(num_edges) with 0 value
        edge_time = np.zeros(o_o_edge_type.shape[0],dtype=np.float32)
        
        ### all this value is for obj-obj relation
        # graph_info['node_type'] = node_type
        # graph_info['edge_index'] = edge_index
        # graph_info['edge_type'] = edge_type
        # graph_info['edge_time'] = edge_time
        # set all graph info to None
        graph_info['obj_lens'] = obj_lens
        graph_info['node_type'] = node_type
        graph_info['edge_index'] = o_o_edge_index
        graph_info['edge_type'] = o_o_edge_type
        graph_info['edge_time'] = edge_time
        return graph_info
    
    # def get_HSG_inputs(self, scan):
    #     Base_dir = "/home/lg1/lujia/VLN_HGT/PlacesAndRoom/"
    #     data_name = scan + ".pkl"
    #     print(Base_dir + scan + data_name)
    #     with open(Base_dir + scan + data_name, 'rb') as f:
    #         loaded_data = pickle.load(f)
    #     loaded_places_np = loaded_data['places']
    #     loaded_room_connection = loaded_data['room_connection']
    #     print(f"loaded_places_np: {loaded_places_np}")
    #     print(f"loaded_room_connection: {loaded_room_connection}")
    #     return loaded_places_np, loaded_room_connection
    
    # def(self, edge):
    #     # on is 0, in is 1
    #     # bei on is 2. bei in is 3
    #     pass
    #     return edge_index, edge_type
    
    def get_obj_list(self, scan, path):        
        cg_file_path = "/home/lg1/lujia/VLN_HGT/pretrain_src/datasets/cg_data/cgobj_clip_90scans.hdf5"
        path_mapping_file = "/home/lg1/lujia/VLN_HGT/pretrain_src/datasets/cg_data/all_paths_90scans.json"
        # 将path转换为字符串形式的键，假设path是一个列表
        # path_key = '_'.join(path) if isinstance(path, list) else path

        # # get the path mapping from json file
        # if self.in_memory and scan in self._hsg_store:
        #     batch_obj_clips = self._hsg_store[scan]["clip"]
        #     batch_obj_pos = self._hsg_store[scan]["pos"]
        #     paths_map_indices = self._hsg_store[scan]["paths_map_indices"]
        #     if path_key in self._hsg_store[scan]:
        #         obj_clip = self._hsg_store[scan][path_key]["clip"]
        #         obj_pos = self._hsg_store[scan][path_key]["pos"]
        #     else:
        #         all_paths_indices = self._hsg_store["all_paths_indices"]
        #         # get path index from a list by input the element
        #         paths_indices = all_paths_indices[scan]
        #         path_indx = paths_indices.index(path)
        #         # get the path_map_index
        #         path_map_index = paths_map_indices[path_indx]
        #         # get the obj_clip_np
        #         obj_clip, obj_pos = self.get_scene_map(batch_obj_clips, batch_obj_pos, path_map_index)
        #         if self.in_memory:
        #             self._hsg_store[scan][path_key] = {}
        #             self._hsg_store[scan][path_key]['clip'] = obj_clip
        #             self._hsg_store[scan][path_key]['pos'] = obj_pos
        # else:
            # read the obj_clip from hdf5
            # t1 = time.time()
        with h5py.File(cg_file_path, 'r') as f:
            batch_obj_clips = f[f'{scan}_obj_clip_ft'][...].astype(np.float32)
            paths_map_indices = f[f'{scan}_paths_indices'][...].astype(np.float32)
            batch_obj_pos = f[f'{scan}_obj_pos'][...].astype(np.float32)

        # print(f" time for read hdf5 file: {time.time() - t1}")
        # read the paths_indices from json
        # check whether "all_paths_indices" is in the memory
        if "all_paths_indices" not in self._hsg_store:
            with open(path_mapping_file, 'r') as f:
                all_paths_indices = json.load(f)
            
            self._hsg_store["all_paths_indices"] = all_paths_indices
        else:
            all_paths_indices = self._hsg_store["all_paths_indices"]

        # get the object list and pos information
        # get path index from a list by input the element
        paths_indices = all_paths_indices[scan]
        path_indx = paths_indices.index(path)
        # get the path_map_index
        path_map_index = paths_map_indices[path_indx]
        # get the obj_clip_np
        # t_match_1 = time.time()
        obj_clip, obj_pos = self.get_scene_map(batch_obj_clips, batch_obj_pos, path_map_index)
            # print(f" time for match: {time.time() - t_match_1}")
            # save to the memory
            # if self.in_memory:
            #     self._hsg_store[scan] = {}
            #     self._hsg_store[scan]['clip'] = batch_obj_clips
            #     self._hsg_store[scan]['pos'] = batch_obj_pos
            #     self._hsg_store[scan][path_key] = {}
            #     self._hsg_store[scan][path_key]['clip'] = obj_clip
            #     self._hsg_store[scan][path_key]['pos'] = obj_pos
            #     self._hsg_store[scan]["paths_map_indices"] = paths_map_indices
        return obj_clip, obj_pos
    
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
        import numpy as np
        """Calculate the center of a bbox given its eight corners."""
        return np.mean(bbox_corners, axis=0)
    

    def get_all_edges_from_hdf5_memory(
        self,
        hdf5_path="/data0/vln_datasets/preprocessed_data/edges_hdf5", 
        hdf5_file_name="edges.hdf5"):
        if self.in_memory and self._dict_scan_edges != {}:
            return self._dict_scan_edges
        else:
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
            self._dict_scan_edges = edges_dict
            print("Loading all edges from hdf5 successful!")
            return self._dict_scan_edges

    def get_obj_edges(self, scan, obj_clip, obj_pos):
        
        edges_objects = None
        edges_relationship = None
        edges_relationship_extend = None

        # objs, edges_relationship = self.get_edges_from_file(scan)
        dict_scan_edges = self.get_all_edges_from_hdf5_memory()
        objs, edges_relationship = self.get_edges_from_dict(dict_scan_edges, scan)
        correspondence_dict = self.find_correspondece_similarity_bboxcenter(objs, obj_pos)
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
        
        return np.array(edges_objects), np.array(edges_relationship), np.array(edges_objects_extend), np.array(edges_relationship_extend)

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
            # elif edge_relationship == "none of these":
            #     encoded_relationship = 4
        elif revert_flag == True:
            if edge_relationship =="a on b":
                encoded_relationship = 2
            elif edge_relationship == "a in b":
                encoded_relationship = 3
            elif edge_relationship == "b on a":
                encoded_relationship = 0
            elif edge_relationship == "b in a":
                encoded_relationship = 1
            # elif edge_relationship == "none of these":
            #     encoded_relationship = 4
        
        return encoded_relationship
    
    def get_edges_from_dict(self, dict_scan_edges, scan):
        objs_bbox_list, edges_relationship = dict_scan_edges[scan]
        objs = [{"bbox_np": obj} for obj in objs_bbox_list]
        return objs, edges_relationship

    def get_places_room_inputs(self, scan):
        if self.in_memory and scan in self._hsg_store and 'places' in self._hsg_store[scan]:
            # judge wether places in _hsg_store as a key
            places_info = self._hsg_store[scan]['places']
            room_connection_info = self._hsg_store[scan]['room_connection']
        else:
            Base_dir = "/home/lg1/lujia/VLN_HGT/PlacesAndRoom/"
            data_name = "/" + scan + ".pkl"
            #print(Base_dir + scan + data_name)
            with open(Base_dir + scan + data_name, 'rb') as f:
                loaded_data = pickle.load(f)
            places_info = loaded_data['places']
            loaded_room_connection = loaded_data['room_connection']
            # print(f"loaded_room_connection: {loaded_room_connection}")
            # transfer the room_connection to adj_matrix representation
            room_connection_info = self.dict_to_adj_matrix(loaded_room_connection, scan)
            if self.in_memory:
                self._hsg_store[scan] = {}
                self._hsg_store[scan]['places'] = places_info
                self._hsg_store[scan]['room_connection'] = room_connection_info
        return places_info, room_connection_info

    def dict_to_adj_matrix(self, room_dict, scan):
        # 获取房间数量
        if len(room_dict) == 0:
            n = 1
            adj_matrix = np.zeros((n, n), dtype=int)
            return adj_matrix
        n = max(room_dict.keys()) + 1
        # max_value = max(max(values) for values in room_dict.values())
        # if n>100:
        #     print(f" there are {n} rooms in the scan, which more than 100!!!!!!!!")
        # print(f" there are {n} rooms in the scan")
        # print(f"this is the room_dict: {room_dict}")
        
        # 初始化 n x n 的邻接矩阵，全部元素为 0
        adj_matrix = np.zeros((n, n), dtype=int)
        # if adj_matrix.shape[0] == 14:
        #     print(room_dict)
        # print("################# dataset.py #########################")
        # print(f"n: {n}")
        # print(f"adj_matrix.shape: {adj_matrix.shape}")
        # if n -1 == adj_matrix.shape[0]:
        #     print("the room number is incorrect#############")
        
        # 遍历字典，填充邻接矩阵
        for room, neighbors in room_dict.items():
            for neighbor in neighbors:
                adj_matrix[room][neighbor] = 1
        
        return adj_matrix
    ###### room feature projection
    def get_scanvp_grid_feature(self, scan, viewpoint):
        img_grid_file = "/home/lg1/lujia/VLN_HGT/pretrain_src/img_features/vit_b16_224_clip_patch_habitat.hdf5"
        dep_grid_file = "/home/lg1/lujia/VLN_HGT/pretrain_src/img_features/depth_14x14.hdf5"
        key = '%s_%s' % (scan, viewpoint)
        with h5py.File(img_grid_file, 'r') as f:
            rgbs = f[key][...].astype(np.float32)
        with h5py.File(dep_grid_file, 'r') as f:
            depths = f[key][...].astype(np.float32)
        return rgbs, depths
    def get_single_vp_prjInfo(self, scan, vp, cur_heading=None, cur_elevation=None):
        x, y, z = self.graphs[scan].nodes[vp]['position_cg'][:3]
        rgbs, depths= self.get_scanvp_grid_feature(scan, vp)   # idx0: view 12

        # camera to world transformation for each viewpoint in the path
        # xyzhe = np.zeros([12, 5]).astype(np.float32)
        # xyzhe[:, 0] = x
        # xyzhe[:, 1] = z
        # xyzhe[:, 2] = y
        # xyzhe[:, 3] = -np.arange(12) * np.radians(30)  # counter-clock
        # xyzhe[:, 4] = np.pi
        # T_c2w = transfrom3D(xyzhe)
        xyzhe = np.zeros([12, 5]).astype(np.float32)
        xyzhe[:, 0] = x
        xyzhe[:, 1] = y
        xyzhe[:, 2] = z
        xyzhe[:, 3] = -np.arange(12) * np.radians(30)  # counter-clock
        xyzhe[:, 4] = 0
        T_c2w = transfrom3D(xyzhe)

        # world to camera transformation only if cur_heading and cur_elevation are provided
        # This is optional, used only if specific heading and elevation are given for the last viewpoint
        if cur_heading is not None and cur_elevation is not None:
            xyzhe = np.zeros([1, 5]).astype(np.float32)
            xyzhe[:, 0] = x
            xyzhe[:, 1] = z
            xyzhe[:, 2] = -y
            xyzhe[:, 3] = cur_heading
            xyzhe[:, 4] = cur_elevation
            T_w2c = transfrom3D(xyzhe)
        else:
            T_w2c = None

        # shift from world to camera is constant for each viewpoint
        S_w2c = np.array([x, z, -y], dtype=np.float32)

        return rgbs, depths, T_c2w, T_w2c, S_w2c
    
    def get_room_fts_projection(self, scan, path, cur_heading, cur_elevation):
        # 初始化列表来存储路径中所有视点的信息
        room_fts, room_depth, rm_T_c2w, rm_S_w2c = [], [], [], []

        # 遍历路径中的所有视点，收集特征和转换矩阵
        for vp in path:
            rgbs, depths, T_c2w, _, S_w2c = self.get_single_vp_prjInfo(scan, vp)
            room_fts.append(rgbs)
            room_depth.append(depths)
            rm_T_c2w.append(T_c2w)
            rm_S_w2c.append(S_w2c)

        # 对于路径中的最后一个视点，计算世界到摄像机的变换
        cur_vp = path[-1]  # 当前视点作为路径的最后一个点
        _, _,  _, T_w2c, _ = self.get_single_vp_prjInfo(scan, cur_vp, cur_heading, cur_elevation)

        # 将列表转换为NumPy数组或Tensor，以便进一步处理
        room_fts = np.array(room_fts)        # (num_vp, 12, 196, 768)
        room_depth = np.array(room_depth)    # (num_vp, 12, 14, 14)
        rm_T_c2w = np.array(rm_T_c2w)        # (num_vp, 12, 4, 4)
        rm_T_w2c = np.array([T_w2c])         # (1, 4, 4)
        rm_S_w2c = np.array(rm_S_w2c)        # (num_vp, 1, 3)

        return room_fts, room_depth, rm_T_c2w, rm_T_w2c, rm_S_w2c
    
    def get_cur_angle_hat(self, scan, path, start_heading):
        if len(path) < 2:
            heading = -start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp][0]
            heading = -(viewidx % 12) * math.radians(30)
            # elevation = (viewidx // 12 - 1) * math.radians(30)
            elevation = 0
        return heading, elevation
    
    def get_obj_transformation(self, scan, path, cur_heading):
        # get the position of last viewpoint for S_w2c_cg
        cur_vp = path[-1]
        x, y, z = self.graphs[scan].nodes[cur_vp]['position_cg']
        S_w2c_cg = np.array([[x,y,z]], dtype=np.float32)
        # get the transformation matrix for T_w2c_cg
        xyzhe = np.zeros([1, 5]).astype(np.float32)
        xyzhe[:, 3] = - cur_heading
        T_w2c_cg = transfrom3D(xyzhe)
        return S_w2c_cg, T_w2c_cg
    def get_path_pos(self, scan, path):
        path_pos = []
        for vp in path:
            x, y, z = self.graphs[scan].nodes[vp]['position_cg']
            path_pos.append([x, y, z])
        return np.array(path_pos, dtype=np.float32)
    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, end_vp=None
    ):
        # start_time = time.time()
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item['heading']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos': 
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ['neg_in_gt_path', 'neg_others']:
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1]
                end_idx = np.random.randint(len(end_vps))
                end_vp = end_vps[end_idx]
        else:
            assert end_vp in gt_path
            end_idx = gt_path.index(end_vp)
            
        gt_path = gt_path[:end_idx+1]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)
        cur_heading_cg, cur_elevation_cg = self.get_cur_angle_hat(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            
        traj_view_img_fts, traj_view_dep_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading_cg, cur_elevation_cg)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading_cg, cur_elevation_cg, len(traj_nav_types[-1]))

        # cg result
        T_w2c = self.get_bev_inputs(scan, end_vp, cur_heading, cur_elevation, traj_cand_vpids[-1])
        # obj_lens = 6
        # obj_clip_fts = np.ones((obj_lens, 1024),dtype=np.float32)
        # obj_global_pos = np.ones((obj_lens, 3),dtype=np.float32)
        # count the time for get_obj_list
        # time_1 = time.time()
        obj_clip_fts, obj_global_pos = self.get_obj_list(scan, gt_path)
        # get transformation operation for the object list which transform the object list to the camera coordinate
        S_w2c_cg, T_w2c_cg = self.get_obj_transformation(scan, gt_path, cur_heading)
        # print(f" S_w2c_cg.shape: {S_w2c_cg.shape}")
        # time_2 = time.time()
        # print(f"get_obj_list time: {time_2 - time_1}")
        # get the time in seconds
        
        # get edges
        # time_1 = time.time()
        _, _, o_o_edges_index, o_o_edges_type = self.get_obj_edges(scan, obj_clip_fts, obj_global_pos)
        # print(f"get_obj_edges time: {time.time() - time_1}")
        
        # if obj_global_pos == None:
        #     print(f"obj_global_pos {obj_global_pos}")
        obj_lens = obj_clip_fts.shape[0]
        # print(f"obj_lens: {obj_lens}")
        ### for HGT  
        room_fts, room_depth, rm_T_c2w, _, _ = self.get_room_fts_projection(scan, gt_path, cur_heading, cur_elevation)
        graph_info = self.get_cg_inputs(obj_lens, o_o_edges_index, o_o_edges_type)
        # time_room_1 = time.time()
        places, room_connection = self.get_places_room_inputs(scan)
        # time_room_2 = time.time()
        # print(f"get_places_room_inputs time: {time_room_2 - time_room_1}")
        places_lens = places.shape[0]
        # get path pos
        path_pos = self.get_path_pos(scan, gt_path)
        # print(f"places {places}")
        # if self.in_memory:
        #     self._place_store[scan] = places_info
        #     self._room_store[scan] = room_connection_info
        # calculate the size of obj_clip_fts, obj_global_pos,o_o_edges_index, o_o_edges_type,room_fts, room_depth, rm_T_c2w, rm_T_w2c, rm_S_w2c
        # and places, room_connection
        # 统计变量所占空间大小

        # def get_size(obj):
        #     if isinstance(obj, (list, tuple)):
        #         return sum(get_size(item) for item in obj)
        #     elif hasattr(obj, 'nbytes'):
        #         return obj.nbytes
        #     else:
        #         return sys.getsizeof(obj)
        
        # variables = {
        #     "obj_clip_fts": obj_clip_fts,
        #     "obj_global_pos": obj_global_pos,
        #     "S_w2c_cg": S_w2c_cg,
        #     "T_w2c_cg": T_w2c_cg,
        #     "o_o_edges_index": o_o_edges_index,
        #     "o_o_edges_type": o_o_edges_type,
        #     "room_fts": room_fts,
        #     "room_depth": room_depth,
        #     "rm_T_c2w": rm_T_c2w,
        #     "rm_T_w2c": rm_T_w2c,
        #     "rm_S_w2c": rm_S_w2c,
        #     "graph_info": graph_info,
        #     "places": places,
        #     "room_connection": room_connection
        # }
        
        # # 计算每个变量的大小（以字节为单位）
        # sizes = {var_name: get_size(var_value) for var_name, var_value in variables.items()}
        
        # # 将大小转换为MB
        # sizes_mb = {var_name: size / (1024 ** 2) for var_name, size in sizes.items()}
        
        # # 计算总大小（以MB为单位）
        # total_size_mb = sum(sizes_mb.values())
        
        # # 按大小排序并获取前5个变量
        # sorted_sizes_mb = sorted(sizes_mb.items(), key=lambda item: item[1], reverse=True)
        # top_5_mb = sorted_sizes_mb[:5]
        
        # # 输出结果
        # print(f"总大小: {total_size_mb:.2f} MB")
        # print("前5个占用空间最大的变量:")
        # for var_name, size in top_5_mb:
        #     print(f"{var_name}: {size:.2f} MB")
        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_view_dep_fts': [x[:, :self.depth_feat_size] for x in traj_view_dep_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            ### for ConceptGraph
            'T_w2c': T_w2c,                 # (1, 4, 4)
            'obj_global_pos': obj_global_pos, # (num_obj, 3)
            'obj_clip_fts': obj_clip_fts,   # (num_obj, 1024) not 768
            'obj_lens': obj_lens,             # (1)
            'S_w2c_cg': S_w2c_cg,           # (1, 3)
            'T_w2c_cg': T_w2c_cg,           # (1, 4, 4)
            # for HGT input
            'graph_info': graph_info,    # dict{node_type, edge_time, edge_type, edge_index}
            ### for Hydra
            'room_fts': room_fts,           # (num_vp, 12, 196, 768)
            'room_depth': room_depth,       # (num_vp, 12, 14, 14)
            'rm_T_c2w': rm_T_c2w,           # (num_vp, 12, 4, 4)
            # 'rm_T_w2c': rm_T_w2c,           # (1, 4, 4)
            # 'rm_S_w2c': rm_S_w2c,           # (num_vp, 1, 3)
            
            'places': places, # (num_places, 4)
            'places_lens': places_lens, # (1)
            'room_connection': room_connection, # (num_room, num_room)
            # 'vp_pos_fts': vp_pos_fts,
            # 'vp_angles': last_vp_angles,
            'path_pos': path_pos,
            'scan': scan,
            'path_idxs': gt_path,
        }
        # print(f"total time: {time.time() - start_time}")
        # # print the time for get_obj_list percentage of total time
        # print(f"get_obj_list time: {(time_2 - time_1)/(time.time() - start_time)}%")

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item, gmap_vpids, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)

        return outs

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_view_dep_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []

        for vp in path:
            view_fts, dep_fts = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_dep_fts, view_angles, cand_vpids = [], [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_dep_fts.append(dep_fts[v[0]])
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_dep_fts.extend([dep_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_dep_fts = np.stack(view_dep_fts, 0)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            # view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            
            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_view_dep_fts.append(view_dep_fts)
            # traj_loc_fts.append(np.concatenate([view_ang_fts, view_box_fts], 1))
            traj_loc_fts.append(view_ang_fts)
            traj_nav_types.append([1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)))
            traj_cand_vpids.append(cand_vpids)
            
            last_vp_angles = view_angles

        return traj_view_img_fts, traj_view_dep_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, last_vp_angles