import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
# from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.env_utils import get_places_room_inputs
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST


from .utils import get_camera_orientations12

from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

# from habitat_extensions.utils import observations_to_image, navigator_video_frame, generate_video
from habitat.utils.visualizations.utils import append_text_to_image

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks,pad_tensors_2dim
from torch.nn.utils.rnn import pad_sequence

from vlnce_baselines.obj_edge_processor import ObjEdgeProcessor
from vlnce_baselines.models.graph_utils import heading_from_quaternion
from vlnce_baselines.models.bev_utils import transfrom3D, bevpos_polar, PointCloud

# torch.set_grad_enabled(True)

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    device = tensors[0].device
    output = torch.zeros(*size, dtype=dtype).to(device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output
@baseline_registry.register_trainer(name="SS-ETP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.obj_edge_processor = ObjEdgeProcessor()

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

        
    
    #def _load_ALEX_data(self):
    
            
    #def _load_obj_edge_data(self):
        #instruction_encoder.py adding cg_model()  
    def _init_obj_edge_processor(self):
        self.obj_edge_processor = ObjEdgeProcessor(
            objs_hdf5_save_dir="/data0/vln_datasets/preprocessed_data/finetune_cg_hdf5",
            objs_hdf5_save_file_name="finetune_cg_data.hdf5",
            edges_hdf5_save_dir="/data0/vln_datasets/preprocessed_data/edges_hdf5",
            edges_hdf5_save_file_name="edges.hdf5",
            connectivity_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity",
            connectivity_file_name="scans.txt",
            obj_feature_name="clip",
            obj_pose_name="bbox_np",
            allobjs_list=[],
            alledges_dict={},
            allvps_pos_dict={}
            )
        
        self.obj_edge_processor.allobjs_list = self.obj_edge_processor.load_allobjs_from_hdf5()
        self.obj_edge_processor.alledges_dict = self.obj_edge_processor.get_edges_from_hdf5()
        self.obj_edge_processor.allvps_pos_dict = self.obj_edge_processor.load_allvps_pos_from_connectivity_json()
        # print("Loaded objs and edges from hdf5")

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                # print(action, orient)
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        # os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
        self.config.VIDEO_OPTION = False
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            #R2R HFOV 90
            #RxR HFOV 63
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config( # this is to initialize the policy commentor: Alex
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)
        # Note: the self.policy.net is the vln_bert in Policy_ViewSelection_ETP.py
        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.IL.lr)

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
			
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == 'ndtw':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError
       
        return torch.tensor(teacher_actions).cuda()

    def _vp_feature_variable(self, obs): # return is a dict
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left,
        }

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        #obj_edge_processor initialization
        self._init_obj_edge_processor()

        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))
        self.places_dict, self.room_connection_dict = get_places_room_inputs(self.config)
        
        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)
        # mix float training
        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)
        
    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.

            with autocast():
                self.rollout('train', ml_weight, sample_ratio)
            # self.rollout('train', ml_weight, sample_ratio)
            self.scaler.scale(self.loss).backward() # self.loss.backward()
            self.scaler.step(self.optimizer)        # self.optimizer.step()
            self.scaler.update()
            # cancle mix float training
            

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        os.makedirs(self.config.VIDEO_DIR, exist_ok=False)
        # if self.config.VIDEO_OPTION:
        self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
        self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
        self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
        # os.makedirs(self.config.VIDEO_DIR, exist_ok=False)
        shift = 0.
        orient_dict = {
            'Back': [0, math.pi + shift, 0],            # Back
            'Down': [-math.pi / 2, 0 + shift, 0],       # Down
            'Front':[0, 0 + shift, 0],                  # Front
            'Right':[0, math.pi / 2 + shift, 0],        # Right
            'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
            'Up':   [math.pi / 2, 0 + shift, 0],        # Up
        }
        sensor_uuids = []
        H = 224
        for sensor_type in ["RGB"]:
            sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
            for camera_id, orient in orient_dict.items():
                camera_template = f"{sensor_type}{camera_id}"
                camera_config = deepcopy(sensor)
                camera_config.WIDTH = H
                camera_config.HEIGHT = H
                camera_config.ORIENTATION = orient
                camera_config.UUID = camera_template.lower()
                camera_config.HFOV = 90
                sensor_uuids.append(camera_config.UUID)
                setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()
        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None
        # print("initial self.stat_eps:", len(self.stat_eps))
        # print("eps_to_eval:", eps_to_eval)
        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori
    ############################# hsg processing ########################
    def cartesian_to_polar(self, coords):
        """
        Convert Cartesian coordinates to polar coordinates [cos(theta), sin(theta), normalized distance].
        
        Parameters:
        coords (torch.Tensor): Tensor of shape (bs, N, 3) representing (x, y, z) coordinates.

        Returns:
        torch.Tensor: Tensor of shape (bs, N, 3) representing [cos(theta), sin(theta), normalized distance].
        """
        x = coords[:, :, 0]
        y = coords[:, :, 1]
        z = coords[:, :, 2]

        # Calculate distance
        distances = torch.sqrt(x**2 + y**2 + z**2)

        # Calculate theta
        theta = torch.atan2(torch.sqrt(x**2 + y**2), z)  # theta is the angle with the z-axis

        # Avoid division by zero
        safe_distances = torch.where(distances == 0, torch.tensor(1.0).to(coords.device), distances)
        cos_theta = x / safe_distances
        sin_theta = y / safe_distances

        # Handle distance normalization for each batch independently
        max_distances, _ = torch.max(distances, dim=1, keepdim=True)
        normalized_distances = distances / max_distances

        # Combine results
        polar_coords = torch.stack([cos_theta, sin_theta, normalized_distances], dim=-1)
        
        return polar_coords
    def get_pos_feat(self, batch_global_pos,cur_pos, cur_ori):
        '''
        input pos in global coordinates
        output pos in local coordinates in polar system
        '''
        # print(f" shape of batch_global_pos:{batch_global_pos.shape}")
        bs = self.envs.num_envs
        batch_global_pos = batch_global_pos - torch.tensor([0, 1.25, 0]).to(torch.float32).cuda()
        S = []
        for i, pos_i in enumerate(cur_pos):
            x, y, z = pos_i
            S.append([np.array([x, y, z])])
        S = np.vstack(S).astype(np.float32)              # bs, 3
        S = torch.from_numpy(S).cuda()
        xyzhe = np.zeros([bs, 5])
        for i, ori_i in enumerate(cur_ori):
            xyzhe[i, 3] = -heading_from_quaternion(ori_i)
        T = torch.from_numpy(transfrom3D(xyzhe)).cuda()  # bs, 4, 4

        # transform to ego coord
        batch_ego_pos = batch_global_pos - S[:, None, :]
        ones = torch.ones(batch_ego_pos.shape[:2]).unsqueeze(-1).cuda()
        batch_ego_pos1 = torch.cat([batch_ego_pos, ones], dim=-1)               # bs, N, 4
        batch_ego_pos1 = torch.matmul(batch_ego_pos1.to(torch.float32), T.transpose(1, 2).to(torch.float32))        # bs, N, 4
        batch_ego_pos = batch_ego_pos1[:, :, :3]                                # bs, N, 3
        
        #transform to polar coord
        batch_pos_feat = self.cartesian_to_polar(batch_ego_pos)                 # bs, N, 3

        return batch_pos_feat
    def get_obj_room_relation(self, batch_obj_global_pos, batch_places_global_pos, place_masks):
        '''
        calculate the distance between object and places node and assign the object to the nearest room
        input:batch_obj_global_pos (bs, n_obj, 3), batch_places_global_pos (bs, n_place, 4)
        ouput: assigned_room_idxs (bs, n_obj)
        '''
        B, N, _ = batch_obj_global_pos.shape
        P = batch_places_global_pos.shape[1]

        # 扩展pc和places以便广播
        batch_obj_global_pos_exp = batch_obj_global_pos.unsqueeze(2)  # [B, N, 1, 3]
        places_exp = batch_places_global_pos[:, None, :, :3]  # [B, 1, P, 3]

        # 计算距离
        # 使用norm而不是sqrt(sum((a-b)^2))是因为norm是更通用的做法
        # 在这里p=2指L2范数，即欧氏距离
        distances = torch.norm(batch_obj_global_pos_exp - places_exp, p=2, dim=3)  # [B, N, P]
        inf = float('inf')
        distances = distances.masked_fill(~place_masks.unsqueeze(1), inf)  # [B, N, P]
        # 找到最小距离及其索引（即最近的place）
        min_dist, min_indices = torch.min(distances, dim=2)  # [B, N], [B, N]

        # 选择每个feature最近的place的room idx
        # gather用于根据最小索引选择places中对应的room idx
        assigned_room_idxs = torch.gather(batch_places_global_pos[:, :, 3], 1, min_indices)  # [B, N]
        return assigned_room_idxs
    def map_room_idx_to_local(self, map_list, global_room_idx):
        # 创建索引映射
        indices = torch.argsort(map_list)

        # 使用torch.gather进行映射转换
        # 首先，将tensor转换为映射列表中的索引位置
        positions = torch.searchsorted(map_list[indices], global_room_idx)

        # 然后使用gather获取最终的映射索引
        mapped_tensor = torch.gather(indices, 0, positions)
        return mapped_tensor
    def extract_subgraph_edges(self, connection_tensor, sub_nodes):
        """
        提取子图的边列表。
        :param connection_tensor: 原图的完整的连接矩阵，torch.Tensor格式。
        :param sub_nodes: 子节点的列表，表示我们感兴趣的节点的索引。
        :return: 边列表，每个边是一个包含两个节点索引的列表 [n_edge, 2] 格式。
        """
        # print(f"connection_tensor {connection_tensor}")
        # print("###################### pretrain_cmt.py 0######################")
        # print(f"connection_tensor.shape {connection_tensor.shape}")
        # print(f"sub_nodes {sub_nodes}")
        # 提取子图的连接矩阵
        sub_tensor = connection_tensor[sub_nodes][:, sub_nodes]

        # 找到所有的边（即连接存在的地方）
        edges = (sub_tensor == 1).nonzero(as_tuple=False)
        # 转换索引为原始图中的索引
        actual_edges = sub_nodes[edges].tolist()
        # print("###################### pretrain_cmt.py 1######################")
        # print(f"actual_edges {actual_edges}")
        
        return actual_edges
    def get_graph_info(self, batch_graph_info, batch_obj_room_idxs, batch_room_connection):
        '''
        gather the graph information
        '''
        bs = batch_obj_room_idxs.shape[0]
        # places = batch['places']
        batch_hsg_lens = []
        for bs_idx in range(bs):
            # graph info
            room_connection = batch_room_connection[bs_idx]
            graph_info = batch_graph_info[bs_idx]
            ### get the room lens
            current_room_idxs = batch_obj_room_idxs[bs_idx]
            current_room_idxs = current_room_idxs[current_room_idxs != -1].unique().long().sort()[0]
            current_map_list = current_room_idxs
            current_room_lens = current_room_idxs.shape[0]
            graph_info['room_lens'] = current_room_lens
            current_hsg_lens = graph_info['obj_lens'] + current_room_lens
            batch_hsg_lens.append(current_hsg_lens)
            ### update node type
            #generate tensor with value 1 with lens of current_room_lens
            room_node_type = torch.ones(current_room_lens, dtype=torch.long).to(room_connection.device)
            graph_info['node_type'] = torch.cat([graph_info['node_type'], room_node_type], dim=0)
            ### update edge index
            obj_room_idxs = batch_obj_room_idxs[bs_idx][:graph_info['obj_lens']] # here are scan global room idxs
            # change to local room idxs
            obj_room_idxs = self.map_room_idx_to_local(current_map_list, obj_room_idxs)
            # generate a tensor from 0 to range(obj_lens)
            obj_node_idx = torch.arange(graph_info['obj_lens']).to(room_connection.device)
            # offset the obj_room_idxs with the obj_lens
            global_obj_room_idxs = obj_room_idxs + graph_info['obj_lens']
                # stack the obj_node_idx and global_obj_room_idxs
            obj_room_edge_index = torch.stack([obj_node_idx, global_obj_room_idxs], dim=1)
                # corresponding edge type with 2 with the same length
            obj_room_edge_type = torch.full((obj_room_edge_index.shape[0],), 0, dtype=torch.long).to(room_connection.device)
                # opposite direction
            room_obj_edge_index = torch.stack([global_obj_room_idxs, obj_node_idx], dim=1)
                # corresponding edge type with 3 with the same length
            room_obj_edge_type = torch.full((room_obj_edge_index.shape[0],), 1, dtype=torch.long).to(room_connection.device)
            ###
            room_room_edge_index = self.extract_subgraph_edges(room_connection, current_room_idxs) # here are also scan global room idxs
            room_room_edge_index = torch.tensor(room_room_edge_index).to(room_connection.device)
            # change to local room idxs
            room_room_edge_index = self.map_room_idx_to_local_2dim(current_map_list, room_room_edge_index)
            # r-r edge type
            room_room_edge_type = torch.full((room_room_edge_index.shape[0],), 2, dtype=torch.long).to(room_connection.device)
            ### concat all edge index
            # edge_index = torch.cat([obj_room_edge_index, room_obj_edge_index, room_room_edge_index], dim=0).t()  for single processing
            edge_index = torch.cat([obj_room_edge_index, room_obj_edge_index, room_room_edge_index], dim=0)  #for batch_processing
            edge_type = torch.cat([obj_room_edge_type, room_obj_edge_type, room_room_edge_type], dim=0)
            # update the graph_info
            graph_info['edge_index'] = edge_index.to(torch.long)
            graph_info['edge_type'] = edge_type
            ### update edge time
            edge_type = torch.full((edge_type.shape[0],), 0, dtype=torch.long).to(room_connection.device)
            graph_info['edge_time'] = edge_type
        
        # update the batch
        batch_hsg_lens = torch.LongTensor(batch_hsg_lens).to(room_connection.device)
        return batch_graph_info, batch_hsg_lens
    def get_room_info(self, assigned_room_idxs, batch_graph_list, places):
        bs = places.shape[0]
        batch_room_pos = []
        batch_room_lens = []
        for batch_idx in range(bs):
            # processing in per map
            # get unique room idxs for assigned_room_idxs
            room_idxs = assigned_room_idxs[batch_idx]
            room_idxs = room_idxs[room_idxs != -1].unique()
            # print(f" there are these room idxs{room_idxs}")
            graph_info = batch_graph_list[batch_idx]
            # pring obj_lens
            obj_lens = graph_info['obj_lens']
            # print(f"obj_lens {obj_lens}")
            # update room length
            batch_room_lens.append(room_idxs.shape[0])
            rooms_pos = []
            for room_idx in room_idxs:
                room_idx = room_idx.item()
                # get the feature idxs for the room
                current_room_places_idxs = torch.where(places[batch_idx][:,3] == room_idx)[0]
                # get the room feature for the room
                current_room_places_pos = places[batch_idx, current_room_places_idxs,:3]
                # print(f"current_room_places_pos shape {current_room_places_pos.shape}")

                rooms_pos.append(current_room_places_pos)
            
            # first pad the room_fts to get [num_rooms, max_places_num, 3]
            current_scan_room_pos = pad_tensors_wgrad(rooms_pos)
            batch_room_pos.append(current_scan_room_pos)
        # [bs, num_rooms, max_places_num, 3]
        # padding in first two dimension
        batch_room_pos = pad_tensors_2dim(batch_room_pos)
        # calculate the center of each room get [bs, n_room,3] in (x,y,z)
        batch_room_pos = batch_room_pos.mean(dim=2).float()
        # transfer into polar coordinate
        # print(f"batch_room_pos.shape {batch_room_pos.shape}")
        # print the type of tensor
        # print(f"batch_room_pos.dtype {batch_room_pos.dtype}")
        batch_room_pos = self.cartesian_to_polar(batch_room_pos)
        # transfer batch room lens to tensor
        batch_room_lens = torch.tensor(batch_room_lens).to(assigned_room_idxs.device)
        return batch_room_pos, batch_room_lens
    def map_room_idx_to_local_2dim(self, map_list, global_room_idx):
        #global_room_idx has 2 dimensions
        # 创建索引映射
        indices = torch.argsort(map_list)
        indices_2dim = indices.unsqueeze(1)
        # 使用torch.gather进行映射转换
        # 首先，将tensor转换为映射列表中的索引位置
        positions = torch.searchsorted(map_list[indices], global_room_idx)

        # 然后使用gather获取最终的映射索引
        mapped_tensor = torch.gather(indices_2dim.expand(-1,2), 0, positions)
        return mapped_tensor
    
    def dict_to_adj_matrix(self, room_dict):
        '''
        transfer the room_dict to adj_matrix
        '''
        # 获取房间数量
        if len(room_dict) == 0:
            n = 1
            adj_matrix = np.zeros((n, n), dtype=int)
            return adj_matrix
        n = max(room_dict.keys()) + 1
        adj_matrix = np.zeros((n, n), dtype=int)
        for room, neighbors in room_dict.items():
            for neighbor in neighbors:
                adj_matrix[room][neighbor] = 1
        
        return adj_matrix

    def get_cg_inputs(self,obj_lens):
        graph_info = {}
        # generate node_type with shape(num_nodes) with 0 value
        node_type = np.zeros(obj_lens,dtype=np.int32)
        # generate edge_index with shape(num_edges,2) with 0 value
        edge_index = np.zeros((2, obj_lens),dtype=np.int32)
        edge_index = edge_index.T
        # generate edge_type with shape(num_edges) with 0 value
        edge_type = np.zeros(obj_lens,dtype=np.int32)
        # edge_time with shape(num_edges) with 0 value
        edge_time = np.zeros(obj_lens,dtype=np.float32)
        ### all this value is for obj-obj relation
        # graph_info['node_type'] = node_type
        # graph_info['edge_index'] = edge_index
        # graph_info['edge_type'] = edge_type
        # graph_info['edge_time'] = edge_time
        # set all graph info to None
        graph_info['obj_lens'] = obj_lens
        graph_info['node_type'] = torch.from_numpy(node_type).cuda()
        graph_info['edge_index'] = None
        graph_info['edge_type'] = None
        graph_info['edge_time'] = None
        return graph_info
    def get_batch_graph_dict(self, batch_graph_info, hsg_lens):
        bs = len(batch_graph_info)
        batch_node_type = torch.cat([graph_info['node_type'] for graph_info in batch_graph_info], dim=0)
        batch_edge_type = torch.cat([graph_info['edge_type'] for graph_info in batch_graph_info], dim=0)
        batch_edge_time = torch.cat([graph_info['edge_time'] for graph_info in batch_graph_info], dim=0)
        ### batch_edge_index
        cum_sum = torch.cumsum(hsg_lens[:-1], dim=0)
        cum_sum = torch.cat([torch.tensor([0]).to(batch_edge_type.device), cum_sum], dim=0)
         # get repeat num
        repeat_num = torch.tensor([graph_info['edge_type'].shape[0] for graph_info in batch_graph_info]).to(batch_edge_type.device)
        assert cum_sum.shape[0] == repeat_num.shape[0], "the shape of cum_sum and repeat_num should be the same"
        off_set_lens = cum_sum.repeat_interleave(repeat_num)
        batch_edge_index = torch.cat([graph_info['edge_index'] for graph_info in batch_graph_info], dim=0)
        batch_edge_index = batch_edge_index + off_set_lens.unsqueeze(1)
        batch_edge_index = batch_edge_index.t()
        # check whether the data is continuous in memory
        if batch_node_type.is_contiguous():
            batch_node_type = batch_node_type.contiguous()
        if batch_edge_type.is_contiguous():
            batch_edge_type = batch_edge_type.contiguous()
        if batch_edge_time.is_contiguous():
            batch_edge_time = batch_edge_time.contiguous()
        if batch_edge_index.is_contiguous():
            batch_edge_index = batch_edge_index.contiguous()
        batch_graph_info = {
            'node_type':batch_node_type,
            'edge_type':batch_edge_type,
            'edge_index':batch_edge_index,
            'edge_time':batch_edge_time,
            'split_lens':hsg_lens
        }
        
        return batch_graph_info
    def _nav_hsg_variable(self,cur_pos, cur_ori, obj_clip_list, obj_pos_list,places_list, room_connection_list):
        obj_clip_list = [torch.from_numpy(np.stack(obj_clip,axis=0)).cuda() for obj_clip in obj_clip_list]
        obj_pos_list = [torch.from_numpy(np.stack(obj_pos,axis=0)).cuda() for obj_pos in obj_pos_list]
        # print(f" shape of element in object pos list {obj_pos_list[0].shape}")
        # print(f" shape of element in object clip list {obj_clip_list[0].shape}")
        ## room info
        places_list = [torch.from_numpy(places).cuda() for places in places_list]
        room_connection_list = [torch.from_numpy(room_connection).cuda() for room_connection in room_connection_list]
        
        batch_obj_global_pos = pad_tensors(obj_pos_list)
        batch_obj_clip = pad_tensors(obj_clip_list)
        
        batch_places = pad_tensors(places_list)
        batch_graph_list = [self.get_cg_inputs(obj_lens) for obj_lens in [obj_clip.shape[0] for obj_clip in obj_clip_list]]
        # batch_room_connection = [self.dict_to_adj_matrix(room_connection) for room_connection in room_connection_list]
        
        batch_places_lens = [places.shape[0] for places in places_list]
        # print(f" type of bathc_places_lens {type(batch_places_lens)}")
        # batch_places_lens = torch.stack(batch_places_lens,dim=0).cuda()
        batch_places_lens = torch.tensor(batch_places_lens).cuda()
        place_masks = gen_seq_masks(batch_places_lens)
        # get single scan info into a list
        # np to tensor and batch collate
        ## reprojection and relation computation
        batch_obj_pos_fts = self.get_pos_feat(batch_obj_global_pos, cur_pos, cur_ori)
        ## compute the obj room relation
        assigned_room_idx = self.get_obj_room_relation(batch_obj_global_pos, batch_places, place_masks)
        ## gathering hsg info
        batch_graph_list, batch_hsg_lens = self.get_graph_info(batch_graph_list, assigned_room_idx, room_connection_list)
        batch_graph_dict = self.get_batch_graph_dict(batch_graph_list, batch_hsg_lens)
        ## get room lens and room pos
        batch_room_pos, batch_room_lens = self.get_room_info(assigned_room_idx, batch_graph_list, batch_places)
        
        return {
            'obj_pos_fts':batch_obj_pos_fts,
            'obj_clip_fts':batch_obj_clip,
            'room_fts':batch_obj_clip,
            'room_pos':batch_room_pos,
            'assigned_room_idxs':assigned_room_idx,
            'room_lens':batch_room_lens,
            'graph_info':batch_graph_list,
            'batch_hsg_lens':batch_hsg_lens,
            'batch_graph_info_dict':batch_graph_dict
        }
    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError
        # rgb_frames = [[] for _ in range(self.envs.num_envs)]
        
        ### Note: This three line is only used for video generation in eval mode
        # rgb_frames = []
        # episode_ids = []
        # scene_ids = []
        #######################################################
        
        distance_to_goal_list = []
        collision_count_list = []
        self.envs.resume_all()
        observations = self.envs.reset() # find position randomly
        # print("observations length:", len(observations))
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        #####
        # Path(VPlist) 
        # scene_id = self.envs.current_episodes()[0].scene_id
        # logger.info(f"episode_id: {scene_id}")
        # output is similar like "data/scene_datasets/mp3d/cV4RVeZvu5T/cV4RVeZvu5T.glb"
        # extract the scene id
        # scene_id = scene_id.split('/')[-2]
        # logger.info(f"scene_id: {scene_id}")
        # places_info_scene_id = self.places_dict[scene_id]
        # room_connection_scene_id = self.room_connection_dict[scene_id]
        # Comment from Alex: then you need to use places_info_scene_id and room_connection_scene_id
        # with the observation to create a batch and send it to self.policy.net
        
        
        
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]   
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config.IL.loc_noise, 
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs
        
        # path_vp = [[]] * self.envs.num_envs # Don't ever using this to init the path_vp. Each list in the path_vp is the same list. It will be operated together in the loop.
        path_vp = [[] for _ in range(self.envs.num_envs)]
        # debug_distance_list = [[] for _ in range(self.envs.num_envs)]
        # print("self.max_len:", self.max_len)
        id_x = 0
        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]
            
            # cand waypoint prediction
            wp_outputs = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                in_train = (mode == 'train' and self.config.IL.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs) # Policy interaction
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            # print("num of envs: ", self.envs.num_envs)
            cur_pos, cur_ori = self.get_pos_ori() # Env interaction # Habitat API
            # print("cur pos:", cur_pos)
            # print("cur ori:", cur_ori)
            
            cur_vp, cand_vp, cand_pos = [], [], []
            
            cur_scan, closest_vp = [], [] # M2G
            filtered_objs, filtered_objs_feature, filtered_objs_pose, edges_objects_extend, edges_relationship_extend = [], [], [], [], [] # M2G
            batch_places = []
            batch_room_connection_dicts = []
            for i in range(self.envs.num_envs):
                
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i) # cur_vp_i is viewpoint to position (x, y, z), str(len(self.node_pos))
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)
                # start_time = time.time()
                ### Hierarchical Scene Graph (room & places node)
                scene_id = (self.envs.current_episodes()[0].scene_id).split('/')[-2]
                places_info_scene_id = self.places_dict[scene_id]
                room_connection_scene_id = self.room_connection_dict[scene_id]
                batch_places.append(places_info_scene_id)
                batch_room_connection_dicts.append(room_connection_scene_id)
                end_hydra_time = time.time()
                # print(f" time for hydra processing: {time.time()-start_time}")
                ### ConceptGraph data processing
                # M2G
                cur_scan_i = (self.envs.current_episodes()[0].scene_id).split('/')[-2]
                
                closest_vp_i, debug_distance = self.obj_edge_processor.find_similar_vp(cur_scan_i, cur_pos[i])
                # debug_distance_list[i].append(debug_distance)
                
                cur_scan.append(cur_scan_i)
                closest_vp.append(closest_vp_i)
                path_vp[i].append(closest_vp_i)
                
                _objs, _objs_feature, _objs_pose, _edges_objects, _edges_relationship = self.obj_edge_processor.get_merge_feature_edges_relationship(cur_scan[i], path_vp[i])
                filtered_objs.append(_objs)
                filtered_objs_feature.append(_objs_feature)
                filtered_objs_pose.append(_objs_pose)
                edges_objects_extend.append(_edges_objects)
                edges_relationship_extend.append(_edges_relationship)
                
            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            ## check whether nav_inputs is in the same device as policy.net
            
            ### hsg data processing
            # print(f" type of filtered_objs_feature: {type(filtered_objs_feature[0])}")
            # print(f" filtered_objs_feature[0]: {filtered_objs_feature[0]}")
            # print(f" type of filtered_objs_pose: {type(filtered_objs_pose)}")
            # nav_inputs = self._nav_hsg_variable(cur_pos, cur_ori, filtered_objs_feature, filtered_objs_pose, batch_places, batch_room_connection_dicts)
            nav_inputs.update(self._nav_hsg_variable(cur_pos, cur_ori, filtered_objs_feature, filtered_objs_pose, batch_places, batch_room_connection_dicts))
            # if nav_inputs['obj_clip_fts'].device != self.policy.net.device:
            #     print(f" nav_inputs['obj_clip_fts'].device: {nav_inputs['obj_clip_fts'].device}")
            #     print(f" self.policy.net.device: {self.policy.net.device}")
            #     print(f" they are not in the same device")
            # else:                
            #     print(f" nav_inputs['obj_clip_fts'].device: {nav_inputs['obj_clip_fts'].device}")
            #     print(f" self.policy.net.device: {self.policy.net.device}")
            #     print(f" they are in the same device")
            # ###########
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self.policy.net(**nav_inputs) # teh output of model
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            # random sample demo
            # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
            # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
            # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float)<=sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            outputs = self.envs.step(env_actions) # 
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            # self.config.VIDEO_OPTION = ["disk"]
            # calculate metric
            # curr_eps = self.envs.current_episodes()
            # frame = observations_to_image(observations[i], infos[i])
            # frame = append_text_to_image(frame, curr_eps[i].instruction.instruction_text)
            # rgb_frames.append(frame)
            
            ### Note: This part is only used for video generation in eval mode
            # rgb_frames.append(
            #     navigator_video_frame(observations[i],
            #                           infos[i], vis_info)
            # )
            # episode_id = self.envs.current_episodes()[0].episode_id
            # episode_ids.append(episode_id)
            # scene_ids.append(self.envs.current_episodes()[0].scene_id.split('/')[-1].split('.')[-2])
            ########################################
            
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    
                    #add video save function test
                    # if len(self.config.VIDEO_OPTION) > 0:
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    logger.info(f"distanct to goal: {distances[-1]}")
                    distance_to_goal_list.append(distances[-1])
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    logger.info(f"collisions: {info['collisions']['count']}")
                    collision_count_list.append(info['collisions']['count'])
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric                
                    
                    id_x += 1    
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            # test generate video function
            
        # # print(debug_distance_list)
        # # Flatten the debug_distance_list
        # flattened_list = [item for sublist in debug_distance_list for item in sublist]

        # # Save to a .txt file
        # with open('/home/lg1/lujia/VLN_HGT/debug_distance_list.txt', 'a') as file:
        #     for number in flattened_list:
        #         file.write(f"{number}\n")
        # # Free the memory
        # del flattened_list
        # del debug_distance_list
        # gc.collect()
        
        # if len(self.config.VIDEO_OPTION) > 0:
        # print("check if enter here")
        # curr_eps = self.envs.current_episodes()
        # for i in reversed(list(range(self.envs.num_envs))):
        # print("in this iteration, the id_x passed is:", id_x)
        
        ### This part is only used for video generation in eval mode
        # if len(rgb_frames) < 13:
        #     # print("debug value: ", round(info["spl"], 3))
        #     if round(info["spl"], 3) >= 0.8 and distance_to_goal_list[0] < 0.9 and collision_count_list[0] == 0: 
        #         print("enter save video function")
        #         # print("length of rgb_frames:", len(rgb_frames))             
        #         generate_video(
        #             video_option=["disk"],
        #             video_dir=self.config.VIDEO_DIR,
        #             images=rgb_frames,
        #             episode_id=episode_ids[0],
        #             scene_id=scene_ids[0],
        #             checkpoint_idx=0,
        #             metrics={"SPL": round(info["spl"], 3)},
        #             tb_writer=None,
        #             fps=3
        #         )
        ########################################
        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())