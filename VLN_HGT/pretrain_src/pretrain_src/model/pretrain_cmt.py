from collections import defaultdict

from numpy import arange
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vilmodel import BertLayerNorm, BertOnlyMLMHead, GlocalTextPathCMT
from .ops import pad_tensors_2dim, pad_tensors_wgrad, gen_seq_masks
from .bev_utils import bevpos_polar, PointCloud
import math
import numpy as np
import sys
def build_projector():
    projector = PointCloud(math.radians(90),
                           1,
                           feature_map_height=14,
                           feature_map_width=14,
                           map_dim=11,
                           map_res=1,
                           world_shift_origin=torch.FloatTensor([0,0,0]).cuda(),
                           z_clip_threshold=0.5,
                           device=torch.device('cuda'))

    bev_pos = bevpos_polar(11).cuda()
    bev_pos = bev_pos.reshape(11 * 11, 3)[None, :, :] # 1 x 441 x 3
    return projector, bev_pos
class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class NextActionPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = GlocalTextPathCMT(config)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
            if self.config.obj_prob_size > 0 and self.config.obj_prob_size != self.config.image_prob_size:
                self.obj_classifier = RegionClassification(self.config.hidden_size, self.config.obj_prob_size)
            else:
                self.obj_classifier = None
        if 'sap' in config.pretrain_tasks:
            self.global_sap_head = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)
            # self.global_sap_head = ClsPrediction(self.config.hidden_size)
            # self.local_sap_head = ClsPrediction(self.config.hidden_size)
            # if config.glocal_fuse:
            #     self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
            # else:
            #     self.sap_fuse_linear = None
        if 'og' in config.pretrain_tasks:
            self.og_head = ClsPrediction(self.config.hidden_size)

        self.init_weights()
        self.tie_weights()
        self.projector, self.bev_pos_fts = build_projector()

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)
        #### room feature projection
    def room_fts_reprojection(self, batch):
        room_fts = batch.pop('room_fts')
        room_depth = batch.pop('room_depth')
        rm_T_c2w = batch.pop('rm_T_c2w')
        # T_w2c = batch.pop('T_w2c')
        # S_w2c = batch.pop('S_w2c')
        place_masks = batch['places_masks']
        bs = room_depth.shape[0]
        T_w2c = batch['T_w2c_cg']
        S_w2c = batch['S_w2c_cg']
        
        ### reporject to gloabl frame
        print(f" shape of room_depth {room_depth.shape}")
        depths_var = (room_depth * 10).reshape(-1, 1, 14, 14)   # bs*views, 1, 14, 14
        # print(f"deths_var.shape {depths_var.shape}")
        # print(f"rm_T_c2w.shape {rm_T_c2w.reshape.shape}")
        pc, pc_mask = self.projector.forward(depths_var, rm_T_c2w.reshape(-1, 4, 4))
        # save the global coordinate of patch feature in first scan
        
        
        # print(f" the global coordinate of patch feature in first scan {pc[0]}")

        # print(f"pc.shape {pc.shape}")
        # print(f"S_w2c.shape {S_w2c.shape}")
        ### Convert global coordinates to local coordinates
        pc = pc.reshape(bs, -1, 3)          # B, N, 3
        # save the global coordinate of patch feature in first scan
        pc_global_save = pc[0].cpu().detach().numpy()
        np.save("grid_global_save.npy", pc_global_save)
        pc_local = pc - S_w2c
        pc_mask = pc_mask.reshape(bs, -1)   # B, N
        ones = torch.ones(pc.shape[:2]).unsqueeze(-1).cuda()
        pc_local = torch.cat([pc, ones], dim=-1)                          # bs, N, 4
        pc_local = torch.matmul(pc_local, T_w2c.squeeze(1).transpose(1, 2)) # bs, N, 4
        pc_local = pc_local[:, :, :3]                                      # bs, N, 3
        
        # room classification
        padded_places = batch['places']
        _, assigned_room_idxs = self.assign_RoomIndx(pc, padded_places, pc_mask, place_masks)

        ### Convert local coordinates to polar coordinates
        pc_polar = self.cartesian_to_polar(pc_local)
        
        pc_room_feat = room_fts.reshape(bs, -1, 768)  # B, N, 768
        
        ### The pc_local here is in the local coordinate system
        batch.update({
            'room_fts': pc_room_feat,
            #'room_pos': pc_local,  # Use local coordinates
            'room_pos': pc_polar,  # Use polar coordinates
            'assign_room_idxs_grid': assigned_room_idxs
        })
        return batch
    
    def assign_RoomIndx(self, pc, padded_places, pc_masks, place_masks):
        # pc: [B, N, 3]
        # pc_room_feat: [B, N, feat_dim]
        # padded_places: [B, P, 4], P是经过padding后的places数量
        # param object_mask: 对象的掩码，形状为 [B, N]。
        # param place_mask: 房间的掩码，形状为 [B, P]。
        B, N, _ = pc.shape
        P = padded_places.shape[1]

        # 扩展pc和places以便广播
        pc_exp = pc.unsqueeze(2)  # [B, N, 1, 3]
        places_exp = padded_places[:, None, :, :3]  # [B, 1, P, 3]

        # 计算距离
        # 使用norm而不是sqrt(sum((a-b)^2))是因为norm是更通用的做法
        # 在这里p=2指L2范数，即欧氏距离
        distances = torch.norm(pc_exp - places_exp, p=2, dim=3)  # [B, N, P]
        inf = float('inf')
        distances = distances.masked_fill(~place_masks.unsqueeze(1), inf)  # [B, N, P]
        # 找到最小距离及其索引（即最近的place）
        min_dist, min_indices = torch.min(distances, dim=2)  # [B, N], [B, N]

        # 选择每个feature最近的place的room idx
        # gather用于根据最小索引选择places中对应的room idx
        assigned_room_idxs = torch.gather(padded_places[:, :, 3], 1, min_indices)  # [B, N]
        # print(f"assigned_room_idxs {assigned_room_idxs}")
        assigned_room_idxs = assigned_room_idxs.masked_fill(~pc_masks, -1)  # 将无效对象的房间索引设置为-1或其他标志值

        return min_dist, assigned_room_idxs   

    def build_RoomObject_relation(self, batch):
        """
        创建整个batch的图边索引，形状为[B, 2, N]。
        
        :param batch_objects: 一个形状为[B, N]的Tensor，表示每个batch中的N个对象的房间索引。
        :param object_to_global: Tensor，对象ID到全局ID的映射。
        :param room_to_global: Tensor，房间ID到全局ID的映射。
        :return: (edge_index, edge_type)
        """
        batch_objects_pos = batch['obj_global_pos']
        # save the global coordinate of object in first scan
        object_global_save = batch_objects_pos[0].cpu().detach().numpy()
        np.save("object_global_save.npy", object_global_save)
        batch_scan_idx = batch['scan']
        print(f"scan_idx {batch_scan_idx}")
        # first scan id
        scan_idx = batch_scan_idx[0]
        print(f"scan_idx {scan_idx}")
        batch_path_idxs = batch['path_idxs']
        print(f"path_idxs {batch_path_idxs[0]}")
        
        # print(f"object global coordinate in first scan {batch_objects_pos[0]}")
        # print(f"obj_global_pos {batch_objects_pos}")
        batch_places_pos = batch['places']
        # save places global coordinate in first scan
        places_global_save = batch_places_pos[0].cpu().detach().numpy()
        np.save("places_global_save.npy", places_global_save)
        # save the global coordinate of vp
        path_pos = batch['path_pos'][0].cpu().detach().numpy()
        np.save("path_global_save.npy", path_pos)
        # print(f"places global coordinates in first scan {batch_places_pos[0]}")
        sys.exit("stop here for saving the info")
        # object_to_global = batch.pop('obj_to_global')
        # room_to_global = batch.pop('room_to_global')
        object_masks = batch['obj_masks']
        place_masks = batch.pop('places_masks')
        B, N, _ = batch_objects_pos.shape
        ### get object2room index
        _, batch_objects = self.assign_RoomIndx(batch_objects_pos, batch_places_pos, object_masks, place_masks)
        batch.update({'obj_assigned_room_idxs': batch_objects})
        ### update graph infor
        # print(f"batch_objects {batch_objects}")
        # print(f"batch_objects.shape {batch_objects.shape}")
        
        self.generate_graph_info(batch, batch_objects)
        # 使用Tensor索引直接获取全局ID
        #object_global_ids = object_to_global[torch.arange(B) * N].repeat_interleave(N).view(B, N)
        # Handle invalid indices
        # valid_mask = batch_objects != -1
        # room_global_ids = torch.zeros_like(batch_objects)
        # room_global_ids[valid_mask] = room_to_global[torch.arange(B).unsqueeze(1), batch_objects][valid_mask]


        # # 转置以便堆叠
        # object_to_global = object_to_global.unsqueeze(2)  # 形状 [B, N, 1]
        # room_global_ids = room_global_ids.unsqueeze(2)  # 形状 [B, N, 1]

        # # 堆叠并转置
        # edge_index = torch.cat([object_to_global, room_global_ids], dim=2)  # 形状 [B, N, 2]
        # edge_index = edge_index.permute(0, 2, 1)  # 转置以获得形状 [B, 2, N]
        
        # # 构建edge_type，所有值为5，形状[B, N]
        # edge_type = torch.full((B, N), 0, dtype=torch.long)
        # batch.update({
        #     'room2obj_index': edge_index,
        #     'room2obj_type': edge_type,
        #     'added_edge_time': torch.zeros_like(edge_type),
        # })
        
        return batch
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
    def map_room_idx_to_local(self, map_list, global_room_idx):
        # 创建索引映射
        indices = torch.argsort(map_list)

        # 使用torch.gather进行映射转换
        # 首先，将tensor转换为映射列表中的索引位置
        positions = torch.searchsorted(map_list[indices], global_room_idx)

        # 然后使用gather获取最终的映射索引
        mapped_tensor = torch.gather(indices, 0, positions)
        return mapped_tensor
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
    
    def generate_graph_info(self, batch, batch_obj_room_idxs):
        batch_graph_info = batch.pop('graph_info')
        batch_room_connection = batch.pop('room_connection')
        bs = batch_obj_room_idxs.shape[0]
        # places = batch['places']
        batch_hsg_lens = []
        for bs_idx in range(bs):
            # current_places = places[bs_idx]
            # places_room_index = current_places[:,3].unique().long()
            
            # graph info
            room_connection = batch_room_connection[bs_idx]
            graph_info = batch_graph_info[bs_idx]
            ### get the room lens
            current_room_idxs = batch_obj_room_idxs[bs_idx]
            current_room_idxs = current_room_idxs[current_room_idxs != -1].unique().long().sort()[0]
            current_map_list = current_room_idxs
            # print(f"current_room_idx for object clustering {current_room_idxs}")
            # print(f"current_room_idx.shape {current_room_idxs.shape}")
            current_room_lens = current_room_idxs.shape[0]
            graph_info['room_lens'] = current_room_lens
            current_hsg_lens = graph_info['obj_lens'] + current_room_lens
            batch_hsg_lens.append(current_hsg_lens)
            ### update node type
            #generate tensor with value 1 with lens of current_room_lens
            room_node_type = torch.ones(current_room_lens, dtype=torch.long).to(room_connection.device)
            graph_info['node_type'] = torch.cat([graph_info['node_type'], room_node_type], dim=0)
            ### update edge index
            ## obj-room edge 
            # get the obj_room_idxs for current scan map
            # print(f"obj_room_idxs [idx] {batch_obj_room_idxs[bs_idx].shape}")
            # obj_lens = graph_info["obj_lens"]
            # print(f"obj_length {obj_lens}")
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
            obj_room_edge_type = torch.full((obj_room_edge_index.shape[0],), 1, dtype=torch.long).to(room_connection.device)
                # opposite direction
            room_obj_edge_index = torch.stack([global_obj_room_idxs, obj_node_idx], dim=1)
                # corresponding edge type with 3 with the same length
            room_obj_edge_type = torch.full((room_obj_edge_index.shape[0],), 3, dtype=torch.long).to(room_connection.device)
            #obj_room_edge_index = torch.cat([obj_room_edge_index, room_obj_edge_index], dim=0) # the whole set for o-r relationship
            # obj_room_edge_index = obj_room_edge_index.t()
            # graph_info['edge_index'] = torch.cat([graph_info['edge_index'], obj_room_edge_index], dim=1)
            ## room-room edge
            ### debug
            # print(f"room_connection.shape {room_connection.shape}")
            # print(f"c")
            # if room_connection.shape[0] == current_room_idxs.max():
            #     print("######################  it is impossible !!!!!!!!!!!!!!######################")
            #     print(f"places room indexs {places_room_index}")
            #     print(f"current_room_idxs {current_room_idxs}")
            ###
            room_room_edge_index = self.extract_subgraph_edges(room_connection, current_room_idxs) # here are also scan global room idxs
            room_room_edge_index = torch.tensor(room_room_edge_index).to(room_connection.device)
            # change to local room idxs
            room_room_edge_index = self.map_room_idx_to_local_2dim(current_map_list, room_room_edge_index)
            # r-r edge type
            room_room_edge_type = torch.full((room_room_edge_index.shape[0],), 4, dtype=torch.long).to(room_connection.device)
            ### concat all edge index
            edge_index = torch.cat([obj_room_edge_index, room_obj_edge_index, room_room_edge_index], dim=0)  #for single processing
            # edge_index = torch.cat([obj_room_edge_index, room_obj_edge_index, room_room_edge_index], dim=0)  #for batch_processing
            edge_type = torch.cat([obj_room_edge_type, room_obj_edge_type, room_room_edge_type], dim=0)
            # update the graph_info
            graph_info['edge_index'] = torch.cat([graph_info['edge_index'], edge_index.to(torch.long)], dim=0).t()
            
            graph_info['edge_type'] = torch.cat([graph_info['edge_type'], edge_type], dim=0)
            # print the shape of edge index and edge type
            # print("############## pretrain_cmt.py ################")
            # print(f"edge_index.shape {edge_index.shape}")
            # print(f"edge_type.shape {edge_type.shape}")
            # print()
            ### update edge time
            edge_time = torch.full((edge_type.shape[0],), 0, dtype=torch.long).to(room_connection.device)
            graph_info['edge_time'] = torch.cat([graph_info['edge_time'], edge_time], dim=0)
        
        # update the batch
        batch_hsg_lens = torch.LongTensor(batch_hsg_lens).to(room_connection.device)
        batch.update({'graph_info': batch_graph_info,
                      'hsg_lens': batch_hsg_lens})
        
            
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
    def object_pos_to_polar(self, batch):
        """
        Convert object global positions to local polar coordinates.
        
        Parameters:
        batch (dict): Batch of data containing 'object_global_pos' and 'T_w2c'.

        Returns:
        dict: Updated batch with 'object_pos_polar'.
        """
        object_global_pos = batch['obj_global_pos']
        T_w2c_cg = batch['T_w2c_cg']
        S_w2c_cg = batch['S_w2c_cg']
        bs = object_global_pos.shape[0]

        ### Convert global coordinates to local coordinates
        # shift with S_w2c_cg
        object_global_pos = object_global_pos - S_w2c_cg
        # rotation with T_w2c_cg
        ones = torch.ones(object_global_pos.shape[:2]).unsqueeze(-1).to(object_global_pos.device)
        object_global_pos = torch.cat([object_global_pos, ones], dim=-1)  # bs, N, 4
        object_local_pos = torch.matmul(object_global_pos, T_w2c_cg.squeeze(1).transpose(1, 2))  # bs, N, 4
        object_local_pos = object_local_pos[:, :, :3]  # bs, N, 3

        ### Convert local coordinates to polar coordinates
        object_pos_polar = self.cartesian_to_polar(object_local_pos)
        
        ### Update batch
        batch.update({
            'obj_pos_fts': object_pos_polar  # Use polar coordinates
        })

        return batch
    def compute_room_pos(self, batch):
        # assigned_room_idxs = batch['assign_room_idxs']
        assigned_room_idxs = batch["obj_assigned_room_idxs"]
        batch.update({'assigned_room_idxs':assigned_room_idxs,
                      'room_fts':batch['obj_clip_fts']})
        ######### temporal change 
        places = batch.pop('places')
        # print(f"places {places}")
        # print(f"places.shape {places.shape}")
        bs = places.shape[0]
        batch_graph_info = batch['graph_info']
        batch_room_pos = []
        batch_room_lens = []
        for batch_idx in range(bs):
            # processing in per map
            # get unique room idxs for assigned_room_idxs
            room_idxs = assigned_room_idxs[batch_idx]
            room_idxs = room_idxs[room_idxs != -1].unique()
            # print(f" there are these room idxs{room_idxs}")
            graph_info = batch_graph_info[batch_idx]
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
        # print the shape of each element in batch_room_pos
        # for i in range(len(batch_room_pos)):
        #     print(f"batch_room_pos[{i}].shape {batch_room_pos[i].shape}")
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
        batch.update({
            'room_pos':batch_room_pos,
            'room_lens':batch_room_lens
        })
        return batch
    def cat_graph_info(self, batch):
        '''
        input: batch['graph_info'] = [graph_info_1, graph_info_2, ...
        output: batch['batch_graph_info'] = {'node_type':[], 'edge_index':[], 'edge_type':[], 'edge_time':[], 'split_lens':[]}
        '''
        batch_graph_info = batch['graph_info']
        hsg_lens = batch['hsg_lens']
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
        batch_edge_index = torch.cat([graph_info['edge_index'].t() for graph_info in batch_graph_info], dim=0).to(torch.int64)
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
        batch.update({'batch_graph_info':batch_graph_info})
        return batch       
    def check_reporjection(self, batch):
        '''
        check whether grid feature and objects are in the same frame
        '''
        grid_room_indices = batch['assign_room_idxs_grid']
        print(f"grid_room_indices.shape {grid_room_indices.shape}")
        obj_room_indices = batch['assigned_room_idxs']
        print(f"obj_room_indices.shape {obj_room_indices.shape}")
        bs = grid_room_indices.shape[0]
        for i in arange(bs):
            # get grid_indx
            grid_idx = grid_room_indices[i]
            grid_idx = torch.unique(grid_idx)
            # get obj_indx
            obj_idx = obj_room_indices[i]
            obj_idx = torch.unique(obj_idx)
            if grid_idx.equal(obj_idx):
                print(" they are in the same frame!!!!!!!!")
            else:
                print("No No No")
                print(f" grid_room_indices: {grid_idx} for {i} th batch")
                print(f" object_room_indices: {obj_idx} for {i} th batch")
        
        
    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            batch = self.object_pos_to_polar(batch)
            # object global pos to polar
            batch = self.object_pos_to_polar(batch)
            # reproject the patch feature
            batch = self.room_fts_reprojection(batch)
            # assign the room index to object
            batch = self.build_RoomObject_relation(batch)
            batch = self.compute_room_pos(batch)
            # cat the graph info
            self.cat_graph_info(batch)
            batch['places'] = None
            batch['places_masks'] = None
            return self.forward_mlm(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], batch['traj_view_dep_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'],
                batch['txt_labels'], 
                batch['obj_pos_fts'],batch['obj_clip_fts'],batch['obj_masks'], 
                batch['room_fts'],batch['room_pos'], batch['assigned_room_idxs'],batch['room_lens'],
                batch['graph_info'],batch['hsg_lens'],batch['batch_graph_info'],
                compute_loss
            )
        elif task.startswith('mrc'):
            return self.forward_mrc(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['vp_view_mrc_masks'], batch['vp_view_probs'], 
                batch['vp_obj_mrc_masks'], batch['vp_obj_probs'], compute_loss
            )
        elif task.startswith('sap'):
            # object global pos to polar
            batch = self.object_pos_to_polar(batch)
            # reproject the patch feature
            batch = self.room_fts_reprojection(batch)
            # assign the room index to object
            batch = self.build_RoomObject_relation(batch)
            batch = self.compute_room_pos(batch)
            batch = self.cat_graph_info(batch)
            ###
            # print("############## pretrain_cmt.py ################")
            # print(f" obj_masks shape {batch['obj_masks'].shape}")
            batch['places'] = None
            batch['places_masks'] = None
            self.check_reporjection(batch)
            return self.forward_sap(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], batch['traj_view_dep_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['gmap_visited_masks'],
                batch['global_act_labels'], batch['local_act_labels'], 
                batch['obj_pos_fts'],batch['obj_clip_fts'],batch['obj_masks'],
                batch['room_fts'],batch['room_pos'],batch['assigned_room_idxs'],batch['room_lens'],
                batch['graph_info'],batch['hsg_lens'], batch['batch_graph_info'],
                compute_loss
            )
        elif task.startswith('og'):
            return self.forward_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['obj_labels'], compute_loss
            )
        elif task.startswith('valid_sap_og'):
            return self.forward_sap_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'], batch['global_act_labels'], batch['local_act_labels'], 
                batch['obj_labels']
            )
        else:
            raise ValueError('invalid task')

    def forward_mlm(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        txt_labels, 
        object_pos_fts, obj_clip_fts, obj_masks,
        room_fts, room_pos, assigned_room_idxs, room_lens,
        graph_info,batch_hsg_lens, batch_graph_info,
        compute_loss
    ):
        txt_embeds = self.bert.forward_mlm(
            txt_ids, txt_lens, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
            object_pos_fts, obj_clip_fts, obj_masks,room_fts, 
            room_pos, assigned_room_idxs, room_lens, graph_info,batch_hsg_lens,batch_graph_info
        )

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1], reduction='none'
            )
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrc(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        vp_view_mrc_masks, vp_view_probs, vp_obj_mrc_masks, vp_obj_probs, compute_loss=True
    ):
        _, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )
        
        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens)]
        vp_view_embeds = pad_tensors_wgrad(
            [x[1:view_len+1] for x, view_len in zip(vp_embeds, vp_view_lens)]
        )   # [stop] at 0
        # vp_view_mrc_masks = vp_view_mrc_masks[:, :vp_view_embeds.size(1)]
        
        # only compute masked regions for better efficient=cy
        view_masked_output = self._compute_masked_hidden(vp_view_embeds, vp_view_mrc_masks)
        view_prediction_soft_labels = self.image_classifier(view_masked_output)
        view_mrc_targets = self._compute_masked_hidden(vp_view_probs, vp_view_mrc_masks)

        if traj_obj_img_fts is not None:
            vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens)]
            vp_obj_embeds = pad_tensors_wgrad(
                [x[view_len+1:view_len+obj_len+1] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)]
            )
            # vp_obj_mrc_masks = vp_obj_mrc_masks[:, :vp_obj_embeds.size(1)]
            obj_masked_output = self._compute_masked_hidden(vp_obj_embeds, vp_obj_mrc_masks)
            if self.obj_classifier is None:
                obj_prediction_soft_labels = self.image_classifier(obj_masked_output)
            else:
                obj_prediction_soft_labels = self.obj_classifier(obj_masked_output)
            obj_mrc_targets = self._compute_masked_hidden(vp_obj_probs, vp_obj_mrc_masks)
        else:
            obj_prediction_soft_labels, obj_mrc_targets = None, None

        if compute_loss:
            view_prediction_soft_labels = F.log_softmax(view_prediction_soft_labels, dim=-1)
            view_mrc_loss = F.kl_div(view_prediction_soft_labels, view_mrc_targets, reduction='none').sum(dim=1)
            if obj_prediction_soft_labels is None:
                mrc_loss = view_mrc_loss
            else:
                obj_prediction_soft_labels = F.log_softmax(obj_prediction_soft_labels, dim=-1)
                obj_mrc_loss = F.kl_div(obj_prediction_soft_labels, obj_mrc_targets, reduction='none').sum(dim=1)
                mrc_loss = torch.cat([view_mrc_loss, obj_mrc_loss], 0)
            return mrc_loss
        else:
            return view_prediction_soft_labels, view_mrc_targets, obj_prediction_soft_labels, obj_mrc_targets

    def forward_sap(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        gmap_visited_masks, global_act_labels, local_act_labels,
        obj_pos_fts, obj_clip_fts, obj_masks, 
        room_fts, room_pos, assigned_room_idxs, room_lens,graph_info,batch_hsg_lens,batch_graph_info,
        compute_loss
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
            obj_pos_fts, obj_clip_fts, obj_masks, room_fts, room_pos, assigned_room_idxs, room_lens,
            graph_info,batch_hsg_lens,batch_graph_info
        )
        
        # if self.sap_fuse_linear is None:
        #     fuse_weights = 0.5
        # else:
        #     fuse_weights = torch.sigmoid(self.sap_fuse_linear(
        #         torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
        #     ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2)
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        # local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        # vp_nav_masks = pad_tensors_wgrad(
        #     [x[-1]!=1 for x in torch.split(traj_nav_types, traj_step_lens)]
        # )[:, :local_logits.size(1)-1]
        # vp_nav_masks = torch.cat(
        #     [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        # )   # add [stop]
        # local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        # fused_logits = torch.clone(global_logits)
        # fused_logits[:, 0] += local_logits[:, 0]   # stop
        # for i in range(batch_size):
        #     visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
        #     tmp = {}
        #     bw_logits = 0
        #     for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
        #         if cand_vpid in visited_nodes:
        #             bw_logits += local_logits[i, j+1]
        #         else:
        #             tmp[cand_vpid] = local_logits[i, j+1]
        #     for j, vp in enumerate(gmap_vpids[i]):
        #         if j > 0 and vp not in visited_nodes:
        #             if vp in tmp:
        #                 fused_logits[i, j] += tmp[vp]
        #             else:
        #                 fused_logits[i, j] += bw_logits

        if compute_loss:
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            # local_losses = F.cross_entropy(local_logits, local_act_labels, reduction='none')
            # fused_losses = F.cross_entropy(fused_logits, global_act_labels, reduction='none')
            losses = global_losses
            return losses
        else:
            return global_logits,  global_act_labels

    def forward_og(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        obj_labels, compute_loss
    ):
        gmap_embeds, vp_embeds = self.bert.forward(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1+view_len: 1+view_len+obj_len] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        if compute_loss:
            losses = F.cross_entropy(obj_logits, obj_labels, reduction='none')
            return losses
        else:
            return obj_logits

    def forward_sap_og(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        gmap_visited_masks, global_act_labels, local_act_labels, obj_labels
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )
        
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1]!=1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1)-1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )   # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j+1]
                else:
                    tmp[cand_vpid] = local_logits[i, j+1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1+view_len: 1+view_len+obj_len] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))
        
        return global_logits, local_logits, fused_logits, obj_logits
