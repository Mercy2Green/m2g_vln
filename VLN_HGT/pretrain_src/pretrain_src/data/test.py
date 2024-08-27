import h5py
import json
import numpy as np
# in_memory = True
# if in_memory:
#     _feature_store = {}

class test_obj_clip:
    def __init__(self):
        self.in_memory = True
        self._feature_store = {}
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
        assigned_room_idxs = assigned_room_idxs.masked_fill(~pc_masks, -1)  # 将无效对象的房间索引设置为-1或其他标志值
        #### 这里要直接转译为HGT的输入形式

        return min_dist, assigned_room_idxs   

    def get_scene_map(self, batch_obj_clip_np, batch_obj_pos_np, path_map_index):

        obj_indices = np.arange(batch_obj_clip_np.shape[0])
        valid_obs_combination = np.where(path_map_index >= 0, path_map_index, 0).astype(int)
        selected_obj_clip = batch_obj_clip_np[obj_indices, valid_obs_combination,:]
        mask = path_map_index > 0

        obj_clip = selected_obj_clip[mask]
        obj_pos = batch_obj_pos_np[mask]
        
        return obj_clip, obj_pos
    # print the number of obejcts for given scan list
    def print_obj_num(self, scan_list, cg_file_path):
        with h5py.File(cg_file_path, 'r') as f:
            for scan in scan_list:
                obj_clip = f[f'{scan}_obj_clip_ft'][...].astype(np.float32)
                print(scan, obj_clip.shape[0])
    def get_obj_list(self, scan, path, cg_file_path, path_mapping_file):
        # 将path转换为字符串形式的键，假设path是一个列表
        path_key = '_'.join(path) if isinstance(path, list) else path

        # get the path mapping from json file
        if self.in_memory and scan in self._feature_store:
            batch_obj_clips = self._feature_store[scan]["clip"]
            batch_obj_pos = self._feature_store[scan]["pos"]
            if path_key in self._feature_store[scan]:
                obj_clip = self._feature_store[scan][path_key]["clip"]
                obj_pos = self._feature_store[scan][path_key]["pos"]
            else:
                all_paths_indices = self._feature_store["all_paths_indices"]
                # get path index from a list by input the element
                paths_indices = all_paths_indices[scan]
                path_indx = paths_indices.index(path)
                # get the path_map_index
                path_map_index = paths_map_indices[path_indx]
                # get the obj_clip_np
                obj_clip, obj_pos = self.get_scene_map(batch_obj_clips, batch_obj_pos, path_map_index)
                if self.in_memory:
                    self._feature_store[scan][path_key] = {}
                    self._feature_store[scan][path_key]['clip'] = obj_clip
                    self._feature_store[scan][path_key]['pos'] = obj_pos
        else:
            # read the obj_clip from hdf5
            with h5py.File(cg_file_path, 'r') as f:
                batch_obj_clips = f[f'{scan}_obj_clip_ft'][...].astype(np.float32)
                paths_map_indices = f[f'{scan}_paths_indices'][...].astype(np.float32)
                batch_obj_pos = f[f'{scan}_obj_pos'][...].astype(np.float32)

            # read the paths_indices from json
            with open(path_mapping_file, 'r') as f:
                all_paths_indices = json.load(f)
            # get the object list and pos information
            # get path index from a list by input the element
            paths_indices = all_paths_indices[scan]
            path_indx = paths_indices.index(path)
            # get the path_map_index
            path_map_index = paths_map_indices[path_indx]
            # get the obj_clip_np
            obj_clip, obj_pos = self.get_scene_map(batch_obj_clips, batch_obj_pos, path_map_index)

            # save to the memory
            if self.in_memory:
                self._feature_store[scan] = {}
                self._feature_store[scan]['clip'] = batch_obj_clips
                self._feature_store[scan]['pos'] = batch_obj_pos
                self._feature_store[scan][path_key] = {}
                self._feature_store[scan][path_key]['clip'] = obj_clip
                self._feature_store[scan][path_key]['pos'] = obj_pos
                self._feature_store["all_paths_indices"] = all_paths_indices
        return obj_clip, obj_pos

if __name__ == '__main__':
    cg_file_path = "/home/lg1/lujia/VLN_HGT/pretrain_src/datasets/cg_data/cgobj_clip.hdf5"
    path_mapping_file = "/home/lg1/lujia/VLN_HGT/pretrain_src/datasets/cg_data/all_paths.json"
    scan = "17DRP5sb8fy"
    scans_list = ["17DRP5sb8fy","1LXtFkjw3qL","1pXnuDYAj8r","29hnd4uzFmX","2azQ1b91cZZ",
                        "2n8kARJN3HM", "5LpN3gDmAk7","5q7pvUzZiYa","759xd9YjKW5","7y3sRwLe3Va",
                        "8194nk5LbLH","82sE5b5pLXE","8WUmhLawc2A","aayBHfsNo7d","ac26ZMwG7aT",
                        "B6ByNegPMKs","b8cTxDM8gDG","cV4RVeZvu5T"]
    gt_path = [
            "10c252c90fa24ef3b698c6f54d984c5c",
            "77a1a11978b04e9cbf74914c98578ab8",
            "b185432bf33645aca813ac2a961b4140",
            "5e9f4f8654574e699480e90ecdd150c8",
            "08c774f20c984008882da2b8547850eb",
            "da5fa65c13e643719a20cbb818c9a85d"
        ]
    obj = test_obj_clip()
    # obj_clip_fts, obj_global_pos = obj.get_obj_list(scan, gt_path, cg_file_path, path_mapping_file)
    # print(obj_clip_fts.shape)
    # print(obj_global_pos.shape)
    # # print the number of obejcts for given scan list
    # obj.print_obj_num(scans_list, cg_file_path)
    import torch

    # 创建一些假数据
    B, N, P = 2, 5, 3
    pc = torch.rand(B, N, 3)  # 随机点云数据
    padded_places = torch.rand(B, P, 4)  # 随机位置数据，最后一列是房间索引
    padded_places[:, :, 3] = torch.randint(1, 5, (B, P))  # 假设房间索引是1到4

    # 创建掩码
    pc_masks = torch.ones(B, N, dtype=torch.bool)  # 假设所有点都是有效的
    place_masks = torch.tensor([[True, True, False], [True, False, True]])  # 假设某些位置是无效的

    # 实例化并调用函数
    assigned_room_idxs = obj.assign_RoomIndx(pc, padded_places, pc_masks, place_masks)
    print("Assigned Room Indices:\n", assigned_room_idxs[1])

    # 测试输出
    print("Minimum Distances:\n", assigned_room_idxs[0])

