import torch
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
# 示例数据
pc = torch.tensor([[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]])
padded_places = torch.tensor([[[1.0, 2.0, 3.0, 0], [7.0, 8.0, 9.0, 1]]])
pc_masks = torch.tensor([[True, True]])
place_masks = torch.tensor([[True, True]])

# 调用函数
min_dist, assigned_room_idxs = assign_RoomIndx(None, pc, padded_places, pc_masks, place_masks)

print("min_dist:", min_dist)
print("assigned_room_idxs:", assigned_room_idxs)