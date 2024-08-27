import numpy as np
import scipy.sparse as sp
import torch

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def randint():
    return np.random.randint(2**32 - 1)

def map_local_to_global_indexes(layer_data, node_type='object'):
    # 首先计算每种类型的节点总数，以确定每种类型的起始索引位置
    total_counts_before_type = 0
    sorted_types = sorted(layer_data.keys())  # 确保类型的排序与feature数组中的排序一致

    # 遍历排序后的节点类型，直到找到我们关心的类型
    for _type in sorted_types:
        if _type == node_type:
            break
        total_counts_before_type += len(layer_data[_type])

    # 为node_type的每个节点创建全局索引映射
    global_indexes = np.arange(total_counts_before_type, total_counts_before_type + len(layer_data[node_type]))

    return global_indexes

def feature_OAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        # # if 'node_emb' in graph.node_feature[_type]:
        # #     feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        # else:
        #     feature[_type] = np.zeros([len(idxs), 512])
        if _type == 'object':
            if 'obj_global_pos' in graph.node_feature[_type]:
                obj_global_pos= np.array(list(graph.node_feature[_type].loc[idxs, 'obj_global_pos']), dtype=np.float)
            
            if 'clip_fts' in graph.node_feature[_type]:
                obj_clip_fts = np.array(list(graph.node_feature[_type].loc[idxs, 'clip_fts']), dtype=np.float)

            # if 'rel_pos' in graph.node_feature[_type]:
            #     pos_fts = np.array(list(graph.node_feature[_type].loc[idxs, 'rel_pos']), dtype=np.float)
    
    
    # obj_to_global = map_local_to_global_indexes(layer_data, 'object')
    # room_to_global = map_local_to_global_indexes(layer_data, 'room')
    # visited_vp_to_global = map_local_to_global_indexes(layer_data, 'visited_vp')
    # candidate_vp_to_global = map_local_to_global_indexes(layer_data, 'candidate_vp')

    # objwp_to_global = np.concatenate([obj_to_global,visited_vp_to_global,candidate_vp_to_global], axis=0)
    # # Create hgt2vpid dictionary
    # id_hgt = list(range(len(layer_data['visited_vp'] + len(layer_data['candidate_vp']))))  # Using indices as id_hgt
    # visited_vp_keys = list(layer_data['visited_vp'].keys()) 
    # candidate_vp_keys = list(layer_data['candidate_vp'].keys())
    # visited_vp_ids = graph.node_feature['visited_vp'].loc[visited_vp_keys, 'id'].values
    # candidate_vp_ids = graph.node_feature['candidate_vp'].loc[candidate_vp_keys, 'id'].values
    # vp_ids = np.concatenate([visited_vp_ids, candidate_vp_ids], axis=0)
    # hgt2vpid = {id_hgt[i]: vp_ids[i] for i in range(len(id_hgt))}

    return obj_global_pos, obj_clip_fts

