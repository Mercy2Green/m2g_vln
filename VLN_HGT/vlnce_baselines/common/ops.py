import torch

from .transformer import TransformerEncoder, TransformerEncoderLayer

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm

def create_transformer_encoder(config, num_layers, norm=False):
    enc_layer = TransformerEncoderLayer(
        config.hidden_size, config.num_attention_heads,
        dim_feedforward=config.intermediate_size, 
        dropout=config.hidden_dropout_prob,
        activation=config.hidden_act,
        normalize_before=True
    )
    if norm:
        norm_layer = BertLayerNorm(config.hidden_size, eps=1e-12)
    else:
        norm_layer = None
    return TransformerEncoder(enc_layer, num_layers, norm=norm_layer, batch_first=True)

def extend_neg_masks(masks, dtype=None):
    """
    mask from (N, L) into (N, 1(H), 1(L), L) and make it negative
    """
    if dtype is None:
        dtype = torch.float
    extended_masks = masks.unsqueeze(1).unsqueeze(2)
    extended_masks = extended_masks.to(dtype=dtype)
    extended_masks = (1.0 - extended_masks) * -10000.0
    return extended_masks

def gen_seq_masks(seq_lens, max_len=None):
    if max_len is None:
        max_len = max(seq_lens)
    batch_size = len(seq_lens)
    device = seq_lens.device

    masks = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    masks = masks < seq_lens.unsqueeze(1)
    return masks

def pad_tensors_wgrad(tensors, lens=None):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    batch_size = len(tensors)
    hid = list(tensors[0].size()[1:])

    device = tensors[0].device
    dtype = tensors[0].dtype

    output = []
    for i in range(batch_size):
        if lens[i] < max_len:
            tmp = torch.cat(
                [tensors[i], torch.zeros([max_len-lens[i]]+hid, dtype=dtype).to(device)],
                dim=0
            )
        else:
            tmp = tensors[i]
        output.append(tmp)
    output = torch.stack(output, 0)
    return output
def pad_tensors_2dim(tensors):
    # 计算最大长度
    max_len_1 = max(t.size(0) for t in tensors)
    max_len_2 = max(t.size(1) for t in tensors)
    # 假设所有输入张量的第三个维度大小相同，获取第一个张量的第三维大小
    third_dim_size = tensors[0].size(2) if tensors else 0
    
    # 创建新的tensor并复制原始数据
    padded_tensors = []
    for t in tensors:
        # 创建新的tensor，初始化为0，动态设置第三个维度的大小
        padded_tensor = torch.zeros(max_len_1, max_len_2, third_dim_size, dtype=t.dtype, device=t.device)
        # 复制原始数据到新的tensor中
        padded_tensor[:t.size(0), :t.size(1), :] = t
        padded_tensors.append(padded_tensor)
    
    # 使用torch.stack将列表中的tensors合并为一个新的tensor
    return torch.stack(padded_tensors)

# 注意：这个函数现在返回一个单一的tensor，其中包含所有经过padding的tensors
