from torch.nn.attention.flex_attention import create_block_mask
import torch
from einops import rearrange
import torch.nn.functional as F

def create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding = False):

    def create_mac_mask(_, __, q_idx, kv_idx):
        is_persist_mem = kv_idx < persist_mem_len + seq_len
        causal_mask = q_idx >= kv_idx

        if sliding:
            causal_mask &= (q_idx - kv_idx) < window_size
        
        return is_persist_mem | causal_mask

    compiled_cbm = torch.compile(create_block_mask)
    block_mask = compiled_cbm(
        create_mac_mask,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )
    return block_mask

def create_mag_block_mask(seq_len, window_size, persist_mem_len, sliding=False):

    def create_mag_mask(_, __, q_idx, kv_idx):
        is_persist_mem = kv_idx < persist_mem_len

        causal_mask = q_idx >= kv_idx

        if sliding:
            sliding_mask = (q_idx - kv_idx) < window_size
            causal_mask = causal_mask & sliding_mask

        # allow persistent
        return is_persist_mem | (~(is_persist_mem) & causal_mask)

    compiled_cbm = torch.compile(create_block_mask)
    block_mask = compiled_cbm(
        create_mag_mask,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )
    return block_mask

def round_up_multiple(seq: int, mult: int) -> int:
    return ((seq + mult - 1) // mult) * mult

def pad_and_segment_with_inverse(
    seq,
    segment_len,
    fold_into_batch = True,
    inverse_remove_pad = True
):
    seq_len = seq.shape[1]
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)
    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0
    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding, 0, 0))

    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> w b n d', n = segment_len)

    def inverse(out):

        if fold_into_batch:
            out = rearrange(out, 'w b n d -> b (w n) d')

        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]

        return out

    return seq, inverse