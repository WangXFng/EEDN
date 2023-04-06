import torch.nn as nn

from EEDN.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v,dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_output, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_output, enc_output, enc_output, mask=slf_attn_mask)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        return enc_output, enc_slf_attn