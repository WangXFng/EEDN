import torch
import torch.nn as nn


from transformerls.attentionls import *


class TransformerLS(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()

        d_model = d_model
        d_inner = 512
        dropout = dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = AttentionLSEncoder(d_model, n_head, dropout)

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # self.debug = debug

        self.mlpblock = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            torch.nn.Dropout(p=dropout),
            nn.Linear(d_inner, d_model),
            torch.nn.Dropout(p=dropout)
        )

    def forward(self, X, event_emb=None, g=None, mask=None, cls_embed=None):
        if cls_embed is None:
            X = self.dropout1(self.mha(self.norm1(X), event_emb, g, mask)) + X
        else:
            if cls_embed.shape[0] == 1:
                cls_embed = cls_embed.expand(X.shape[0], -1, -1)
            X_prepend = torch.cat([cls_embed, X], dim=1)
            if self.debug:
                cls_embed = self.norm1(cls_embed)
            X = self.dropout1(self.mha(self.norm1(X), mask, cls_embed)) + X_prepend
        X = self.mlpblock(self.norm2(X)) + X
        return X