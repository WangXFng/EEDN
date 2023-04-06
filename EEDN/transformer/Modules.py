import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, n_head, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.n_head = n_head
        self.dropout = nn.Dropout(attn_dropout)

        # self.b = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        # self.b.data.fill_(1e-5)
        # self.c = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        # self.c.data.fill_(1e-1)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # [32, 8, 85, 512]  [32, 8, 85, 512]

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)
        return output, attn
