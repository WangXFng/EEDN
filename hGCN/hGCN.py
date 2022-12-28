import torch
import torch.nn as nn
import Constants as C
import torch.nn.functional as F


class hGCNEncoder(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()

        self.head = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_head)
        ])

    def get_non_pad_mask(self, seq):
        """ Get the non-padding positions. """
        return seq.ne(C.PAD).type(torch.float).unsqueeze(-1)

    def forward(self, output, sparse_norm_adj, event_type):
        output = output * self.get_non_pad_mask(event_type)

        outputs = []
        for linear in self.head:
            if C.ABLATION != 'w/oFeTra' and C.DATASET != 'Gowalla':
            # if C.ABLATION != 'w/oFeTra':
                output = linear(output)
            if C.ABLATION != 'w/oGlobal':
                output = torch.matmul(sparse_norm_adj, F.elu(output))
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)

        return outputs.sum(0)



