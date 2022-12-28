import argparse
import scipy.sparse as sp
import torch.nn
import torch.nn as nn
import numpy as np

import os
import Utils
from Utils import *
from gMLP.gmlp import gMLP
from hGCN.hGCN import hGCNEncoder
from transformer.Layers import EncoderLayer, MultiHeadAttention
from transformerls.lstransformer import TransformerLS
from preprocess.cal_poi_pairwise import read_interaction

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


class Encoder(nn.Module):
    def __init__(
            self,
            num_types, d_model, n_layers, n_head, dropout):
        super().__init__()
        self.d_model = d_model

        if C.ENCODER == 'gMLP':
            self.gmlp = gMLP(d_model, 512, 700, n_layers)

        if C.ENCODER == 'hGCN':
            directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
            train_file = 'poi_matrix.npy'
            if not os.path.exists(directory_path + train_file):
                print('Poi_matrix is not found, generating ...')
                read_interaction()
            print('Loading ', directory_path + train_file, '...')
            self.ui_adj = np.load(directory_path + train_file)
            self.ui_adj = sp.csr_matrix(self.ui_adj)
            print('Computing adj matrix ...')
            self.ui_adj = torch.tensor(self.normalize_graph_mat(self.ui_adj).toarray(), device='cuda:0')

            self.layer_stack = nn.ModuleList([
                hGCNEncoder(d_model, n_head)
                for _ in range(n_layers)])

        if C.ENCODER == 'Transformer':
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_model, 512, n_head, 512, 512, dropout=dropout)  # 512 1024 4 512 512 M
                for _ in range(n_layers)])

        if C.ENCODER == 'LSTransformer':
            self.layer_stack = nn.ModuleList([
                TransformerLS(d_model, n_head, dropout)
                for _ in range(n_layers)])

        if C.ENCODER == 'None':
            self.user_emb = torch.nn.Embedding(C.USER_NUMBER, d_model, padding_idx=C.PAD)

    def forward(self, user_id, event_type, enc_output, slf_attn_mask, non_pad_mask):
        """ Encode event sequences via masked self-attention. """
        if C.ENCODER == 'None':
            return self.user_emb(user_id)

        if C.ENCODER == 'Transformer':

            for enc_layer in self.layer_stack:
                # if C.DATASET == 'Gowalla':
                residual = enc_output
                enc_output, _ = enc_layer(
                    enc_output,
                    non_pad_mask=non_pad_mask,  # non_pad_mask
                    slf_attn_mask=slf_attn_mask,  # slf_attn_mask
                )
                # if C.DATASET == 'Gowalla':
                enc_output += residual


        elif C.ENCODER == 'gMLP':
            enc_output = self.gmlp(enc_output)

        elif C.ENCODER == 'hGCN':
            # get individual adj
            # residual = enc_output
            adj = torch.zeros((event_type.size(0), event_type.size(1), event_type.size(1)), device='cuda:0')
            for i, e in enumerate(event_type):
                adj[i] = self.ui_adj[e-1, e-1]

            for enc_layer in self.layer_stack:
                enc_output = enc_layer(enc_output, adj, event_type)
            # enc_output += residual

        if C.ENCODER == 'LSTransformer':
            for enc_layer in self.layer_stack:
                residual = enc_output
                enc_output = enc_layer(
                    enc_output,
                    mask=non_pad_mask
                )
                enc_output += residual

        return enc_output.mean(1)
        # return self.conv3(enc_output.unsqueeze(1)).squeeze(1).squeeze(1)

    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        rowsum[rowsum==0] = 1e-9
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.dropout = nn.Dropout(0.5)
        self.temperature = 512 ** 0.5

        # self.conv = torch.nn.Conv2d(1, 1, (3, 3), padding=1, padding_mode='zeros')
        self.conv = torch.nn.Conv2d(1, 1, (9, 9), padding=4, padding_mode='zeros')
        self.conv3 = torch.nn.Conv2d(1, 1, (700, 1))

        self.implicit_graph_features = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.implicit_graph_features.weight)

        self.implicit_conv_features = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.implicit_conv_features.weight)

        self.implicit_att_features = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.implicit_att_features.weight)

    def forward(self, user_embeddings, embeddings, enc_output, slf_attn_mask):
        outputs = []
        if C.ABLATION != 'w/oMatcher':
            out = user_embeddings.matmul(embeddings.T[:,1:])
            out = F.normalize(out, p=2, dim=-1, eps=1e-05)
            outputs.append(out)

        if C.ABLATION != 'w/oImFe':

            # if C.ABLATION != "w/oGraIm":
            if C.ABLATION != "w/oGraIm":
                if not (C.ENCODER == "Transformer" and C.DATASET == 'Yelp2018'):
                    # graph implicit
                    graph_implicit = self.implicit_graph_features(user_embeddings)
                    graph_implicit = F.normalize(graph_implicit, p=2, dim=-1, eps=1e-05)
                    outputs.append(graph_implicit)

            if C.ABLATION != "w/oAtt":
                # seq1 implicit
                attn = torch.matmul(enc_output / self.temperature, enc_output.transpose(1, 2))
                attn = self.dropout(torch.tanh(attn)) * slf_attn_mask
                seq1_implicit = torch.matmul(attn, enc_output)
                seq1_implicit = self.implicit_att_features(seq1_implicit.mean(1))
                seq1_implicit = F.normalize(seq1_implicit, p=2, dim=-1, eps=1e-05)
                outputs.append(seq1_implicit/2)

            if C.ABLATION != "w/oConv":
                # seq2 implicit
                seq2_implicit = self.conv(enc_output.unsqueeze(1))
                seq2_implicit = self.conv3(seq2_implicit)
                seq2_implicit = self.implicit_conv_features(seq2_implicit.squeeze(1).squeeze(1))
                seq2_implicit = F.normalize(seq2_implicit, p=2, dim=-1, eps=1e-05)
                outputs.append(seq2_implicit*2)

        outputs = torch.stack(outputs, dim=0).sum(0)
        out = torch.tanh(outputs)
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        # self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.rnn = nn.GRU(d_model, d_rnn, num_layers=1, batch_first=True)  # input_size: d_model, gate_size: 4 * d_rnn
        self.projection = nn.Linear(d_rnn, d_model)  # in_features: int d_rnn, out_features: int d_model

    def forward(self, data, non_pad_mask):

        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)

        return out


class Model(nn.Module):
    def __init__(
            self,  # conf, training_set, test_set,
            num_types, d_model=256, n_layers=4, n_head=4, dropout=0.1, device=0):
        super(Model, self).__init__()

        # event(POI) type embedding (23 512) (K M)
        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=C.PAD)  # dding 0

        # # set the embeddings of POI (index=0, PAD value) to being the vector fully filled in 0
        # weight_ = self.event_emb.weight.detach()
        # weight_[0] = 0
        # self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=C.PAD, _weight=weight_).to('cuda:0')  # dding 0

        self.encoder = Encoder(
            num_types=num_types, d_model=d_model,
            n_layers=n_layers, n_head=n_head, dropout=dropout)

        self.num_types = num_types

        # # OPTIONAL recurrent layer, this sometimes helps
        # self.rnn = RNN_layers(d_model, d_rnn)

        self.predictor = Predictor(d_model, num_types)

    def forward(self, user_id, event_type):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                user_embeddings: batch.
        """
        slf_attn_mask_subseq = get_subsequent_mask(event_type)  # M * L * L
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)  # M x lq x lk
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        non_pad_mask = get_non_pad_mask(event_type)

        # (K M)  event_emb: Embedding
        enc_output = self.event_emb(event_type)

        user_embeddings = self.encoder(user_id, event_type, enc_output, slf_attn_mask, non_pad_mask)  # H(j,:)

        # enc_output = self.rnn(enc_output, non_pad_mask)  # [16, 166, 512]

        prediction = self.predictor(user_embeddings, self.event_emb.weight, enc_output, slf_attn_mask)
        # prediction = torch.squeeze(prediction, 1) * candidates

        return prediction, user_embeddings  # , candidates_enc  # , mage_output,  phase_output
