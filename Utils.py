import torch.nn.functional as F
import Constants as C
import torch


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(C.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(C.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def type_loss(prediction, label, event_time, test_label, smooth):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    prediction = torch.squeeze(prediction[:, :], 1)

    multi_hots = torch.zeros(label.size(0), C.POI_NUMBER, device='cuda:0', dtype=torch.float32)

    if C.DATASET == 'Foursquare':
        beta, lambda_ = 0.4, 0.7  # 0.4, 0.5  # 0.35, 0.5  # 0.5, 1
    elif C.DATASET == 'Gowalla':
        beta, lambda_ = 1.5, 4  # 0.38, 1  # 1.5, 4
    elif C.DATASET == 'Yelp2018':
        beta, lambda_ = 1.8, 4  # 0.35, 1  # 1, 4
    elif C.DATASET == 'douban-book':
        beta, lambda_ = 0.5, 1
    # elif C.DATASET == 'ml-1M':
    #     beta, lambda_ = 0.005, 0.001
    elif C.DATASET == 'Yelp':
        beta, lambda_ = 1, 2.4  # 0.2, 0.3  # 0.5, 1.2
    else:
        beta, lambda_ = 1, 1

    for i, (t, ti, tl) in enumerate(zip(label, event_time, test_label)):
        multi_hots[i][t[t!=0]-1], multi_hots[i][tl[tl!=0]-1] = beta, lambda_

    log_prb = F.logsigmoid(prediction)  # output [16, 161, 22]   log_prb [16, 161, 22]

    multi_hots = multi_hots * (1 - smooth) + (1 - multi_hots) * smooth / C.POI_NUMBER
    predict_loss = -(multi_hots * log_prb)  # * weight  # [16, 161, 22]

    loss = torch.sum(predict_loss)

    return loss


def mmd_loss(event_type, users_embeddings, model):
    mmd_loss = []
    for l, ue in zip(event_type, users_embeddings):
        l = l[l != 0]
        mmd_loss.append(mmd_rbf(model.event_emb(l), ue.unsqueeze(0)))

    return sum(mmd_loss) / len(mmd_loss) # / lambda_


# def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
#     pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
#     neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
#     loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
#     return torch.mean(loss)
#
