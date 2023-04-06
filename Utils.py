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


def type_loss(prediction, label, event_time, test_label, opt):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    prediction = torch.squeeze(prediction[:, :], 1)

    multi_hots = torch.zeros(label.size(0), C.POI_NUMBER, device='cuda:0', dtype=torch.float32)
    for i, (t, ti, tl) in enumerate(zip(label, event_time, test_label)):
        multi_hots[i][t[t!=0]-1], multi_hots[i][tl[tl!=0]-1] = opt.lambda_, opt.delta

    log_prb = F.logsigmoid(prediction)
    multi_hots = multi_hots * (1 - opt.smooth) + (1 - multi_hots) * opt.smooth / C.POI_NUMBER
    predict_loss = -(multi_hots * log_prb)

    loss = torch.sum(predict_loss)
    return loss

# def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
#     pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
#     neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
#     loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
#     return torch.mean(loss)

