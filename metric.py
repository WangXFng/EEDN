import math
import torch
import Constants as C


def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    map = 0
    ap = 0

    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)
    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        map = ap / count

    return float(count / k), float(count / len(test_matrix)), map, float(dcg_k / idcg_k)


def vaild(prediction, label, top_n, pre, rec, map_, ndcg):
    top_ = torch.topk(prediction, top_n, -1, sorted=True)[1]
    for top, l in zip(top_, label):
        try:
            l = l[l != 0] - 1
        except Exception as e:
            l = l[l != 0]
        recom_list, ground_list = top.cpu().numpy(), l.cpu().numpy()
        if len(ground_list) == 0:
            continue
        # map2, mrr, ndcg2 = metric.map_mrr_ndcg(recom_list, ground_list)
        pre2, rec2, map2, ndcg2 = precision_recall_ndcg_at_k(top_n, recom_list, ground_list)
        pre.append(pre2), rec.append(rec2), map_.append(map2), ndcg.append(ndcg2)


def pre_rec_top(pre, rec, map_, ndcg, prediction, label, event_type):

    # filter out the visited POI
    target_ = torch.ones(event_type.size()[0], C.POI_NUMBER, device='cuda:0', dtype=torch.double)
    for i, e in enumerate(event_type):
        e = e[e!=0]-1
        target_[i][e] = 0
    prediction = prediction * target_

    for i, topN in enumerate([1, 5, 10, 20]):
        vaild(prediction, label, topN, pre[i], rec[i], map_[i], ndcg[i])


# def map_mrr_ndcg(rankedlist, test_matrix):
#     ap = 0
#     map = 0
#     dcg = 0
#     idcg = 0
#     mrr = 0
#     for i in range(len(test_matrix)):
#         idcg += 1 / math.log(i + 2, 2)
#
#     b1 = rankedlist
#     b2 = test_matrix
#     s2 = set(b2)
#     hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
#     count = len(hits)
#
#     for c in range(count):
#         ap += (c + 1) / (hits[c][0] + 1)
#         dcg += 1 / math.log(hits[c][0] + 2, 2)
#
#     if count != 0:
#         mrr = 1 / (hits[0][0] + 1)
#
#     if count != 0:
#         map = ap / count
#
#     return map, mrr, float(dcg / idcg)
