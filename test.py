import numpy as np
import scipy.sparse as sp

def normalize_graph_mat(adj_mat):
    shape = adj_mat.get_shape()
    # shape = adj_mat.shape()
    rowsum = np.array(adj_mat.sum(1))
    rowsum[rowsum == 0] = 1e-9
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



adj = np.array([
    [1, 0, 1],
    [0, 0, 1],
    [1, 1, 1],
])
adj = sp.csr_matrix(adj)
print(normalize_graph_mat(adj).todense())

print(r'w/o Seq$_1$')