import io, os, itertools 
import os.path as osp
from collections import defaultdict, namedtuple

import dgl, pdb
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
from dgl.data.tu import TUDataset
from scipy.sparse import linalg


def batcher():
    def batcher_dev(batch):
        graph_q, graph_k = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k
    return batcher_dev

def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label)
    return batcher_dev


#-------------- 下流任务 图分类 ---------------
def create_graph_classification_dataset(dataset_name):
    name = {
        "imdb-binary": "IMDB-BINARY",
        "imdb-multi": "IMDB-MULTI",
        "rdt-b": "REDDIT-BINARY",
        "rdt-5k": "REDDIT-MULTI-5K",
        "collab": "COLLAB",
    }[dataset_name]
    dataset = TUDataset(name)
    # pdb.set_trace()
    
    dataset.num_labels = dataset.num_labels[0]             # e.g. 3
    dataset.graph_labels = dataset.graph_labels.squeeze()  # e.g. array([0, 0, 0, ..., 2, 2, 2]) [shape: (5000,)]
     
    # pdb.set_trace()
    
    return dataset











# ------------------ 注入 trigger -----------------
def _rwr_trace_to_dgl_graph(g, seed, trace, positional_embedding_size, entire_graph=False):
    subv = torch.unique(torch.cat(trace)).tolist()
    # print(subv) #e.g. [12041, 12042, 12043, 12044, 12046, 12047, 12048, ..., 69453, 369454, 369457, 369461, 369463, 487702, 487703, 487704]
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    
    if entire_graph: # 取全图就从整个图中截取子图
        subg = g.subgraph(g.nodes())
    else:            # 否则就从这样的随机游走序列中截取子图
        subg = g.subgraph(subv) 
    # 每个随机游走的序列都会创建出 子图
    # print(subg)    # e.g. DGLGraph(num_nodes=128, num_edges=10250, ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)} edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}) DGLGraph(...) ...
    
    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    #X print(subg.ndata)
    #{'_ID': tensor([ 12073,  12041,  12042,  12043,  12044,  12046,  12047,  12048,  12050, ..., 487704]), 
    # 'pos_undirected': tensor([[-5.3846e-16,  1.0769e-15, -9.4594e-17,  ..., -5.8311e-02, -7.8533e-02, -1.1012e-01], ..., [ 6.7589e-17, -1.9662e-16, -1.5361e-17,  ...,  1.4722e-02, -9.9942e-01, -3.0169e-02]]), 
    # 'seed': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... 0, 0, 0])}
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    #X print(subg.ndata["seed"])
    return subg














def eigen_decomposision(n, k, laplacian, hidden_size, retry): #@ in _add_undirected_graph_positional_embedding()
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x

# the top eigenvectors of its normalized graph Laplacian 
def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10): #@ in _rwr_trace_to_dgl_graph()
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions. See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n    = g.number_of_nodes()
    adj  = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    # I -D A D = U ^ UT
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g