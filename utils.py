import os
import copy
import math
import pickle
import logging
import numpy as np
import networkx as nx
import scipy.sparse as sp
from copy import deepcopy
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from dgl.data.citation_graph import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset


def eval_ep_batched(logits, labels, n_pos):
    # roc-auc and ap
    roc_auc = roc_auc_score(labels, logits)
    ap_score = average_precision_score(labels, logits)
    results = {'auc': roc_auc,
               'ap': ap_score}
    # hits@K
    evaluator = Evaluator(name='ogbl-ddi')
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': logits[:n_pos],
            'y_pred_neg': logits[n_pos:],
        })[f'hits@{K}']
        results[f'hits@{K}'] = hits
    return results

def eval_ep(A_pred, edges, edges_false):
    preds = A_pred[edges.T]
    preds_neg = A_pred[edges_false.T]
    logits = np.hstack([preds, preds_neg])
    labels = np.hstack([np.ones(preds.size(0)), np.zeros(preds_neg.size(0))])
    # roc-auc and ap
    roc_auc = roc_auc_score(labels, logits)
    ap_score = average_precision_score(labels, logits)
    results = {'auc': roc_auc,
               'ap': ap_score}
    # hits@K
    evaluator = Evaluator(name='ogbl-ddi')
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': preds,
            'y_pred_neg': preds_neg,
        })[f'hits@{K}']
        results[f'hits@{K}'] = hits
    return results

def normalize_sp(adj_matrix):
    # normalize adj by D^{-1/2}AD^{-1/2} for scipy sparse matrix input
    degrees = np.array(adj_matrix.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
    degree_mat_inv_sqrt = np.nan_to_num(degree_mat_inv_sqrt)
    adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
    return adj_norm

def load_data(args, logger):
    path = args.datapath
    ds = args.dataset
    if ds.startswith('ogbl'):
        dataset = DglLinkPropPredDataset(name=ds, root=args.datapath)
        graph = dataset[0]
        adj_train = graph.adjacency_matrix(scipy_fmt='csr')
        g = nx.from_scipy_sparse_matrix(adj_train)
        print('density',nx.density(g))
        print('edges:', len(g.edges()) )
        print('nodes:', len(g.nodes()) )
        adj_train.setdiag(1)
        if 'feat' in graph.ndata:
            features = graph.ndata['feat']
            dim_feat = features.shape[-1]
        else:
            # construct one-hot degree features
            degrees = torch.LongTensor(adj_train.sum(0) - 1)
            indices = torch.cat((torch.arange(adj_train.shape[0]).unsqueeze(0), degrees), dim=0)
            features = torch.sparse.FloatTensor(indices, torch.ones(adj_train.shape[0])).to_dense().numpy()
            features = torch.Tensor(features)
        # using adj_train as adj_label as training loss is only calculated with train_pairs (excluding val/test edges and no_edges)
        adj_label = copy.deepcopy(adj_train)
        # load given train/val/test edges and no_edges
        split_edge = dataset.get_edge_split()
        val_split, test_split = split_edge["valid"], split_edge["test"]
        val_edges, val_edges_false = val_split['edge'].numpy(), val_split['edge_neg'].numpy()
        test_edges, test_edges_false = test_split['edge'].numpy(), test_split['edge_neg'].numpy()
        # get training node pairs (edges and no-edges)
        if os.path.exists(f'{path}{ds}_trainpairs.pkl'):
            train_pairs = pickle.load(open(f'{path}{ds}_trainpairs.pkl', 'rb'))
        else:
            train_mask = np.ones(adj_train.shape)
            for edges_tmp in [val_edges, val_edges_false, test_edges, test_edges_false]:
                train_mask[edges_tmp.T[0], edges_tmp.T[1]] = 0
                train_mask[edges_tmp.T[1], edges_tmp.T[0]] = 0
            train_pairs = np.asarray(sp.triu(train_mask, 1).nonzero()).T
            pickle.dump(train_pairs, open(f'{path}{ds}_trainpairs.pkl', 'wb'))
    else:
        if args.dataset in ['cora', 'citeseer', 'pubmed']:
            # adj matrix (with self-loop): sp.csr_matrix
            adj_label = pickle.load(open(f'{path}{ds}_adj.pkl', 'rb'))
            # node features: sp.lil_matrix（一种系数矩阵类型，使用list of list格式存储矩阵，可以高效进行动态修改）
            features = pickle.load(open(f'{path}{ds}_feat.pkl', 'rb'))
            if isinstance(features, sp.lil.lil_matrix):
                features= features.toarray()
            features = torch.FloatTensor(features)
            dim_feat = features.shape[-1]

        elif args.dataset == 'facebook':
            filename = f'data/{args.dataset}.txt'
            g = nx.read_edgelist(filename,create_using=nx.Graph(), nodetype = int,  data=(("weight", float),))
            adj_label = nx.adjacency_matrix(g, nodelist = sorted(g.nodes()))
            adj_label = (adj_label > 0).astype('int') # to binary

        #load tvt_edges
        tvt_edges_file = f'{args.datapath}{args.dataset}_tvtEdges_val{args.val_frac}test{args.test_frac}.pkl'
        adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_label, args.val_frac, args.test_frac, tvt_edges_file, logger)

        if args.dataset == 'facebook':
            degrees = np.array(adj_train.sum(axis=1)).flatten().astype('int')
            dim_feat = degrees.max()
            one_hot_feat = np.eye(dim_feat)[degrees - 1]
            one_hot_feat = one_hot_feat.reshape((adj_train.shape[0], dim_feat))
            features = torch.FloatTensor(one_hot_feat)

    dim_feat = features.shape[-1]
    # print('dim feature, ', dim_feat)
    return adj_label, features, dim_feat, adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_edges(adj_orig, val_frac, test_frac, filename, logger):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    if os.path.exists(filename):
        adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false = pickle.load(open(filename, 'rb'))
        logger.info(f'loaded cached val and test edges with fracs of {val_frac} and {test_frac}')
        return adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false

    # Remove diagonal elements
    adj = deepcopy(adj_orig)
    # set diag as all zero
    adj.setdiag(0) # 将对角线元素置0
    adj.eliminate_zeros() # 将矩阵中的0元素进行消除。稀疏矩阵通常包含很多0元素，通过消除可以减少矩阵的存储空间和计算复杂度
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0 #检查修改后的矩阵的对角线元素是否全部为0，这个检查确保对角线元素已经全部成功设置为0

    adj_triu = sp.triu(adj, 1) # 将原始矩阵转换为上三角矩阵
    # adj_tuple = sparse_to_tuple(adj_triu)
    # edges = adj_tuple[0]
    edges = sparse_to_tuple(adj_triu)[0] # 将上三角矩阵转换为稀疏矩阵的坐标表示形式，并提取出其中的边信息
    edges_all = sparse_to_tuple(adj)[0] # 将原始矩阵adj转换为稀疏矩阵的坐标表示形式，并提取出所有的边信息
    num_test = int(np.floor(edges.shape[0] * test_frac)) # 计算测试集合中的边数，下取整
    num_val = int(np.floor(edges.shape[0] * val_frac))# 计算验证集合中的边数，下取整

    all_edge_idx = list(range(edges.shape[0])) # 创建一个包含所有连边索引的列表
    np.random.shuffle(all_edge_idx) # 打乱
    val_edge_idx = all_edge_idx[:num_val] # 提取验证集边索引
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)] # 提取测试集边索引
    test_edges = edges[test_edge_idx] # 提取测试集边
    val_edges = edges[val_edge_idx] # 提取验证集边
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0) # 从边中删除测试集边和验证集边，得到训练集边

    noedge_mask = np.ones(adj.shape) - adj_orig # 创建了一个与原始邻接矩阵形状相同的全1矩阵，并与原始邻接矩阵进行减法操作，得到节点间不存在边的掩码矩阵
    noedges = np.asarray(sp.triu(noedge_mask, 1).nonzero()).T # 使用上三角掩码矩阵将节点间不存在边的信息提取出来，并转换为坐标表示形式
    all_edge_idx = list(range(noedges.shape[0])) # 创建了一个包含所有节点间不存在边索引的列表
    np.random.shuffle(all_edge_idx) # 打乱
    val_edge_idx = all_edge_idx[:num_val] # 验证集合和测试集合中不存在边索引
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges_false = noedges[test_edge_idx] # 验证集合和测试集合中不存在边
    val_edges_false = noedges[val_edge_idx]
    # following lines for getting the no-edges are substituted with above lines
    """
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    test_edges_false = np.asarray(test_edges_false).astype("int32")

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    val_edges_false = np.asarray(val_edges_false).astype("int32")
    """
    def ismember(a, b, tol=5): # 判断两个矩阵是否存在相同的行
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0]) # 创建一个长度为训练集中连边（正样本）数量的全1数组

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T # 将邻接矩阵转换为无向图，即将其与对应转置矩阵相加
    adj_train.setdiag(1) # 将邻接矩阵的对角线元素设置为1，以确保每个节点都和自身相连

    # get training node pairs (edges and no-edges)
    train_mask = np.ones(adj_train.shape) # 创建一个和训练集对应邻接矩阵形状相同的全1数组，用于表示训练集连边的掩码
    for edges_tmp in [val_edges, val_edges_false, test_edges, test_edges_false]: # 遍历验证集和测试集中的所有正负样本
        for e in edges_tmp:
            assert e[0] < e[1] # 节点编号升序
        train_mask[edges_tmp.T[0], edges_tmp.T[1]] = 0 # 将验证集和测试集的掩码设为0，表示这些边是训练集中的连边
        train_mask[edges_tmp.T[1], edges_tmp.T[0]] = 0
    train_pairs = np.asarray(sp.triu(train_mask, 1).nonzero()).T # 将掩码转换为稀疏矩阵坐标表示形式，并提取出其中的非0元素坐标。这些坐标表示训练集中所有可能存在的边和非边

    # cache files for future use
    pickle.dump((adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false), open(filename, 'wb'))
    logger.info(f'masked and cached val and test edges with fracs of {val_frac} and {test_frac}')

    # NOTE: all these edge lists only contain single direction of edge!
    return adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false

def cache_dataset(dataset):
    # download and cache Cora/CiteSeer/PubMed datasets
    def download_dataset(dataset):
        if dataset == 'cora':
            return CoraGraphDataset(verbose=False)
        elif dataset == 'citeseer':
            return CiteseerGraphDataset()
        elif dataset == 'pubmed':
            return PubmedGraphDataset()
        else:
            raise TypeError('unsupported dataset.')
    ds = download_dataset(dataset)
    adj = nx.to_scipy_sparse_matrix(ds._graph)
    feats = ds._g.ndata['feat']
    labels = ds._g.ndata['label']
    idx_train = np.argwhere(ds._g.ndata['train_mask'] == 1).reshape(-1)
    idx_val = np.argwhere(ds._g.ndata['val_mask'] == 1).reshape(-1)
    idx_test = np.argwhere(ds._g.ndata['test_mask'] == 1).reshape(-1)

    adj.setdiag(1)
    feats = sp.lil_matrix(feats.numpy())

    pickle.dump(adj, open(f'data/{dataset}_adj.pkl', 'wb'))
    pickle.dump(feats, open(f'data/{dataset}_feat.pkl', 'wb'))
    pickle.dump(labels, open(f'data/{dataset}_label.pkl', 'wb'))
    pickle.dump((idx_train, idx_val, idx_test), open(f'data/{dataset}_tvt-idx.pkl', 'wb'))
    print(adj.shape, (adj.nnz-adj.shape[0])/2, feats.shape, len(np.unique(labels)), len(idx_train), len(idx_val), len(idx_test))

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, lr_scheduler, *op):
        self.optimizers = op
        self.steps = 0
        self.reset_count = 0
        self.next_start_step = 10
        self.multi_factor = 2
        self.total_epoch = 0
        if lr_scheduler == 'sgdr':
            self.update_lr = self.update_lr_SGDR
        elif lr_scheduler == 'cos':
            self.update_lr = self.update_lr_cosine
        elif lr_scheduler == 'zigzag':
            self.update_lr = self.update_lr_zigzag
        elif lr_scheduler == 'none':
            self.update_lr = self.no_update

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
    def no_update(self, base_lr):
        return base_lr

    def update_lr_SGDR(self, base_lr):
        end_lr = 1e-3 # 0.001
        total_T = self.total_epoch + 1
        if total_T >= self.next_start_step:
            self.steps = 0
            self.next_start_step *= self.multi_factor
        cur_T = self.steps + 1
        lr = end_lr + 1/2 * (base_lr - end_lr) * (1.0 + math.cos(math.pi*cur_T/total_T))
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        self.total_epoch += 1
        return lr

    def update_lr_zigzag(self, base_lr):
        warmup_steps = 50
        annealing_steps = 20
        end_lr = 1e-4
        if self.steps < warmup_steps:
            lr = base_lr * (self.steps+1) / warmup_steps
        elif self.steps < warmup_steps+annealing_steps:
            step = self.steps - warmup_steps
            q = (annealing_steps - step) / annealing_steps
            lr = base_lr * q + end_lr * (1 - q)
        else:
            self.steps = self.steps - warmup_steps - annealing_steps
            lr = end_lr
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        return lr

    def update_lr_cosine(self, base_lr):
        """ update the learning rate of all params according to warmup and cosine annealing """
        # 400, 1e-3
        warmup_steps = 10
        annealing_steps = 500
        end_lr = 1e-3
        if self.steps < warmup_steps:
            lr = base_lr * (self.steps+1) / warmup_steps
        elif self.steps < warmup_steps+annealing_steps:
            step = self.steps - warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / annealing_steps))
            lr = base_lr * q + end_lr * (1 - q)
        else:
            # lr = base_lr * 0.001
            self.steps = self.steps - warmup_steps - annealing_steps
            lr = end_lr
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        return lr

# 定义一个用于创建日志记录器logger的函数get_logger(name)：将日志记录器的级别设置为 DEBUG，以便记录所有级别的日志消息。函数返回配置好的日志记录器
def get_logger(name):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if name is not None:
        fh = logging.FileHandler(f'{name}.log')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger

def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx

