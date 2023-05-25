import argparse
import os
import dgl

import dgl.nn.pytorch as layer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.data import load_data, register_data_args
from dgl.nn import DotGatConv
from dgl.nn.pytorch import GraphConv
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from sklearn.model_selection import train_test_split
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss


class InnerProductDecoder(nn.Module):
    """
    InnerProduct Decoder Layer
    """

    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj


class WeightedGraphConv(GraphConv):
    """
    An encapsulation of dgl GraphConv model to adapt edge weights
    """

    def edge_selection_simple(self, edges):
        return {'m': edges.src['h'] * edges.data['weight']}

    def forward(self, graph, features, weight=None, agg='sum'):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise ValueError('There are 0-in-degree nodes in the graph, '
                                     'output for those nodes will be invalid. '
                                     'This is harmful for some applications, '
                                     'causing silent performance regression. '
                                     'Adding self-loop on the input graph by '
                                     'calling `g = dgl.add_self_loop(g)` will resolve '
                                     'the issue. Setting ``allow_zero_in_degree`` '
                                     'to be `True` when constructing this module will '
                                     'suppress the check and let the code run.')
            features_src, features_dst = expand_as_pair(features, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (features_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                features_src = features_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise ValueError('External weight is provided while at the same time the'
                                     ' module has defined its own weight parameter. Please'
                                     ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                features_src = torch.matmul(features_src, weight)
            graph.srcdata['h'] = features_src
            if agg == "sum":
                graph.update_all(self.edge_selection_simple, fn.sum(msg='m', out='h'))
            if agg == "mean":
                graph.update_all(self.edge_selection_simple, fn.mean(msg='m', out='h'))

            rst = graph.dstdata['h']
            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (features_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GCNAE(nn.Module):
    """
    autoencoder
    """

    def __init__(self, in_features, n_hidden, n_layers,
                 activation=None, norm=None, dropout=0.1,
                 hidden=None, hidden_relu=False, hidden_bn=False,
                 agg="sum"):
        """

        :param in_features:
        :param n_hidden:
        :param n_layers:
        :param activation:
        :param norm:
        :param dropout:
        :param hidden:
        :param hidden_relu:
        :param hidden_bn:
        :param agg:
        """
        super(GCNAE, self).__init__()
        self.agg = agg
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layer1 = WeightedGraphConv(in_feats=in_features, out_feats=n_hidden, activation=activation)
        if n_layers == 2:
            self.layer2 = WeightedGraphConv(in_feats=n_hidden, out_feats=n_hidden, activation=activation)
        self.decoder = InnerProductDecoder(activation=lambda x: x)
        self.hidden = hidden
        if hidden is not None:
            enc = []
            for i, s in enumerate(hidden):
                if i == 0:
                    enc.append(nn.Linear(n_hidden, hidden[i]))
                else:
                    enc.append(nn.Linear(hidden[i - 1], hidden[i]))
                if hidden_bn and i != len(hidden):
                    enc.append(nn.BatchNorm1d(hidden[i]))
                if hidden_relu and i != len(hidden):
                    enc.append(nn.ReLU())
            self.encoder = nn.Sequential(*enc)

    def forward(self, blocks, features):
        x = blocks[0].srcdata['features']
        for i in range(len(blocks)):
            with blocks[i].local_scope():
                if self.dropout is not None:
                    x = self.dropout(x)
                blocks[i].srcdata['h'] = x
                if i == 0:
                    x = self.layer1(blocks[i], x, agg=self.agg)
                else:
                    x = self.layer2(blocks[i], x, agg=self.agg)
        if self.hidden is not None:
            x = self.encoder(x)
        adj_rec = self.decoder(x)
        return adj_rec, x


class WeightedGraphConvAlpha(GraphConv):
    """
    An encapsulation of the dgl GraphConv model to learn the extra edge weight parameter
    """

    def edge_selection_simple(self, edges):
        number_of_edges = edges.src['h'].shape[0]
        indices = np.expand_dims(
            np.array([self.gene_num + 1] * number_of_edges, dtype=np.int32), axis=1)
        src_id, dst_id = edges.src['id'].cpu(
        ).numpy(), edges.dst['id'].cpu().numpy()
        indices = np.where((src_id >= 0) & (dst_id < 0),
                           src_id, indices)  # gene->cell
        indices = np.where((dst_id >= 0) & (src_id < 0),
                           dst_id, indices)  # cell->gene
        indices = np.where((dst_id >= 0) & (src_id >= 0),
                           self.gene_num, indices)  # gene-gene
        h = edges.src['h'] * self.alpha[indices.squeeze()]
        return {'m': h}

    #         return {'m': h * edges.data['weight']}

    def forward(self, graph, feat, weight=None, alpha=None, gene_num=None):
        self.alpha = alpha
        self.gene_num = gene_num
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise ValueError('There are 0-in-degree nodes in the graph, '
                                     'output for those nodes will be invalid. '
                                     'This is harmful for some applications, '
                                     'causing silent performance regression. '
                                     'Adding self-loop on the input graph by '
                                     'calling `g = dgl.add_self_loop(g)` will resolve '
                                     'the issue. Setting ``allow_zero_in_degree`` '
                                     'to be `True` when constructing this module will '
                                     'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            #             print(f"feat_src : {feat_src.shape}, feat_dst {feat_dst.shape}")
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise ValueError('External weight is provided while at the same time the'
                                     ' module has defined its own weight parameter. Please'
                                     ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                feat_src = torch.matmul(feat_src, weight)
            graph.srcdata['h'] = feat_src
            graph.update_all(self.edge_selection_simple,
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            if self._norm != 'none':

                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GCNAEAlpha(nn.Module):
    """
    Graph autoencoder learning additional edge weight parameter, as proposed in scDeepSort
    """

    def __init__(self, in_feats, n_hidden, n_layers, gene_num,
                 activation=None, norm=None, dropout=0.1,
                 hidden=None,
                 hidden_relu=False,
                 hidden_bn=False):
        super(GCNAEAlpha, self).__init__()
        self.gene_num = gene_num
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layer1 = WeightedGraphConvAlpha(
            in_feats=in_feats, out_feats=n_hidden, activation=activation)
        self.alpha = nn.Parameter(torch.tensor(
            [1] * (gene_num + 2), dtype=torch.float32).unsqueeze(-1))

        self.decoder = InnerProductDecoder(activation=lambda x: x)
        self.hidden = hidden
        if hidden is not None:
            enc = []
            for i, s in enumerate(hidden):
                if i == 0:

                    enc.append(nn.Linear(n_hidden, hidden[i]))
                else:

                    enc.append(nn.Linear(hidden[i - 1], hidden[i]))
                if hidden_bn and i != len(hidden):
                    enc.append(nn.BatchNorm1d(hidden[i]))
                if hidden_relu and i != len(hidden):
                    enc.append(nn.ReLU())
            #             print(enc)
            self.encoder = nn.Sequential(*enc)

    def forward(self, blocks, features):
        x = blocks[0].srcdata['features']
        for i in range(len(blocks)):
            with blocks[i].local_scope():
                if self.dropout is not None:
                    x = self.dropout(x)
                blocks[i].srcdata['h'] = x
                if i == 0:
                    x = self.layer1(
                        blocks[i], x, alpha=self.alpha, gene_num=self.gene_num)
                else:
                    x = self.layer2(
                        blocks[i], x, alpha=self.alpha, gene_num=self.gene_num)
        if self.hidden is not None:
            x = self.encoder(x)
        adj_rec = self.decoder(x)
        return adj_rec, x


class GAE(nn.Module):
    """
    Generic implementation of Graph Autoencoder, supporting several dgl models
    [GraphConv, GATConv, EdgeConv, SAGEConv, GIN]
    """

    def __init__(self, in_feats, n_hidden, n_layers,
                 activation=None, norm=None, dropout=0.1,
                 hidden=None,
                 hidden_relu=False,
                 hidden_bn=False,
                 model="GraphConv"):
        super(GAE, self).__init__()
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if model == "GraphConv":
            self.layer1 = GraphConv(in_feats=in_feats, out_feats=n_hidden,
                                    activation=activation)
        if model == "GATConv":
            self.layer1 = dgl.nn.GATConv(in_feats=in_feats,
                                         out_feats=n_hidden, activation=activation, num_heads=3)

        if model == "EdgeConv":
            self.layer1 = dgl.nn.EdgeConv(in_feats, n_hidden)

        if model == "SAGEConvMean":
            self.layer1 = dgl.nn.SAGEConv(in_feats, n_hidden, 'mean')
        if model == "SAGEConvGCN":
            self.layer1 = dgl.nn.SAGEConv(in_feats, n_hidden, 'gcn')
        if model == "SAGEConvPool":
            self.layer1 = dgl.nn.SAGEConv(
                in_feats, n_hidden, 'pool')
        if model == "GINMax":
            lin = torch.nn.Linear(in_feats, n_hidden)
            self.layer1 = dgl.nn.GINConv(lin, 'max')

        if model == "GINSum":
            lin = torch.nn.Linear(in_feats, n_hidden)
            self.layer1 = dgl.nn.GINConv(lin, 'sum')
        if model == "GINMean":
            lin = torch.nn.Linear(in_feats, n_hidden)
            self.layer1 = dgl.nn.GINConv(lin, 'mean')

        self.decoder = InnerProductDecoder(activation=lambda x: x)
        self.hidden = hidden
        if hidden is not None:
            enc = []
            for i, _ in enumerate(hidden):
                if i == 0:

                    enc.append(nn.Linear(n_hidden, hidden[i]))
                else:

                    enc.append(nn.Linear(hidden[i - 1], hidden[i]))
                if hidden_bn and i != len(hidden):
                    enc.append(nn.BatchNorm1d(hidden[i]))
                if hidden_relu and i != len(hidden):
                    enc.append(nn.ReLU())
            print(enc)
            self.encoder = nn.Sequential(*enc)

    def forward(self, blocks, features):
        x = blocks[0].srcdata['features']
        for i in range(len(blocks)):
            with blocks[i].local_scope():
                if self.dropout is not None:
                    x = self.dropout(x)
                blocks[i].srcdata['h'] = x
                if i == 0:
                    x = self.layer1(blocks[i], x)
                else:
                    x = self.layer2(blocks[i], x)
        x = x.view(x.shape[0], -1)
        if self.hidden is not None:
            x = self.encoder(x)
        adj_rec = self.decoder(x)
        return adj_rec, x
