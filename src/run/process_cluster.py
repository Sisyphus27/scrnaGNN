import sys

sys.path.append("..")
import argparse
import numpy as np
import dgl
from dgl import DGLGraph
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from collections import Counter
import pickle
import h5py
import random
import glob2
import seaborn as sns

import src.train
import src.models

from src.utils import embed2swift

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = src.train.get_device()

category = "real_data"

epochs = 10
batch_size = 128
pca_size = 50
path = "./"
files = glob2.glob(f'{path}/datasets/real_data/*.h5')
files = [f[len(f"'{path}/datasets/real_data"):-3] for f in files]
# files = files[:3]
print(files)

results = pd.DataFrame()
model_name = "GraphConv"
normalize_weights = "log_per_cell"
node_features = "scale"
same_edge_values = False
edge_norm = True
hidden_relu = False
hidden_bn = False
n_layers = 1
hidden_dim = 200
hidden = [100]
nb_genes = 3000
activation = F.relu
files = files[-1]
for _, dataset in enumerate([files]):
    print(f">> {dataset}")

    data_mat = h5py.File(f"{path}/datasets/real_data/{dataset}.h5", "r")

    Y = np.array(data_mat['Y'])
    X = np.array(data_mat['X'])
    n_clusters = len(np.unique(Y))

    genes_idx, cells_idx = src.train.filter_data(X, highly_genes=nb_genes)
    X = X[cells_idx][:, genes_idx]
    Y = Y[cells_idx]

    t0 = time.time()
    graph = src.train.make_graph(
        X,
        Y,
        dense_dim=pca_size,
        node_features=node_features,
        normalize_weights=normalize_weights,
    )
    graph = graph.to(device)

    labels = graph.ndata["label"]
    train_ids = np.where(labels.cpu() != -1)[0]
    train_ids = torch.from_numpy(train_ids).to(device)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)

    dataloader = dgl.dataloading.DataLoader(
        graph,
        train_ids,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    print(
        f"INPUT: {model_name}  {hidden_dim}, {hidden}, {same_edge_values}, {edge_norm}"
    )
    t1 = time.time()

    model = src.models.GCNAE(
        in_features=pca_size,
        n_hidden=hidden_dim,
        n_layers=n_layers,
        activation=activation,
        dropout=0.1,
        hidden=hidden,
        hidden_relu=hidden_relu,
        hidden_bn=hidden_bn,
    ).to(device)

    for run in range(3):
        t_start = time.time()
        torch.manual_seed(run)
        torch.cuda.manual_seed_all(run)
        np.random.seed(run)
        random.seed(run)

        if run == 0:
            print(f">", model)

        optim = torch.optim.Adam(model.parameters(), lr=1e-5)

        scores, z = src.train.train(model,
                                    optim,
                                    epochs,
                                    dataloader,
                                    n_clusters,
                                    plot=True,
                                    save=True)
        scores["dataset"] = dataset
        scores["run"] = run
        scores["nb_genes"] = nb_genes
        scores["hidden"] = str(hidden)
        scores["hidden_dim"] = str(hidden_dim)
        scores["tot_kmeans_time"] = (t1 - t0) + (
                scores['ae_end'] - t_start) + scores['kmeans_time']
        # scores["tot_leiden_time"] = (t1 - t0) + (
        #         scores['ae_end'] - t_start) + scores['leiden_time']
        scores["time_graph"] = t1 - t0
        scores["time_training"] = (scores['ae_end'] - t_start)
        # embed2swift(scores["features"], scores["y"], dateset_name=scores["dataset"] + "_" + str(run))
        results = results.append(scores, ignore_index=True)

        #         results.to_pickle(
        #             f"../output/pickle_results/{category}/{category}_gae.pkl")
        #         print("Done")
    results.to_pickle(
        f"./output/{category}/{dataset}_gae.pkl")
print("Done")
