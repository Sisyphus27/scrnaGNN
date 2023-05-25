import csv
import os

import h5py
import numpy as np
import pandas as pd
import pandas

_color_set = (
    '#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#e74c3c', '#f1c40f','#FF0000',
'#00FF00','#0000FF','#00FFFF','#000080','#FF8000')


def h52csv(path="./", dataset="Quake_10x_Spleen", save_label=True, save_label_color=True):
    workdir = os.path.join(path, dataset + '/')
    os.chdir(workdir)
    data_mat = h5py.File(f"{workdir}{dataset}.h5", "r")
    for group in data_mat.keys():
        print(group)
    cellnames = list(data_mat['cell_names'])
    genenames = list(data_mat['gene_names'])
    X = np.array(data_mat['X'])
    X = X.T
    Y = np.array(data_mat['Y'])
    np.savetxt(f'{dataset}_label.csv', Y, fmt='%d')
    data = pd.DataFrame(X)
    data.columns = cellnames
    new_col = pd.Series(genenames)
    data.insert(0, column=None, value=new_col)
    data.to_csv(f'{dataset}_data.csv', index=False, sep='\t')
    np.savetxt(f'{dataset}_X.csv', X, fmt='%d')
    if save_label:
        np.savetxt(f'{dataset}_label.csv', cellnames, fmt='%s')
    data_color = []

    if save_label_color:
        label_set = set(cellnames)
        label_set = tuple(label_set)
        for i in range(len(label_set)):
            if i >= len(_color_set):
                raise ValueError("cells' type bigger than default colors.")
            data_color.append([label_set[i], _color_set[i]])
        with open(f'{dataset}_label_color.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in data_color:
                writer.writerow(row)
    print(0)


fileName = '../Nestorowa_2016/data_Gottgens.tsv'
tsv_file = pd.read_csv(
    fileName,
    sep='\t',
    # header=0,
    # index_col='id'
)
# csv_file=pd.read_csv('10X_PBMC_data.csv',sep='\t')
h52csv('../', 'Quake_Smart-seq2_Lung')

print('done')
