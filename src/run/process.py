import src

import os

import tarfile
import tempfile
from shutil import rmtree

_root = os.path.abspath(os.path.dirname(__file__))

def sequence():
    dataset = 'Quake_Smart-seq2_Lung'  # 10X_PBMC Nestorowa_2016 Romanov
    if dataset == 'Synthetic':
        workdir = os.path.join(_root, 'datasets/Synthetic/')
        print(workdir + f'data_synthetic.tsv')
        input_file = os.path.join(workdir, 'data_synthetic.tsv')
        label_file = os.path.join(workdir, 'cell_label.tsv')
        label_color_file = os.path.join(workdir, f'cell_label_color.tsv')
    elif dataset == 'Nestorowa_2016':
        workdir = os.path.join(_root, 'datasets/Nestorowa_2016/')
        print(workdir + f'data_Nestorowa_2016.tsv.gz')
        input_file = os.path.join(workdir, 'data_Nestorowa.tsv.gz')
        label_file = os.path.join(workdir, 'cell_label.tsv.gz')
        label_color_file = os.path.join(workdir, f'cell_label_color.tsv.gz')
    else:
        workdir = os.path.join(_root, f'datasets/real_data/{dataset}/')  # 'datasets/Nestorowa_2016/'
        print(workdir + f'data_{dataset}.csv')
        input_file = os.path.join(workdir, f'{dataset}_data.csv')
        label_file = os.path.join(workdir, f'{dataset}_label.csv')
        label_color_file = os.path.join(workdir, f'{dataset}_label_color.csv')
    temp_folder = tempfile.gettempdir()
    #
    # tar = tarfile.open(workdir + 'output/stream_result.tar.gz')  # 'output/stream_result.tar.gz'
    # tar.extractall(path=temp_folder)
    # tar.close()
    ref_temp_folder = os.path.join(temp_folder, 'stream_result')
    print(ref_temp_folder)
    # print(workdir + f'data_{dataset}.tsv.gz')
    # input_file = os.path.join(workdir, f'{dataset}_data.tsv.gz')#'data_Nestorowa.tsv.gz'
    # input_file = os.path.join(workdir, 'data_Nestorowa.tsv.gz')
    # label_file = os.path.join(workdir, f'{dataset}_label.csv')
    # label_file = os.path.join(workdir, 'cell_label.tsv.gz')
    # label_color_file = os.path.join(workdir, f'{dataset}_label_color.csv')
    # label_color_file = os.path.join(workdir, f'cell_label_color.tsv.gz')
    comp_temp_folder = os.path.join(temp_folder, f'stream_result_comp/{dataset}')

    src.set_figure_params(
        dpi=80,
        style='white',
        figsize=[5.4, 4.8],
        rc={'image.cmap': 'viridis'})
    adata = src.read(
        file_name=input_file,
        workdir=comp_temp_folder)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    src.add_cell_labels(adata, file_name=label_file)
    src.add_cell_colors(adata, file_name=label_color_file)
    src.cal_qc(adata, assay='rna')
    src.filter_features(adata, min_n_cells=5)
    src.select_variable_genes(adata, n_genes=2000, save_fig=True)
    src.select_top_principal_components(
        adata, feature='var_genes', first_pc=True, n_pc=30, save_fig=True)
    src.dimension_reduction(
        adata, method='se',
        # feature='top_pcs',
        feature='var_genes',
        n_neighbors=100,
        n_components=4,
        n_jobs=2)
    src.plot_dimension_reduction(
        adata,
        color=['label', 'n_genes'],  # 'Gata1' for 'Nestorowa_2016'
        n_components=3,
        show_graph=False,
        show_text=False,
        save_fig=True,
        fig_name=f'{dataset}_dimension_reduction.pdf')
    src.plot_visualization_2D(
        adata,
        method='umap',
        n_neighbors=100,
        color=['label', 'n_genes'],  # 'Gata1' for 'Nestorowa_2016'
        use_precomputed=False,
        save_fig=True,
        fig_name=f'{dataset}_visualization_2D.pdf')
    src.seed_elastic_principal_graph(adata, n_clusters=20)
    src.plot_dimension_reduction(
        adata,
        color=['label', 'n_genes'],  # 'Gata1' for 'Nestorowa_2016'
        n_components=2,
        show_graph=True,
        show_text=False,
        save_fig=True,
        fig_name=f'{dataset}_dr_seed.pdf')
    src.plot_branches(
        adata,
        show_text=True,
        save_fig=True,
        fig_name=f'{dataset}_branches_seed.pdf')
    # TODO: this function have a bug with R$
    src.elastic_principal_graph(
        adata,
        epg_alpha=0.01,
        epg_mu=0.05,
        epg_lambda=0.01)
    src.plot_dimension_reduction(
        adata,
        color=['label', 'n_genes'],  # 'Gata1' for 'Nestorowa_2016'
        n_components=2,
        show_graph=True,
        show_text=False,
        save_fig=True,
        fig_name=f'{dataset}_dr_epg.pdf')
    src.plot_branches(
        adata,
        show_text=True,
        save_fig=True,
        fig_name=f'{dataset}_branches_epg.pdf')
    # Extend leaf branch to reach further cells
    # TODO: has a bug
    # st.extend_elastic_principal_graph(
    #     adata,
    #     epg_ext_mode='QuantDists',
    #     epg_ext_par=0.8)
    # src.plot_dimension_reduction(
    #     adata,
    #     color=['label'],
    #     n_components=2,
    #     show_graph=True,
    #     show_text=True,
    #     save_fig=True,
    #     fig_name=f'{dataset}_dr_extend.pdf')
    # src.plot_branches(
    #     adata,
    #     show_text=True,
    #     save_fig=True,
    #     fig_name=f'{dataset}_branches_extend.pdf')
    # src.plot_visualization_2D(
    #     adata,
    #     method='umap',
    #     n_neighbors=100,
    #     color=['label', 'branch_id_alias', 'S0_pseudotime'],
    #     use_precomputed=False,
    #     save_fig=True,
    #     fig_name=f'{dataset}_visualization_2D_2.pdf')
    src.plot_flat_tree(
        adata,
        color=['label', 'branch_id_alias', 'S0_pseudotime'],
        dist_scale=0.5,
        show_graph=True,
        show_text=True,
        save_fig=True,
        fig_name=f'{dataset}_flat_tree.pdf')
    # exit(0)
    # src.plot_stream_sc(
    #     adata,
    #     root='S0',
    #     color=['label'],  # 'Gata1' for 'Nestorowa_2016'
    #     dist_scale=0.5,
    #     show_graph=True,
    #     show_text=False,
    #     save_fig=True)
    # src.plot_stream(
    #     adata,
    #     root='S0',
    #     color=['label'],  # 'Gata1' for 'Nestorowa_2016'
    #     save_fig=True)
    # src.detect_leaf_markers(
    #     adata,
    #     marker_list=adata.uns['var_genes'][:300],
    #     root='S0',
    #     n_jobs=4)
    # src.detect_transition_markers(
    #     adata,
    #     root='S0',
    #     marker_list=adata.uns['var_genes'][:300],
    #     n_jobs=4)
    # src.detect_de_markers(
    #     adata,
    #     marker_list=adata.uns['var_genes'][:300],
    #     root='S0',
    #     n_jobs=4)
    # st.write(adata,file_name='stream_result.pkl')
    exit(66)
    # print(ref_temp_folder)
    # print(comp_temp_folder)

    # pathlist = Path(ref_temp_folder)
    # for path in pathlist.glob('**/*'):
    #     if path.is_file() and (not path.name.startswith('.')):
    #         file = os.path.relpath(str(path), ref_temp_folder)
    #         print(file)
    #         if(file.endswith('pdf')):
    #             if(os.path.getsize(
    #                os.path.join(comp_temp_folder, file)) > 0):
    #                 print('The file %s passed' % file)
    #             else:
    #                 raise Exception(
    #                     'Error! The file %s is not matched' % file)
    #         else:
    #             checklist = list()
    #             df_ref = pd.read_csv(
    #                 os.path.join(ref_temp_folder, file), sep='\t')
    #             # print(df_ref.shape)
    #             # print(df_ref.head())
    #             df_comp = pd.read_csv(
    #                 os.path.join(comp_temp_folder, file), sep='\t')
    #             # print(df_comp.shape)
    #             # print(df_comp.head())
    #             for c in df_ref.columns:
    #                 # print(c)
    #                 if(is_numeric_dtype(df_ref[c])):
    #                     checklist.append(
    #                         all(np.isclose(df_ref[c], df_comp[c])))
    #                 else:
    #                     checklist.append(
    #                         all(df_ref[c] == df_comp[c]))
    #             if(all(checklist)):
    #                 print('The file %s passed' % file)
    #             else:
    #                 raise Exception(
    #                     'Error! The file %s is not matched' % file)

    rmtree(comp_temp_folder, ignore_errors=True)
    rmtree(ref_temp_folder, ignore_errors=True)
    print('Successful!')


def main():
    sequence()

if __name__ == "__main__":
    main()
