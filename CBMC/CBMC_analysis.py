# %%% scMSI CBMC analysis %%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import norm
import os
import scanpy as sc
import anndata as ad
from scMSI.scMSI_main import SCMSIRNA, SCMSIProtein, SCMSIRNAProtein
from scMSI.utils import read_txt, init_library_size
from scMSI.utils import get_top_coeff, get_knn_Aff
import time

# data path
cbmc_path = "E:\\scMSI_project\\data\\CBMC\\"
# RNA read
RNA_data = ad.read_csv(cbmc_path + "RNA_Data_for_anndata.csv")
# ADT read
ADT_data = ad.read_csv(cbmc_path + "ADT_Data_for_anndata.csv")
RNA_data.var_names_make_unique()
gene_name = []
for name in RNA_data.var_names:
    name = name.split("_")[1]
    gene_name.append(name)
RNA_data.var_names = gene_name
RNA_data.uns["protein_names"] = np.array(ADT_data.var_names)
RNA_data.obsm["protein_expression"] = ADT_data.X
RNA_data.layers["rna_expression"] = RNA_data.X.copy()
# pre progress
# basic filtering
sc.pp.filter_cells(RNA_data, min_genes=200)
sc.pp.filter_genes(RNA_data, min_cells=3)
RNA_data.var['mt'] = RNA_data.var_names.str.startswith('HUMAN_MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(RNA_data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# sc.pl.violin(RNA_data, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
#              jitter=0.4, multi_panel=True)
RNA_data = RNA_data[RNA_data.obs.n_genes_by_counts < 2500, :]
RNA_data = RNA_data[RNA_data.obs.pct_counts_mt < 9, :]
# sc.pl.violin(RNA_data, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
#              jitter=0.4, multi_panel=True)

RNA_data.layers["rna_expression"] = RNA_data.X.copy()
sc.pp.normalize_total(RNA_data, target_sum=1e4)
sc.pp.log1p(RNA_data)
sc.pp.highly_variable_genes(
    RNA_data,
    n_top_genes=4000,
    flavor="seurat",
    # batch_key="batch",
    subset=True
)
adata = RNA_data
# train RNA model
max_epochs = 400
rna_model = SCMSIRNA(
    adata,
    n_latent=10,
    latent_distribution="normal",
)
a = time.time()
rna_model.train(max_epochs=max_epochs, n_sample_ref=100)
b = time.time()
print("RNA_use_time: ", b - a)
# torch.save(rna_model.module.state_dict(), cbmc_path + 'cbmc_RNA_train.pt')
# model_state_dict = torch.load('E:\\scMSI_project\\data\\CBMC\\cbmc_RNA_train.pt', map_location=torch.device("cpu"))
# rna_model.module.state_dict(model_state_dict)
# Protein model
protein_model = SCMSIProtein(
    adata,
    n_latent=10,
    latent_distribution="normal",
    n_hidden=256
)
a = time.time()
protein_model.train(max_epochs=max_epochs)
b = time.time()
print("Protein_use_time: ", b - a)
# torch.save(protein_model.module.state_dict(), cbmc_path + 'cbmc_Protein_train.pt')
# model_state_dict = torch.load('E:\\scMSI_project\\data\\CBMC\\cbmc_Protein_train.pt', map_location=torch.device("cpu"))
# protein_model.module.state_dict(model_state_dict)
# RNA + Protein model
cbmc_model = SCMSIRNAProtein(
    adata,
    n_latent=10,
    latent_distribution="normal",
    n_sample_ref=100,
)
a = time.time()
cbmc_model.train(max_epochs=max_epochs, record_loss=True)
b = time.time()
print("CBMC_use_time: ", b - a)
# torch.save(cbmc_model.module.state_dict(), cbmc_path + 'cbmc_ProteinAndRNA_train.pt')
# model_state_dict = torch.load('E:\\scMSI_project\\data\\CBMC\\cbmc_ProteinAndRNA_train.pt', map_location=torch.device("cpu"))
# cbmc_model.module.state_dict(model_state_dict)


# model analysis
latent_rna = rna_model.get_latent_representation(batch_size=128)
adata.obsm["rna_latent"] = latent_rna
# RNA model visualization
sc.set_figure_params(figsize=(6.7, 4.5))
sc.pp.neighbors(adata, use_rep="rna_latent", key_added="rna_neighbor_solo")
sc.tl.leiden(adata, resolution=0.4, random_state=0, key_added="rna_leiden_solo", neighbors_key="rna_neighbor_solo")
sc.tl.umap(adata, min_dist=0.3, random_state=0, neighbors_key="rna_neighbor_solo")
sc.pl.umap(adata, color=['rna_leiden_solo'], frameon=False, title="rna_latent", legend_loc="on data",
           legend_fontsize=13, legend_fontoutline=2)
adata.obsm["denoised_rna"] = rna_model.get_normalized_expression(
    n_samples=25,
    return_mean=True,
    transform_batch=[0],  # [0, 1]
    return_numpy=True
)
sc.pp.log1p(adata.obsm["denoised_rna"])
adata.X = adata.obsm["denoised_rna"]
sc.tl.rank_genes_groups(adata, 'rna_leiden_solo', method='wilcoxon')
# sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)
marker_genes_rna_latent = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(15)
cluster2annotation_rna = {
    "0": "CD4 T",
    "1": "Monocyte",
    "2": "NK Cell",
    "3": "Monocyte",
    "4": "Ery.",
    "5": "CD4 T",
    "6": "B Cell",
    "7": "CD4 T",
    "8": "NK Cell",
    "9": "Monocyte",
    "10": "Pre.",
    "11": "Ery.",
    "12": "Platelet",
    "13": "NK Cell"
}
adata.obs['RNA_solo_celltype'] = adata.obs['rna_leiden_solo'].map(cluster2annotation_rna).astype('category')
sc.pl.umap(adata, color='RNA_solo_celltype', legend_loc='on data', title='scMSI latent',
           frameon=False, legend_fontsize=13, legend_fontoutline=2)  # ave='cbmc_RNAlatent.pdf')

# # RNA marker visualization
marker_gene_dict_rna = {
    "B Cell": ["CD79A", "MS4A1"],
    "CD4 T": ["IL7R"],
    "Erythrocyte": ["HBB", "HBA1", "HBG1"],
    "Monocyte": ["LYZ", "CST3", "CFD"],
    "NK Cell": ["GNLY", "NKG7"],
    "Platelet": ["PPBP"],
    "Precursor": ["PRSS57", "STMN1"]
}
adata.layers["scale"] = preprocessing.scale(adata.X)
ax = sc.pl.heatmap(adata, marker_gene_dict_rna, groupby='RNA_solo_celltype', layer="scale", vmin=-2, vmax=2,
                   cmap='RdBu_r', dendrogram=False, swap_axes=True, figsize=(11, 4))

ax = sc.pl.heatmap(adata, marker_gene_dict_rna, groupby='RNA_solo_celltype', cmap='viridis', dendrogram=False,
                   layer="scale")
# #
sc.pl.stacked_violin(adata, marker_gene_dict_rna, groupby='RNA_solo_celltype', swap_axes=False, dendrogram=False)
sc.pl.tracksplot(adata, marker_gene_dict_rna, dendrogram=False, groupby="RNA_solo_celltype")
sc.pl.dotplot(adata, marker_gene_dict_rna, groupby='RNA_solo_celltype')
# Protein model
latent_protein = protein_model.get_latent_representation(batch_size=128)
adata.obsm["protein_latent"] = latent_protein

sc.pp.neighbors(adata, use_rep="protein_latent", key_added="protein_neighbor_latent")
sc.tl.leiden(adata, resolution=0.8, key_added="protein_leiden", neighbors_key="protein_neighbor_latent")
sc.tl.umap(adata, min_dist=0.3, neighbors_key="protein_neighbor_latent")
sc.pl.umap(adata, color=['protein_leiden'], frameon=False, title="protein_latent", legend_loc="on data")
adata.obsm["denoised_protein"] = protein_model.get_normalized_expression(
    n_samples=10,
    return_mean=True,
    transform_batch=[0]
)

pro_adata = sc.AnnData(adata.obsm["protein_expression"].copy(), obs=adata.obs)
pro_adata1 = sc.AnnData(adata.obsm["denoised_protein"].copy(), obs=adata.obs)
sc.pp.log1p(pro_adata)
sc.pp.log1p(pro_adata1)
# # Keep log normalized data in raw
pro_adata.var["protein_names"] = ADT_data.var.index
pro_adata.var_names = ADT_data.var.index
pro_adata.obsm["X_umap"] = adata.obsm["X_umap"]
pro_adata.obsm["latent"] = adata.obsm["protein_latent"]

pro_adata1.var["protein_names"] = ADT_data.var.index
pro_adata1.var_names = ADT_data.var.index
pro_adata1.obsm["X_umap"] = adata.obsm["X_umap"]
pro_adata1.obsm["latent"] = adata.obsm["protein_latent"]

sc.pl.umap(
    pro_adata,
    legend_loc="on data",
    color=pro_adata.var_names,
    gene_symbols="protein_names",
    ncols=3,
    vmax="p99",
    use_raw=False,
    frameon=False,
    wspace=0.1
)
sc.pl.umap(
    pro_adata1,
    legend_loc="on data",
    color=pro_adata.var_names,
    gene_symbols="protein_names",
    ncols=3,
    vmax="p99",
    use_raw=False,
    frameon=False,
    wspace=0.1
)
cluster2annotation_protein = {
    "0": "CD4 T",
    "1": "CD8- NK",
    "2": "CD4 T",
    "3": "CD4 T",
    "4": "Pre.",
    "5": "CD14 Monocyte",
    "6": "CD14 Monocyte",
    "7": "CD14 Monocyte",
    "8": "CD14 Monocyte",
    "9": "CD4 T",
    "10": "B Cell",
    "11": "CD8 T",
    "12": "CD14 Monocyte",
    "13": "CD8+ NK",
    "14": "CD14 Monocyte",
    "15": "CD4 T",
    "16": "CD8+ NK"
}
pro_adata.obs['protein_solo_celltype'] = pro_adata.obs['protein_leiden'].map(cluster2annotation_protein).astype(
    'category')
pro_adata1.obs['protein_solo_celltype'] = pro_adata1.obs['protein_leiden'].map(cluster2annotation_protein).astype(
    'category')
sc.pl.umap(pro_adata, color='protein_solo_celltype', legend_loc='on data',
           frameon=False, legend_fontsize=13, legend_fontoutline=2)
# # Protein marker
# visualization
marker_gene_dict_protein = {
    "CD4 T": ["CD3", "CD4"],
    "Monocyte": ["CD14", "CD11c", "CD16"],
    "NK Cell": ["CD56"],
    "B Cell": ["CD19"],
    "Pre.": ["CD34"],
    "CD8 T": ["CD8"]
}
marker_protein = ["CD3", "CD4", "CD8", "CD45RA", "CD56", "CD16", "CD11c", "CD14", "CD19", "CD34"]
sc.pl.stacked_violin(pro_adata, marker_protein, groupby='protein_solo_celltype', swap_axes=False,
                     dendrogram=False, frameon=True)
sc.pl.stacked_violin(pro_adata1, marker_gene_dict_protein, groupby='protein_solo_celltype', swap_axes=False,
                     dendrogram=False)
sc.pl.rank_genes_groups_dotplot(pro_adata, n_genes=30, values_to_plot='logfoldchanges', min_logfoldchange=4, vmax=7,
                                vmin=-7, cmap='bwr')
sc.pl.tracksplot(pro_adata, marker_gene_dict_protein, groupby='rna_leiden_solo', dendrogram=True)
sc.pl.dotplot(pro_adata, marker_gene_dict_protein, groupby="protein_solo_celltype")
sc.pl.dotplot(pro_adata1, marker_gene_dict_protein, groupby="protein_solo_celltype")
pro_adata1.layers["scale"] = preprocessing.scale(pro_adata1.X)
ax = sc.pl.heatmap(pro_adata1, marker_gene_dict_protein, groupby='protein_leiden', vmin=-2, vmax=2, layer="scale",
                   cmap='bwr', dendrogram=False, swap_axes=True, figsize=(11, 4))

ax = sc.pl.stacked_violin(pro_adata1, marker_gene_dict_protein, groupby='protein_solo_celltype',
                          var_group_positions=[(7, 8)], var_group_labels=['6'])

# # RNA+Protein

rna_latent, pro_latent = cbmc_model.get_latent_representation(batch_size=128)
adata.obsm["rna_latent"] = rna_latent
adata.obsm["pro_latent"] = pro_latent
cbmc_latent = np.concatenate([adata.obsm["rna_latent"], adata.obsm["pro_latent"]], axis=1)
adata.obsm["cbmc_latent"] = cbmc_latent
rna_coeff, pro_coeff, coeff = cbmc_model.get_coeff(pro_w=1.0, ord=2)
coeff_pca = PCA(n_components=20, svd_solver="arpack", random_state=0).fit_transform(coeff)  # svd_solver="arpack"
adata.obsm["cbmc_coeff"] = coeff_pca
# coeff
sc.pp.neighbors(adata, use_rep="cbmc_coeff", key_added="cbmc_neighbor_coeff")
sc.tl.leiden(adata, resolution=0.8, key_added="cbmc_leiden", neighbors_key="cbmc_neighbor_coeff")
sc.tl.umap(adata, min_dist=0.3, neighbors_key="cbmc_neighbor_coeff", spread=3)
sc.pl.umap(adata, color=['cbmc_leiden'], frameon=False, title="coeff", legend_loc="on data")
# #
sc.tl.rank_genes_groups(adata, 'cbmc_leiden', method='wilcoxon')
# sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)
marker_genes_rna_latent = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(15)

pro_adata = sc.AnnData(adata.obsm["protein_expression"].copy(), obs=adata.obs)
pro_adata = sc.AnnData(adata.obsm["denoised_protein"].copy(), obs=adata.obs)
sc.pp.log1p(pro_adata)
# Keep log normalized data in raw
pro_adata.raw = pro_adata
pro_adata.var["protein_names"] = ADT_data.var.index
pro_adata.var_names = ADT_data.var.index
pro_adata.obsm["X_umap"] = adata.obsm["X_umap"]
pro_adata.obsm["latent"] = adata.obsm["cbmc_coeff"]
sc.pl.umap(
    pro_adata,
    legend_loc="on data",
    color=pro_adata.var_names,
    gene_symbols="protein_names",
    ncols=3,
    vmax="p99",
    use_raw=False,
    frameon=False,
    wspace=0.1
)
cluster2annotation_CBMC = {
    "0": "CD4 T",
    "1": "CD4 T",
    "2": "CD14 Monocyte",
    "3": "CD8- NK",
    "4": "CD14 Monocyte",
    "5": "CD14 Monocyte",
    "6": "Ery.",
    "7": "CD4 T",
    "8": "B Cell",
    "9": "CD8 T",
    "10": "CD16 Monocyte",
    "11": "CD14 Monocyte",
    "12": "CD8+ NK",
    "13": "Pre.",
    "14": "Ery.",
    "15": "CD14 Monocyte",
    "16": "CD8+ NK",
    "17": "CD8- NK",
    "18": "Platelet",
    "19": "B Cell",
    "20": "CD4 T"
}
adata.obs['CBMC_celltype'] = adata.obs['cbmc_leiden'].map(cluster2annotation_CBMC).astype('category')
marker_gene_dict_protein = {
    "B Cell": ["CD19"],
    "Monocyte": ["CD14", "CD11c", "CD16"],
    "CD4 T": ["CD3", "CD4"],
    "CD8 T": ["CD8"],
    "NK Cell": ["CD56"],
    "Pre.": ["CD34"]
}
marker_gene_dict_rna = {
    "B Cell": ["CD79A", "MS4A1"],
    "Monocyte": ["LYZ", "CST3", "CFD"],
    "CD4 T": ["IL7R"],
    "NK Cell": ["GNLY", "NKG7"],
    "Erythrocyte": ["HBB", "HBA1", "HBG1"],
    "Platelet": ["PPBP"],
    "Precursor": ["PRSS57", "STMN1"]
}

adata.layers["scale"] = preprocessing.scale(adata.X)
pro_adata.layers["scale"] = preprocessing.scale(pro_adata.X)
ax = sc.pl.heatmap(adata, marker_gene_dict_rna, groupby="CBMC_celltype", layer="scale", vmin=-2, vmax=2,
                   cmap="RdBu_r", dendrogram=False, swap_axes=True)
ax = sc.pl.heatmap(pro_adata, marker_gene_dict_protein, groupby="CBMC_celltype", layer="scale", vmin=-2, vmax=2,
                   cmap="bwr", dendrogram=False, swap_axes=False)  # , swap_axes=True

sc.pl.umap(adata, color='CBMC_celltype', legend_loc='on data', title='scMSI latent',
           frameon=False, legend_fontsize=11, legend_fontoutline=2)  # ave='cbmc_RNAlatent.pdf')

# Use the same coordinates for visualization
sc.pl.umap(adata, color=['RNA_solo_celltype'], frameon=False, title="RNA",
           legend_loc="on data", legend_fontsize=11, legend_fontoutline=2)  # rna_leiden_solo RNA_solo_celltype

adata.obs["protein_solo_celltype"] = pro_adata.obs['protein_solo_celltype']
sc.pl.umap(adata, color=['protein_solo_celltype'], frameon=False, title="Protein",
           legend_loc="on data", legend_fontsize=11, legend_fontoutline=2)

# 颜色修改一致
len(adata.uns["CBMC_celltype_colors"])
len(adata.uns["protein_solo_celltype_colors"])
len(adata.uns["RNA_solo_celltype_colors"])
adata.uns["protein_solo_celltype_colors"]
color_use = adata.uns["CBMC_celltype_colors"]
color_use[0:8]
pro_adata.uns["protein_solo_celltype_colors"] = ['#1f77b4',
                                                 "#ff7f0e",
                                                 '#279e68',
                                                 '#aa40fc',
                                                 '#e377c2',
                                                 '#8c564b',
                                                 '#d62728',
                                                 ]
sc.pl.umap(pro_adata, color='protein_solo_celltype', legend_loc='on data', frameon=False, legend_fontsize=13,
           legend_fontoutline=2)
adata.uns["RNA_solo_celltype_colors"] = ['#1f77b4',
                                         '#279e68',
                                         '#d62728',
                                         '#ff7f0e',
                                         '#8c564b',
                                         '#8c564b',
                                         '#e377c2',
                                         '#b5bd61']

adata.uns["CBMC_celltype_colors"] = ['#1f77b4',
                                     '#ff7f0e',
                                     '#b5bd61',
                                     '#279e68',
                                     '#aa40fc',
                                     '#e377c2',
                                     '#8c564b',
                                     '#d62728',
                                     '#17becf',
                                     '#aec7e8']

