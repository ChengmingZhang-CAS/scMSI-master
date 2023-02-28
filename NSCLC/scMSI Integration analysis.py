# The data has removd batch effect
# analysis NSCLC RNA/Protein

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from matplotlib import pyplot
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import norm
import os
from sklearn import metrics, preprocessing
import scanpy as sc
import anndata as ad
from scMSI.scMSI_main import SCMSIRNA, SCMSIProtein, SCMSIRNAProtein
from scMSI.utils import read_txt, init_library_size
from scMSI.utils import get_top_coeff, get_knn_Aff
import time
import os
from mttkinter import mtTkinter as tk

NSCLC_path = "E:/scMSI_project/data/NSCLC/NSCLC Integration analysis/"
NSCLC = ad.read_h5ad(NSCLC_path + "NSCLC_adata_removing_4personbatch_analysis_inner_ALLfeature.h5ad")

# Integration model and visualization
# use denoised_data
NSCLC.X = NSCLC.obsm["denoised_rna"]
sc.pp.normalize_total(NSCLC, target_sum=1e4)
sc.pp.log1p(NSCLC)
sc.set_figure_params(figsize=(6.7, 4.5))
sc.pp.neighbors(NSCLC, use_rep="nsclc_latent", key_added="nsclc_neighbor")
sc.tl.leiden(NSCLC, resolution=0.4, random_state=0, key_added="nslcl_leiden", neighbors_key="nsclc_neighbor")
sc.tl.umap(NSCLC, min_dist=0.3, random_state=0, neighbors_key="nsclc_neighbor")
sc.pl.umap(NSCLC, color=['nslcl_leiden', "data_info"], frameon=False, title="NSCLC_latent")

sc.pp.neighbors(NSCLC, use_rep="nsclc_coeff", key_added="nsclc_neighbor_coeff", n_neighbors=20)
sc.tl.leiden(NSCLC, resolution=2, random_state=0, key_added="nsclc_leiden_coeff", neighbors_key="nsclc_neighbor_coeff")
sc.tl.umap(NSCLC, min_dist=0.2, spread=0.3, random_state=0, neighbors_key="nsclc_neighbor_coeff")
sc.pl.umap(NSCLC, color=['nsclc_leiden_coeff', "data_info"], frameon=False, title="NSCLC_coeff")
sc.pl.umap(NSCLC, color=['nsclc_leiden_coeff'], frameon=False, title="NSCLC_coeff", legend_loc="on data")
sc.tl.rank_genes_groups(NSCLC, 'nsclc_leiden_coeff', method='wilcoxon')
# marker_genes_rna_latent = pd.DataFrame(NSCLC.uns['rank_genes_groups']['names']).head(20)
# marker_genes_rna_latent.to_csv(NSCLC_path + "Integration_marker_genes_rna_latent_top20.csv")
# protein visual
pro_adata = sc.AnnData(NSCLC.obsm["denoised_protein"].copy(), obs=NSCLC.obs)
sc.pp.log1p(pro_adata)
# Keep log normalized data in raw
pro_adata.raw = pro_adata
pro_adata.var["protein_names"] = NSCLC.uns["protein_names"]
pro_adata.var_names = NSCLC.uns["protein_names"]
pro_adata.obsm["X_umap"] = NSCLC.obsm["X_umap"]
pro_adata.obsm["leiden_coeff"] = NSCLC.obsm["nsclc_coeff"]
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
# use dotplot visual
marker_genes_dict = {
    "B & plasma": ["MZB1", "MS4A1", "BANK1"],
    "Mast cell": ["TPSAB1", "TPSB2"],
    "cDC1": ["IRF8", "WDFY4", "CLEC9A"],
    "cDC2": ["CD1C", "FCER1A"],
    "MregDC": ["FSCN1", "CCR7"],
    "cDC3": ["CLEC10A", "S100A8", "S100A9", "C1QA", "C1QB"],
    "AMΦ": ["SERPINA1", "PPARG"],
    "IMΦ": ["CSF1R", "LYVE1", "CX3CR1"],
    "Mono": ["FCGR3A"],
    "MΦ": ["MRC1", "VSIG4", "SIGLEC1"],
    "MOMΦ": ["MAFB", "CEBPD", "FCGR2B", 'CSF1R'],
    "AZU1+ MΦ": ["AZU1", "CTSG"],
    "non expression": ["MPO", "CSF3R", "LRG1", "FFAR4", "VASP"],
    "NK": ["KLRD1", "NKG7", "GNLY", "NCR1", "SPON2", "PRF1", "GZMB"],
    "T cell": ["CD8A", "CD3D", "TRAC"],
    "pDC": ["LILRA4"]
}

marker_protein_dict = {
    "B & plasma": ["CD19", "CD38"],
    "Mast cell": ["CD33"],
    "cDC1": ["CD141", "CD26"],
    "cDC2": ["CD1c", "CD5"],
    "MregDC": ["HLA-DR", "CD86", "PDL1", "CD40"],
    "DC3": ["CD11b", "CD14", "CD163"],
    "Mono": ["CD16"],
    "NK": ["CD16", "CD56"],
    "T cell": ["CD3", "CD8", "CD4"],
    "pDC": ["CD123"]
}
sc.pl.dotplot(NSCLC, marker_genes_dict, "nsclc_leiden_coeff", dendrogram=True)
sc.pl.dotplot(pro_adata, marker_protein_dict, "nsclc_leiden_coeff", dendrogram=True)

NSCLC.layers["scale"] = preprocessing.scale(NSCLC.X)
pro_adata.layers["scale"] = preprocessing.scale(pro_adata.X)
# ax = sc.pl.heatmap(NSCLC, marker_genes_dict, groupby="nsclc_leiden_coeff", layer="scale", vmin=-2, vmax=2, cmap="RdBu_r", dendrogram=True, swap_axes=True)

# MΦ expression: CD33
cluster2annotation_NSCLC = {
    "0": "CD4 T",
    "1": "CD8 T",
    "2": "MΦ",
    "3": "CD8 T",
    "4": "CD4 T",
    "5": "MΦ",
    "6": "B & plasma",
    "7": "CD4 T",
    "8": "CD4 T",
    "9": "B & plasma",
    "10": "CD8 T",
    "11": "Mono",
    "12": "B & plasma",
    "13": "DC",
    "14": "NK cell",
    "15": "DC",
    "16": "CD4 T",
    "17": "DC",
    "18": "DC",
    "19": "MΦ",
    "20": "MoMΦ",
    "21": "MoMΦ",
    "22": "MoMΦ",
    "23": "MΦ",
    "24": "NK cell",
    "25": "CD4 T",
    "26": "MoMΦ",
    "27": "MΦ",
    "28": "CD4 T",
    "29": "Mono",
    "30": "MΦ",
    "31": "B & plasma",
    "32": "B & plasma",
    "33": "B & plasma",
    "34": "MoMΦ",
    "35": "MoMΦ",
    "36": "pDC",
    "37": "MoMΦ",
    "38": "CD8 T",
    "39": "Mast cell",  # check
    "40": "MoMΦ",
    "41": "DC"
}
NSCLC.obs['NSCLC_celltype_globle'] = NSCLC.obs['nsclc_leiden_coeff'].map(cluster2annotation_NSCLC).astype('category')
pro_adata.obs["NSCLC_celltype_protein"] = NSCLC.obs['NSCLC_celltype_globle']
ax = sc.pl.heatmap(NSCLC, marker_genes_dict, groupby="NSCLC_celltype_globle", layer="scale", vmin=-2, vmax=2,
                   cmap="RdBu_r", dendrogram=False, swap_axes=True)
ax = sc.pl.heatmap(pro_adata, marker_protein_dict, groupby="NSCLC_celltype_protein", layer="scale", vmin=-2, vmax=2,
                   cmap="bwr", dendrogram=False)# , swap_axes=True

sc.pl.umap(NSCLC, color=['NSCLC_celltype_globle', "data_info"], title='NSCLC',
           frameon=False, legend_fontsize=11, legend_fontoutline=2)
sc.pl.umap(NSCLC, color=['NSCLC_celltype_globle'], legend_loc='on data', title='NSCLC',
           frameon=False, legend_fontsize=11, legend_fontoutline=2)
# TO get DC cluster and Differentiate MΦ by CD33
DC_cell = NSCLC[NSCLC.obs["NSCLC_celltype_globle"] == "DC"].copy()
sc.pp.neighbors(DC_cell, use_rep="nsclc_coeff", key_added="DC_neighbor_coeff")
sc.tl.leiden(DC_cell, resolution=1, random_state=0, key_added="DC_leiden_coeff", neighbors_key="DC_neighbor_coeff")
sc.tl.umap(DC_cell, min_dist=0.2, spread=0.4, random_state=0, neighbors_key="DC_neighbor_coeff")
sc.pl.umap(DC_cell, color=['DC_leiden_coeff', "data_info"], frameon=False, title="DC_coeff")
sc.pl.umap(DC_cell, color=['DC_leiden_coeff'], frameon=False, title="MNP_coeff", legend_loc="on data")
# use dotplot visual
marker_genes_dict_DC = {
    "cDC1": ["IRF8", "WDFY4", "CLEC9A"],
    "cDC2": ["CD1C", "FCER1A"],
    "MregDC": ["FSCN1", "CCR7"]
}
marker_protein_DC = {
    "cDC1": ["CD141", "CD26"],
    "cDC2": ["CD1c", "CD5"],
    "MregDC": ["HLA-DR", "CD86", "PDL1", "CD40"]
}
pro_DC = sc.AnnData(DC_cell.obsm["denoised_protein"].copy(), obs=DC_cell.obs)
sc.pp.log1p(pro_DC)
# Keep log normalized data in raw
pro_DC.raw = pro_DC
pro_DC.var["protein_names"] = DC_cell.uns["protein_names"]
pro_DC.var_names = DC_cell.uns["protein_names"]
pro_DC.obsm["X_umap"] = DC_cell.obsm["X_umap"]
pro_DC.obsm["leiden_coeff"] = DC_cell.obs["DC_leiden_coeff"]
sc.pl.dotplot(DC_cell, marker_genes_dict_DC, "DC_leiden_coeff", dendrogram=True)
sc.pl.dotplot(pro_DC, marker_protein_DC, "DC_leiden_coeff", dendrogram=True)

cluster2annotation_DC = {
    "0": "cDC2",
    "1": "cDC2",
    "2": "mregDC",#C
    "3": "mregDC",
    "4": "cDC2",#C
    "5": "cDC2",
    "6": "cDC2",
    "7": "cDC2",
    "8": "cDC1", #C
    "9": "cDC2",
    "10": "cDC1",
    "11": "cDC2",#C
    "12": "cDC2",
    "13": "cDC1",#C
    "14": "cDC2"
}
DC_cell.obs['DC_leiden_coeff_label'] = DC_cell.obs['DC_leiden_coeff'].map(cluster2annotation_DC).astype('category')
pro_DC.obs["DC_leiden_coeff_label_protein"] = DC_cell.obs['DC_leiden_coeff_label']
DC_cell.layers["scale"] = preprocessing.scale(DC_cell.X)
pro_DC.layers["scale"] = preprocessing.scale(pro_DC.X)
ax = sc.pl.heatmap(DC_cell, marker_genes_dict_DC, groupby="DC_leiden_coeff_label", layer="scale", vmin=-2, vmax=2,
                   cmap="RdBu_r", dendrogram=False, swap_axes=True)
ax = sc.pl.heatmap(pro_DC, marker_protein_DC, groupby="DC_leiden_coeff_label_protein", layer="scale", vmin=-2, vmax=2,
                   cmap="bwr", dendrogram=False, swap_axes=True)

sc.pl.umap(DC_cell, color=['DC_leiden_coeff_label', "data_info"], title='NSCLC',
           frameon=False, legend_fontsize=11, legend_fontoutline=2)
sc.pl.umap(DC_cell, color=['DC_leiden_coeff_label'], legend_loc='on data', title='NSCLC',
           frameon=False, legend_fontsize=11, legend_fontoutline=2)
# "MΦ": ["S100A8", "S100A9", "C1QA", "C1QB"],  # 巨噬细胞
# "AMΦ": ["SERPINA1"],  # 肺泡巨噬细胞
# "IMΦ": ["LYVE1", "CX3CR1"],
# "Mono": ["FCGR3A"],
# "MoMΦ": ["MAFB", "CEBPD", "FCGR2B"] # 可做
# NSCLC.write(NSCLC_path + "NSCLC_adata_Intermediate process result_have anntotaion.h5ad")
# the number cell of every celltype in total cell
# for barplot
# celltype_info = pd.DataFrame(NSCLC.obs["NSCLC_celltype_globle"])
# data_info = pd.DataFrame(NSCLC.obs["data_info"])
# df = pd.concat([celltype_info, data_info], axis=1)
# df.to_csv(NSCLC_path + "NSCLC_celltype_info.csv")
# dc_sub = pd.DataFrame(DC_cell.obs["DC_leiden_coeff_label"])
# dc_datainfo = pd.DataFrame(DC_cell.obs["data_info"])
# df2 = pd.concat([dc_sub, dc_datainfo], axis=1)
# df2.to_csv(NSCLC_path + "DC_celltype_info.csv")
# SAVE
NSCLC.write(NSCLC_path + "NSCLC_adata_Intermediate process result_have anntotaion_20230102.h5ad")
DC_cell.write(NSCLC_path + "DC_adata_Intermediate process result_have anntotaion_20230102.h5ad")
pro_DC.write(NSCLC_path + "PRODC_adata_Intermediate process result_have anntotaion_20230102.h5ad")
pro_adata.write(NSCLC_path + "PRO_adata_Intermediate process result_have anntotaion_20230102.h5ad")