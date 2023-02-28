# %%% scMSI In-house PBMC analysis %%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import norm
import os
from sklearn import metrics
import scanpy as sc
import anndata as ad
from scMSI.scMSI_main import SCMSIRNA, SCMSIProtein, SCMSIRNAProtein
from scMSI.utils import read_txt, init_library_size
from scMSI.utils import get_top_coeff, get_knn_Aff
import time

pbmc_path = "E:\\scMSI_project\\for_github\\In-house_PBMC\\"

# seurat_to_scanpy
TOselect_feature = sc.read_h5ad(pbmc_path + "seuratObject_to_sacnpy.h5ad")
feature_select = TOselect_feature.var["vst.variable"].keys()
cell_select = TOselect_feature.obs["nCount_RNA"].keys()
index_col = []
for col in cell_select:
    tmp = col.replace(".", "-")
    index_col.append(tmp)

cell_select = index_col
df = pd.read_csv(pbmc_path + "truth_InHouse.csv", index_col=0)
# RNA read
RNA_data = pd.read_csv(pbmc_path + "GSM4476363_RNA_raw.csv.gz", index_col=0, header=0)
RNA_data = RNA_data.T
RNA_data = ad.AnnData(RNA_data)
RNA_data.var_names_make_unique()
# ADT read
ADT_data = pd.read_csv(pbmc_path + "GSM4476364_ADT_raw.csv", index_col=0, header=0)
ADT_data = ADT_data.T
ADT_data = ad.AnnData(ADT_data)
RNA_data.uns["protein_names"] = ADT_data.var
RNA_data.obsm["protein_expression"] = ADT_data.X
RNA_data.layers["rna_expression"] = RNA_data.X.copy()
RNA_data = RNA_data[cell_select, feature_select]

sc.pp.normalize_total(RNA_data, target_sum=1e4)
sc.pp.log1p(RNA_data)
RNA_data.raw = RNA_data
adata = RNA_data
model = SCMSIRNAProtein(
    adata,
    n_latent=10,
    latent_distribution="normal",
    n_sample_ref=100

)
max_epochs = 400
a = time.time()
model.train(max_epochs=max_epochs, record_loss=True)
b = time.time()
print("times: ", b - a)
rna_coeff, pro_coeff, coeff = model.get_coeff(pro_w=1.0, ord=2)
coeff_pca = PCA(n_components=20, svd_solver="arpack", random_state=0).fit_transform(coeff)
adata.obsm["coeff_pca"] = coeff_pca
adata.obs['trueType'] = df.loc[adata.obs_names, 'trueType']
new_adata = adata[adata.obs["trueType"] != "unknown"].copy()
use_rep = "coeff_pca"
sc.pp.neighbors(new_adata, use_rep=use_rep)
sc.tl.leiden(new_adata, key_added="leiden", resolution=0.3)
new_adata.obs['kmeans'] = KMeans(n_clusters=6).fit(new_adata.obsm[use_rep]).labels_
sc.tl.umap(new_adata, min_dist=0.4, spread=0.6)
sc.pl.umap(new_adata, color=["kmeans"], frameon=False, legend_loc="on data", title=use_rep)
labels_true = new_adata.obs['trueType']
labels_pred = new_adata.obs['kmeans']
print("labels_pred cluster num: ", len(labels_pred.unique()))
cur_ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
cur_AMI = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
cur_NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
