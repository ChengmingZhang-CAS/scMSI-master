# %%% scMSI Breast cancer analysis %%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import norm
import os
import scanpy as sc
import anndata as ad
import sys

sys.path.append("E://scMSI_project")
from scMSI.scMSI_main import SCMSIRNA, SCMSIProtein, SCMSIRNAProtein, SCMSIIMAGE
from scMSI.utils import read_txt, init_library_size
from scMSI.utils import get_top_coeff, get_knn_Aff, get_sample_index
import time

file_path = "E:/scMSI_project/data/Mouse Brain Serial Section 1 (Sagittal-Anterior)/data/"
adata = sc.read_h5ad(os.path.join(file_path, "Mouse_brain_ST_adata.h5ad"))
adata.X = adata.X.toarray()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.layers["rna_expression"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    flavor="seurat",
    # batch_key="batch",
    subset=True
)
idxs = get_sample_index(adata.X, n_sample_ref=int(adata.shape[0] * 0.9), train_size=1.0)
# train RNA model
max_epochs = 600
rna_model = SCMSIRNA(
    adata,
    n_latent=10,
    latent_distribution="ln",
    gene_likelihood="zinb",
    log_variational=True
)
a = time.time()
rna_model.train(max_epochs=max_epochs, n_sample_ref=int(adata.shape[0] * 0.9), split_index=idxs)
b = time.time()
print("RNA_use_time: ", b - a)

img_model = SCMSIIMAGE(
    adata,
    n_latent=10,
    latent_distribution="normal",
)

a = time.time()
img_model.train(max_epochs=200, n_sample_ref=int(adata.shape[0] * 0.9), split_index=idxs)
b = time.time()

# model analysis
# img_coeff get
from scipy.linalg import norm

coeff_img = img_model.get_coeff()
coeff_img_norm = coeff_img / norm(coeff_img, ord=2, axis=1, keepdims=True)
coeff_img_pca = PCA(n_components=20, random_state=0).fit_transform(coeff_img_norm)
# express_Coeff
coeff_rna = rna_model.get_coeff()
coeff_rna_norm = coeff_rna / norm(coeff_rna, ord=2, axis=1, keepdims=True)
coeff_rna_pca = PCA(n_components=20, random_state=0).fit_transform(coeff_rna)  # svd_solver="arpack"

img_proportion = 0.05
rna_proportion = 0.95
adata.obsm["coeff"] = img_proportion * coeff_img + rna_proportion * coeff_rna
adata.obsm["coeff_norm"] = img_proportion * coeff_img_norm + rna_proportion * coeff_rna_norm
adata.obsm["coeff_pca"] = img_proportion * coeff_img_pca + rna_proportion * coeff_rna_pca
# coeff
sc.pp.neighbors(adata, use_rep="coeff", key_added="cbmc_neighbor_coeff")
sc.tl.leiden(adata, resolution=0.6, key_added="cbmc_leiden", neighbors_key="cbmc_neighbor_coeff")
# sc.tl.umap(adata, min_dist=0.2, neighbors_key="cbmc_neighbor_coeff", spread=4)
# sc.pl.umap(adata, color=['cbmc_leiden'], frameon=False, title="norm_coeff_ln", legend_loc="on data")
sc.pl.spatial(adata, img_key="hires", color=['cbmc_leiden'], title="coeff_resolution0.6")
# adata.write(file_path + "mouse_brain_final.h5ad")
adata.uns["cbmc_leiden_colors"] = ['#279e68',
                                   '#d62728',
                                   '#1f77b4',
                                   "#ff7f0e",
                                   '#8c564b',
                                   '#e377c2',
                                   '#aa40fc',
                                   '#b5bd61',
                                   '#aec7e8',
                                   '#ffbb78',
                                   '#8c6d31',
                                   '#98df8a',
                                   '#17becf',
                                   '#ff9896',
                                   '#c5b0d5',
                                   '#c49c94',
                                   '#f7b6d2',
                                   '#dbdb8d',
                                   '#9edae5',
                                   '#ad494a',
                                   ]
sc.pl.spatial(
    adata,
    img_key="hires",
    color=["L2/3 IT", "L4", "L5 PT", "L6 CT"],
    size=1.5,
)
sc.pl.spatial(adata, img_key="hires", color=["Olig1", "Olig2"], cmap="Spectral_r", ncols=1,
              basis="spatial", img=None, size=1.0, spot_size=None, bw=False, alpha_img=1)
sc.pl.spatial(adata, img_key="hires", color="cbmc_leiden", groups=["0", "5"])
cluster_info = adata.obs["cbmc_leiden"]
cluster_info.to_csv(file_path + "scMSI_cluster_info.csv")