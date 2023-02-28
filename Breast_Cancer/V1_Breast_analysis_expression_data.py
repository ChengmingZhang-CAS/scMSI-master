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
from scMSI.scMSI_main import SCMSIRNA, SCMSIProtein, SCMSIRNAProtein
from scMSI.utils import read_txt, init_library_size
from scMSI.utils import get_top_coeff, get_knn_Aff
import time
# Load data
breast_path = "E:/scMSI_project/data/V1_Breast_Cancer/data/"
augment = pd.read_csv(os.path.join(breast_path, "rna_expression_augment.csv"), index_col=0)
raw_adata = sc.read_h5ad(os.path.join(breast_path, "breast_ST_adata.h5ad"))
row_id = augment.index & raw_adata.obs_names
col_id = augment.columns & raw_adata.var_names
adata = raw_adata[row_id, col_id].copy()
adata.X = augment.loc[row_id, col_id]
df = pd.read_csv(breast_path + "metadata.tsv", sep="\t")
df.index = df["ID"]
# adata.X = adata.X.toarray()
# adata.var_names_make_unique()
# # Visualization of spatial data
# sc.set_figure_params(facecolor="white", figsize=(6, 6), dpi_save=300, dpi=100)
# sc.pl.spatial(adata, img_key="hires", color=["in_tissue", "CD3D", "MS4A1", "CD68", "ERBB2"], cmap="Spectral_r", ncols=3,
#               basis="spatial", img=None, size=1.0, spot_size=None, bw=False, alpha_img=1)
# sc.pl.spatial(adata, img_key="hires", color=["in_tissue", "CD3D", "MS4A1"], cmap="Spectral_r", spot_size=0)
# adata.var["mt"] = adata.var_names.str.startswith("MT-")
# sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
# sc.pl.highest_expr_genes(adata, n_top=20)
# sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
#              jitter=0.4, multi_panel=True)
# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_cells=3)
adata.layers["rna_expression"] = adata.X.copy()
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(
#     adata,
#     n_top_genes=2000,
#     flavor="seurat",
#     # batch_key="batch",
#     subset=True,
# )
# train RNA model
max_epochs =500
rna_model = SCMSIRNA(
    adata,
    n_latent=10,
    latent_distribution="ln",
    gene_likelihood="zinb",
    log_variational=False
)
a = time.time()
rna_model.train(max_epochs=max_epochs, n_sample_ref=adata.shape[0])
b = time.time()
print("RNA_use_time: ", b - a)


rna_model2 = SCMSIRNA(
    adata,
    n_latent=10,
    latent_distribution="ln",
    gene_likelihood="zinb",
    log_variational=True
)
a = time.time()
rna_model2.train(max_epochs=2000, n_sample_ref=adata.shape[0])
b = time.time()
print("RNA_use_time: ", b - a)

rna_model4 = SCMSIRNA(
    adata,
    n_latent=10,
    latent_distribution="ln",
    gene_likelihood="zinb",
    log_variational=True
)
a = time.time()
rna_model4.train(max_epochs=1000, n_sample_ref=adata.shape[0])
b = time.time()
print("RNA_use_time: ", b - a)
# torch.save(rna_model.module.state_dict(), cbmc_path + 'cbmc_RNA_train.pt')
# model_state_dict = torch.load('E:\\scMSI_project\\data\\CBMC\\cbmc_RNA_train.pt', map_location=torch.device("cpu"))
# rna_model.module.state_dict(model_state_dict)
# Protein model
# protein_model = SCMSIProtein(
#     adata,
#     n_latent=10,
#     latent_distribution="normal",
#     n_hidden=256
# )
# a = time.time()
# protein_model.train(max_epochs=max_epochs)
# b = time.time()
# print("Protein_use_time: ", b - a)
# torch.save(protein_model.module.state_dict(), cbmc_path + 'cbmc_Protein_train.pt')
# model_state_dict = torch.load('E:\\scMSI_project\\data\\CBMC\\cbmc_Protein_train.pt', map_location=torch.device("cpu"))
# protein_model.module.state_dict(model_state_dict)
# RNA + Protein model
# cbmc_model = SCMSIRNAProtein(
#     adata,
#     n_latent=10,
#     latent_distribution="normal",
#     n_sample_ref=100,
# )
# a = time.time()
# cbmc_model.train(max_epochs=max_epochs, record_loss=True)
# b = time.time()
# print("CBMC_use_time: ", b - a)
# torch.save(cbmc_model.module.state_dict(), cbmc_path + 'cbmc_ProteinAndRNA_train.pt')
# model_state_dict = torch.load('E:\\scMSI_project\\data\\CBMC\\cbmc_ProteinAndRNA_train.pt', map_location=torch.device("cpu"))
# cbmc_model.module.state_dict(model_state_dict)


# model analysis
adata.obs['trueType'] = df.loc[adata.obs_names, 'fine_annot_type']
latent_rna = rna_model4.get_latent_representation(batch_size=128)
adata.obsm["rna_latent"] = latent_rna
# RNA model visualization
sc.set_figure_params(figsize=(6.7, 4.5))
sc.pp.neighbors(adata, use_rep="rna_latent", key_added="rna_neighbor_solo")
sc.tl.leiden(adata, resolution=0.6, random_state=0, key_added="rna_leiden_solo", neighbors_key="rna_neighbor_solo")
adata.obs['kmeans'] = KMeans(n_clusters=20).fit(adata.obsm["rna_latent"]).labels_
adata.obs['kmeans'] = adata.obs['kmeans'].astype("category")
# sc.tl.umap(adata, min_dist=0.5, random_state=0, neighbors_key="rna_neighbor_solo", spread=4)
# sc.pl.umap(adata, color=['rna_leiden_solo'], frameon=False, title="rna_latent_ln", legend_loc="on data")
sc.pl.spatial(adata, img_key="hires", color=['rna_leiden_solo'], title="latent0.6")
sc.pl.spatial(adata, img_key="hires", color=['kmeans'], title="kmeans20")

# latent ARI
from sklearn import metrics
labels_true = adata.obs["trueType"]
labels_pred = adata.obs["rna_leiden_solo"]
labels_pred = adata.obs["kmeans"]
cur_ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
cur_ARI


# Coeff
coeff_rna = rna_model4.get_coeff()
# norm
from scipy.linalg import norm
coeff_rna_norm = coeff_rna / norm(coeff_rna, ord=2, axis=1, keepdims=True)
coeff_pca = PCA(n_components=50, random_state=0).fit_transform(coeff_rna)  # svd_solver="arpack"
adata.obsm["coeff"] = coeff_rna
adata.obsm["coeff_norm"] = coeff_rna_norm
adata.obsm["coeff_pca"] = coeff_pca

# latent
# sc.pp.neighbors(adata, use_rep="cbmc_latent", key_added="cbmc_neighbor_latent")
# sc.tl.leiden(adata, resolution=0.4, key_added="cbmc_leiden", neighbors_key="cbmc_neighbor_latent")
# sc.tl.umap(adata, min_dist=0.3, neighbors_key="cbmc_neighbor_latent")
# sc.pl.umap(adata, color=['cbmc_leiden'], frameon=False, title="latent", legend_loc="on data")
# coeff
sc.pp.neighbors(adata, use_rep="coeff", key_added="neighbor_coeff")
sc.tl.leiden(adata, resolution=1.1, key_added="cbmc_leiden", neighbors_key="neighbor_coeff")
# sc.tl.umap(adata, min_dist=0.4, neighbors_key="cbmc_neighbor_coeff", spread=4)
adata.obs['coeff_kmeans'] = KMeans(n_clusters=20).fit(adata.obsm["coeff"]).labels_
adata.obs['coeff_kmeans'] = adata.obs['coeff_kmeans'].astype("category")
# sc.pl.umap(adata, color=['cbmc_leiden'], frameon=False, title="norm_coeff_ln", legend_loc="on data")
sc.pl.spatial(adata, img_key="hires", color=['cbmc_leiden', "trueType"], title="coeff_resolution1.1")
sc.pl.spatial(adata, img_key="hires", color=['coeff_kmeans'], title="coeff_kmeans20")


sc.pp.neighbors(adata, use_rep="coeff_norm", key_added="neighbor_coeff_norm")
sc.tl.leiden(adata, resolution=0.9, key_added="cbmc_leiden", neighbors_key="neighbor_coeff_norm")
# sc.tl.umap(adata, min_dist=0.4, neighbors_key="neighbor_coeff_norm", spread=3)
adata.obs['coeff_norm_kmeans'] = KMeans(n_clusters=20).fit(adata.obsm["coeff_norm"]).labels_
adata.obs['coeff_norm_kmeans'] = adata.obs['coeff_norm_kmeans'].astype("category")
# sc.pl.umap(adata, color=['cbmc_leiden'], frameon=False, title="norm_coeff_ln", legend_loc="on data")
sc.pl.spatial(adata, img_key="hires", color=['cbmc_leiden', "trueType"], title="coeff_norm_resolution0.9")
sc.pl.spatial(adata, img_key="hires", color=['coeff_norm_kmeans'], title="coeff_norm_kmeans20")

sc.pp.neighbors(adata, use_rep="coeff_pca", key_added="neighbor_coeff_pca")
sc.tl.leiden(adata, resolution=0.8, key_added="cbmc_leiden", neighbors_key="neighbor_coeff_pca")
# sc.tl.umap(adata, min_dist=0.5, neighbors_key="cbmc_neighbor_coeff", spread=1.2)
adata.obs['coeff_pca_kmeans'] = KMeans(n_clusters=20).fit(adata.obsm["coeff_pca"]).labels_
adata.obs['coeff_pca_kmeans'] = adata.obs['coeff_pca_kmeans'].astype("category")
# sc.pl.umap(adata, color=['cbmc_leiden'], frameon=False, title="norm_coeff_ln", legend_loc="on data")
sc.pl.spatial(adata, img_key="hires", color=['cbmc_leiden', "trueType"], title="coeff_pca_resolution0.8")
sc.pl.spatial(adata, img_key="hires", color=['coeff_norm_kmeans'], title="coeff_pca_kmeans20")

# Switching environment totalVI_env
import scvi

adata.raw = adata
scvi.model.SCVI.setup_anndata(
    adata,
)
model = scvi.model.SCVI(adata, gene_likelihood='nb', latent_distribution='ln')
model.train()
adata.obsm["X_totalVI"] = model.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_totalVI")
sc.tl.leiden(adata, key_added="leiden_totalVI", resolution=0.6)

adata.obs['X_totalVI_kmeans'] = KMeans(n_clusters=20).fit(adata.obsm["X_totalVI"]).labels_
adata.obs['X_totalVI_kmeans'] = adata.obs['X_totalVI_kmeans'].astype("category")
# sc.pl.umap(adata, color=['cbmc_leiden'], frameon=False, title="norm_coeff_ln", legend_loc="on data")
sc.pl.spatial(adata, img_key="hires", color=['leiden_totalVI'], title="scVI_resolution0.6")
sc.pl.spatial(adata, img_key="hires", color=['X_totalVI_kmeans'], title="scvi_kmeans20")
from sklearn import metrics
labels_true = adata.obs["trueType"]
labels_pred = adata.obs["leiden_totalVI"]
cur_ARI = metrics.adjusted_rand_score(labels_true, labels_pred)