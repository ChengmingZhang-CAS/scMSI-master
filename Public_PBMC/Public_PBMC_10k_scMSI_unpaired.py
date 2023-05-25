# %%% scMSI PBMC_in-house analysis %%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from scipy.linalg import norm
import os
import scanpy as sc
import anndata as ad
import sys
# sys.path.append(r"E:\benchmark multi-omics analysis\scMSI\scMSI_V6")
from scMSI.scMSI_main import SCMSIRNA, SCMSIProtein, SCMSIRNAProtein, SCMSICiteRna
from scMSI.utils import read_txt, init_library_size
from scMSI.utils import get_top_coeff, get_knn_Aff
import time
import os
os.environ["OMP_NUM_THREADS"] = '1'


debug_mode = True
# debug_mode = False
data_path = "PBMC_10k/"
if debug_mode:
    data_path = "../PBMC_10k/"

pbmc_path = data_path
# seurat_to_scanpy
TOselect_feature = sc.read_h5ad(pbmc_path + "seuratObject_to_sacnpy.h5ad")
feature_select = TOselect_feature.var["vst.variable"].keys()
cell_select = TOselect_feature.obs["nCount_RNA"].keys()
# RNA read
pbmc = sc.read_10x_h5(pbmc_path + "pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5",  gex_only=False)
pbmc.var_names_make_unique()
pbmc.layers["rna_expression"] = pbmc.X.copy().A
pbmc.var["feature_types"].value_counts()
protein = pbmc[:, pbmc.var["feature_types"] == "Antibody Capture"].copy()
protein = protein[:, 0:14]
rna = pbmc[:, pbmc.var["feature_types"] == "Gene Expression"].copy()
adata = rna
protein_names = protein.var["gene_ids"].values
adata.obsm["protein_expression"] = protein.X.toarray()
adata = adata[cell_select, feature_select]
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.X = adata.X.toarray()
adata.raw = adata

nan_pro_num = int(adata.shape[0] * 0.0)
nan_pro_idx = np.random.permutation(adata.shape[0])[0:nan_pro_num]
held_out_proteins = adata.obsm["protein_expression"][nan_pro_idx].copy()
# adata.obsm["protein_expression"][nan_pro_idx] = np.zeros_like(adata.obsm["protein_expression"][nan_pro_idx])
adata.obsm["protein_expression"][nan_pro_idx] = 0

cur_ARI_all = pd.DataFrame()
cur_AMI_all = pd.DataFrame()
cur_NMI_all = pd.DataFrame()
model = SCMSICiteRna(
    adata,
    n_latent=10,
    latent_distribution="normal",
)

max_epochs = 200
a = time.time()
model.train(max_epochs=max_epochs, record_loss=True)
b = time.time()
print("time: ", b - a)
time_use = b - a

rna_latent, pro_latent = model.get_latent_representation(batch_size=256)
adata.obsm["rna_latent"] = rna_latent
adata.obsm["pro_latent"] = pro_latent
rna_rec_latent, pro_rec_latent = model.get_reconstruct_latent(batch_size=256)
adata.obsm["rna_rec_latent"] = rna_rec_latent
adata.obsm["pro_rec_latent"] = pro_rec_latent

mix_latent = np.concatenate([adata.obsm["rna_latent"], adata.obsm["pro_latent"]], axis=1)
adata.obsm["latent"] = mix_latent
rna_coeff, pro_coeff, coeff = model.get_coeff(pro_w=1.0, ord=2)
coeff_pca = PCA(n_components=20, svd_solver="arpack", random_state=0).fit_transform(coeff)
adata.obsm["coeff_pca"] = coeff_pca

# sc.set_figure_params(figsize=(16, 9))

df = pd.read_csv(pbmc_path + "truth_10X10k.csv", index_col=0)
# adata.obs_names = [name.replace('.', '-') for name in adata.obs_names]
adata.obs['trueType'] = df.loc[adata.obs_names, 'trueType']
new_adata = adata[adata.obs["trueType"] != "unknown"].copy()
use_rep_keys = ["latent", "coeff_pca"]
for use_rep in use_rep_keys:
    print("*" * 100)
    print("use_rep: ", use_rep)
    sc.pp.neighbors(new_adata, use_rep=use_rep)
    sc.tl.leiden(new_adata, key_added="leiden", resolution=0.3)
    sc.tl.umap(new_adata, min_dist=0.4)
    # sc.pl.umap(new_adata, color=["leiden", "trueType"], frameon=False, legend_loc="on data", title=use_rep)
    labels_true = new_adata.obs['trueType']
    labels_pred = new_adata.obs['leiden']
    print("labels_pred cluster num: ", len(labels_pred.unique()))
    cur_ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
    cur_AMI = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    cur_NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    cur_ARI_all = cur_ARI_all.append([cur_ARI])
    cur_AMI_all = cur_AMI_all.append([cur_AMI])
    cur_NMI_all = cur_NMI_all.append([cur_NMI])
    print(f"scMSI ARI: {cur_ARI}\n")
    print(f"scMSI AMI: {cur_AMI}\n")
    print(f"scMSI NMI: {cur_NMI}\n")

