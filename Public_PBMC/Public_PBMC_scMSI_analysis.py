# %%% scMSI Public PBMC analysis %%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy.stats import pearsonr
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

# debug_mode = True
debug_mode = False
data_path = "data"
if debug_mode:
    data_path = "../data"

pbmc_path = "E:\\benchmark multi-omics analysis\\data_for_benchmark\\public_PBMC\\"
# seurat_to_scanpy
TOselect_feature = sc.read_h5ad(pbmc_path + "seuratObject_to_sacnpy.h5ad")
feature_select = TOselect_feature.var["vst.variable"].keys()
cell_select = TOselect_feature.obs["nCount_RNA"].keys()
# RNA read
pbmc = sc.read_10x_h5(pbmc_path + "pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5",  gex_only=False)
pbmc.var_names_make_unique()
pbmc.layers["rna_expression"] = pbmc.X.copy()
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
adata.raw = adata
scMSI_output_times10 = pd.DataFrame()
model = SCMSIRNAProtein(
    adata,
    n_latent=10,
    latent_distribution="normal",
)
max_epochs = 400
a = time.time()
model.train(max_epochs=max_epochs, record_loss=True)
b = time.time()
print("time: ", b - a)
rna_latent, pro_latent = model.get_latent_representation(batch_size=128)
adata.obsm["rna_latent"] = rna_latent
adata.obsm["pro_latent"] = pro_latent
rna_rec_latent, pro_rec_latent = model.get_reconstruct_latent(batch_size=128)
adata.obsm["rna_rec_latent"] = rna_rec_latent
adata.obsm["pro_rec_latent"] = pro_rec_latent
mix_latent = np.concatenate([adata.obsm["rna_latent"], adata.obsm["pro_latent"]], axis=1)
adata.obsm["latent"] = mix_latent
rna_coeff, pro_coeff, coeff = model.get_coeff(pro_w=1.0, ord=2)
coeff_pca = PCA(n_components=20, svd_solver="arpack", random_state=0).fit_transform(coeff)
adata.obsm["coeff_pca"] = coeff_pca
sc.set_figure_params(figsize=(16, 9))
new_adata = adata.copy()
use_rep_keys = ["latent", "coeff_pca"]
for use_rep in use_rep_keys:
    print("*" * 100)
    print("use_rep: ", use_rep)
    sc.pp.neighbors(new_adata, use_rep=use_rep)
    sc.tl.leiden(new_adata, key_added="leiden", resolution=0.3)
    sc.tl.umap(new_adata, min_dist=0.4)
    sc.pl.umap(new_adata, color="leiden", frameon=False, legend_loc="on data", title=use_rep)



