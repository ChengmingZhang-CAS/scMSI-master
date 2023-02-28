# %%% scMSI LUAD/SC removing batch %%%
# people batch and Experimental batch

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
import sys
sys.path.append("/sibcb1/chenluonanlab6/yangyiwen")
from scMSI.scMSI_main import SCMSIRNA, SCMSIProtein, SCMSIRNAProtein
from scMSI.utils import read_txt, init_library_size
from scMSI.utils import get_top_coeff, get_knn_Aff
import time
import os

# NSCLC_path = "E:/scMSI_project//data//NSCLC//"

# 服务器
NSCLC_path = "/sibcb1/chenluonanlab6/yangyiwen/NSCLC/"
LUSC_adata = ad.read_h5ad(NSCLC_path + "LUSC_adata.h5ad")
LUAD_adata = ad.read_h5ad(NSCLC_path + "LUAD_adata.h5ad")
LUSC_adata.obs["data_info"] = "LUSC"
LUAD_adata.obs["data_info"] = "LUAD"
NSCLC_data_inner = ad.concat([LUSC_adata, LUAD_adata], join="inner", axis=0)
NSCLC_data_outer = ad.concat([LUSC_adata, LUAD_adata], join="outer", axis=0)
NSCLC_data_outer.obsm["protein_expression"].shape
NSCLC_data_inner.obsm["protein_expression"].shape
NSCLC_data = ad.concat([LUSC_adata, LUAD_adata], join="outer", axis=0)
NSCLC_data.X = pd.DataFrame(NSCLC_data.X).fillna(0)
NSCLC_data.layers["rna_expression"] = pd.DataFrame(NSCLC_data.layers["rna_expression"]).fillna(0)
len(set(NSCLC_data.obs["batch_info"]))
len(set(NSCLC_data.obs["batch"]))
class_encoder = LabelEncoder()
NSCLC_data.obs["batch"] = class_encoder.fit_transform(NSCLC_data.obs["patient_info"].values)  # 先去人的batch
len(set(NSCLC_data.obs["batch"]))
NSCLC_data.uns["n_batch"] = 8  # person batch
NSCLC_data.uns["protein_names"] = LUAD_adata.uns["protein_names"]

# train model
# train RNA model
max_epochs = 200
# rna_model = SCMSIRNA(
#     NSCLC_data,
#     n_latent=10,
#     latent_distribution="normal",
# )
# a = time.time()
# rna_model.train(max_epochs=max_epochs, n_sample_ref=1000)
# b = time.time()
# print("RNA_use_time: ", b - a)
#
# protein_model = SCMSIProtein(
#     NSCLC_data,
#     n_latent=10,
#     latent_distribution="normal",
#     n_hidden=256
# )
# a = time.time()
# protein_model.train(max_epochs=max_epochs)
# b = time.time()
# print("Protein_use_time: ", b - a)

NSCLC_model = SCMSIRNAProtein(
    NSCLC_data,
    n_latent=10,
    latent_distribution="normal",
    n_sample_ref=1000
)

NSCLC_model.train(max_epochs=200, record_loss=True)
# torch.save(NSCLC_model.module.state_dict(), './NSCLC_data_batch0.pt')
# latent_rna = rna_model.get_latent_representation(batch_size=128)
# NSCLC_data.obsm["rna_latent"] = latent_rna
#
#
# coeff_rna = rna_model.get_coeff()
# coeff_rna = coeff_rna / norm(coeff_rna, ord=1, axis=1, keepdims=True)
# adata_coeff_rna = ad.AnnData(coeff_rna)
# sc.pp.pca(adata_coeff_rna)
# NSCLC_data.obsm["coeff_rna"] = adata_coeff_rna
#
# latent_protein = protein_model.get_latent_representation(batch_size=128)
# NSCLC_data.obsm["protein_latent"] = latent_protein

rna_latent, pro_latent = NSCLC_model.get_latent_representation(batch_size=128)
NSCLC_data.obsm["rna_latent_mix"] = rna_latent
NSCLC_data.obsm["pro_latent_mix"] = pro_latent
nsclc_latent = np.concatenate([NSCLC_data.obsm["rna_latent_mix"], NSCLC_data.obsm["pro_latent_mix"]], axis=1)
NSCLC_data.obsm["nsclc_latent"] = nsclc_latent
rna_coeff, pro_coeff, coeff = NSCLC_model.get_coeff(pro_w=1.0, ord=2)
coeff_pca = PCA(n_components=20, svd_solver="arpack", random_state=0).fit_transform(coeff)  # svd_solver="arpack"
NSCLC_data.obsm["nsclc_coeff"] = coeff_pca
NSCLC_data.obsm["denoised_rna"], NSCLC_data.obsm["denoised_protein"] = NSCLC_model.get_normalized_expression(
    n_samples=25,
    return_mean=True,
    transform_batch=[0]  # [0, 1]
)

NSCLC_data.write("./NSCLC_adata_removing_personbatch_analysis_outer.h5ad")
