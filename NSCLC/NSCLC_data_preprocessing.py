# %%% scMSI NSCLC process analysis %%%

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
# sys.path.append("/sibcb1/chenluonanlab6/yangyiwen")
from scMSI.scMSI_main import SCMSIRNA, SCMSIProtein, SCMSIRNAProtein
from scMSI.utils import read_txt, init_library_size
from scMSI.utils import get_top_coeff, get_knn_Aff
import time
import os

NSCLC_path = "E://scMSI_project//data//NSCLC//"


# 服务器
# lusc_path = "/sibcb1/chenluonanlab6/yangyiwen/lusc/"
# pre
def preprocess(data):
    # data = sc.read_10x_mtx(NSCLC_path + "LUAD/" + "raw_data_55", gex_only=False)
    data.var_names_make_unique()
    data_RNA = data[:, data.var["feature_types"] == "Gene Expression"].copy()
    data_ADT = data[:, data.var["feature_types"] == "Custom"].copy()
    protein_select = np.array([not p.startswith("HTO") for p in data_ADT.var_names])
    data_ADT = data_ADT[:, protein_select]
    data_RNA.uns["protein_names"] = np.array(data_ADT.var["gene_ids"])
    data_RNA.obsm["protein_expression"] = data_ADT.X.toarray()
    sc.pp.filter_cells(data_RNA, min_genes=500)
    sc.pp.filter_genes(data_RNA, min_cells=3)
    data_RNA.var['mt'] = data_RNA.var_names.str.startswith('MT-')
    rbc_gene_tmp = []
    for name in data_RNA.var_names:
        tmp_rbc = name in rbc_name
        rbc_gene_tmp.append(tmp_rbc)
    rbc_gene_tmp = np.array(rbc_gene_tmp)
    data_RNA.var["rbc_gene"] = rbc_gene_tmp
    epi_gene = []
    for name in data_RNA.var_names:
        tmp_name_epi = name in epithelial_gene
        epi_gene.append(tmp_name_epi)
    epi_gene = np.array(epi_gene)
    data_RNA.var["epithelial_gene"] = epi_gene.copy()
    sc.pp.calculate_qc_metrics(data_RNA, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(data_RNA, qc_vars=['rbc_gene'], percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(data_RNA, qc_vars=['epithelial_gene'], percent_top=None, log1p=False, inplace=True)
    data_RNA = data_RNA[data_RNA.obs.pct_counts_mt < 25, :]
    data_RNA = data_RNA[data_RNA.obs.pct_counts_rbc_gene < 10, :]
    data_RNA = data_RNA[data_RNA.obs.pct_counts_epithelial_gene < 1, :]
    data_RNA.layers["rna_expression"] = data_RNA.X.copy().toarray()
    sc.pp.normalize_total(data_RNA, target_sum=1e4)
    sc.pp.log1p(data_RNA)
    data_RNA.X = data_RNA.X.toarray()
    sc.pp.highly_variable_genes(
        data_RNA,
        n_top_genes=8000,
        flavor="seurat",
        # batch_key="batch",
        subset=True
    )
    return data_RNA


# The original mentioned the gene that needed to be removed
rbc_name = ["HBB", "HBA1", "HBA2"]
epithelial_gene = ["PLA2G2A", "CLCA1", "REG4", "S100A14", "ITLN1", "ELF3", "PIGR", "EPCAM", "REG1B", "REG1A", "REG3A",
                   "FABP1", "RBP2", "SST", "FABP2", "SPINK1", "FABP6", "AGR2", "AGR3", "CLDN3", "CLDN4", "DEFA6",
                   "DEFA5", "SPINK4", "ALDOB", "LCN2",
                   "MUC2", "KRT8", "KRT18", "TSPAN8", "OLFM4", "GPX2", "IFI27", "PHGR1", "MT1G", "CLDN7", "KRT19",
                   "FXYD3", "LGALS4", "FCGBP", "TFF3", "TFF1"]
# patient information
patient_581 = ["data_48"]
patient_593 = ["data_53"]
patient_596 = ["data_54", "data_55"]
patient_630 = ["data_56"]
patient_626 = ["data_88", "data_89"]
patient_695 = ["data_90", "data_91", "data_92", "data_93"]
patient_706 = ["data_342", "data_343", "data_344", "data_345", "data_346", "data_347", "data_348"]
patient_584 = ["data_51", "data_52"]
patient = [patient_581, patient_593, patient_596, patient_630, patient_626, patient_695, patient_706, patient_584]
patient_id = ["patient_581", "patient_593", "patient_596", "patient_630", "patient_626", "patient_695", "patient_706", "patient_584"]

# LUSC deal
file_lusc = os.listdir(NSCLC_path + "LUSC")
LUSC_Data = {}
for file in file_lusc:
    name = file.split("_", 1)[-1]
    print(name)
    tmp_data = sc.read_10x_mtx(NSCLC_path + "LUSC/" + file, gex_only=False)
    LUSC_Data["LUSC_" + name] = preprocess(tmp_data)
    LUSC_Data["LUSC_" + name].obs["batch"] = name
    for i in range(0, 8):
        if name in patient[i]:
            LUSC_Data["LUSC_" + name].obs["patient_info"] = patient_id[i]
    if name == "data_342":
        pass
    else:
        LUSC_common_genes = LUSC_Data["LUSC_data_342"].var_names.intersection(LUSC_Data["LUSC_" + name].var_names)
        LUSC_Data["LUSC_data_342"] = LUSC_Data["LUSC_data_342"][:, LUSC_common_genes]

LUSC_Data["LUSC_data_343"] = LUSC_Data["LUSC_data_343"][:, LUSC_common_genes]
LUSC_Data["LUSC_data_344"] = LUSC_Data["LUSC_data_344"][:, LUSC_common_genes]
LUSC_Data["LUSC_data_345"] = LUSC_Data["LUSC_data_345"][:, LUSC_common_genes]
LUSC_Data["LUSC_data_346"] = LUSC_Data["LUSC_data_346"][:, LUSC_common_genes]
LUSC_Data["LUSC_data_347"] = LUSC_Data["LUSC_data_347"][:, LUSC_common_genes]
LUSC_Data["LUSC_data_348"] = LUSC_Data["LUSC_data_348"][:, LUSC_common_genes]
LUSC_Data["LUSC_data_51"] = LUSC_Data["LUSC_data_51"][:, LUSC_common_genes]
LUSC_Data["LUSC_data_52"] = LUSC_Data["LUSC_data_52"][:, LUSC_common_genes]
for file in file_lusc:
    name = file.split("_", 1)[-1]
    LUSC_Data["LUSC_" + name].obsm["protein_expression"] = pd.DataFrame(
        LUSC_Data["LUSC_" + name].obsm["protein_expression"],
        columns=LUSC_Data["LUSC_" + name].uns["protein_names"],
        index=LUSC_Data["LUSC_" + name].obs_names,
    )
    del LUSC_Data["LUSC_" + name].uns["protein_names"]


for file in file_lusc:
    name = file.split("_", 1)[-1]
    if name == "data_342":
        pass
    elif name == "data_343":
        LUSC_adata = ad.concat([LUSC_Data["LUSC_data_342"], LUSC_Data["LUSC_" + name]], join="inner", axis=0)
        LUSC_adata.obs_names_make_unique()
    else:
        LUSC_adata = ad.concat([LUSC_adata, LUSC_Data["LUSC_" + name]], join="inner", axis=0)
        LUSC_adata.obs_names_make_unique()
        labes_num = list(set(LUSC_adata.obs["batch"]))
        print(labes_num)
LUSC_adata.obsm["protein_expression"] = LUSC_adata.obsm["protein_expression"].fillna(0)
LUSC_adata.uns['protein_names'] = np.array(LUSC_adata.obsm["protein_expression"].columns)
LUSC_adata.obsm["protein_expression"] = np.array(LUSC_adata.obsm["protein_expression"])
LUSC_adata.uns["n_batch"] = 9

# LUAD deal
file_luad = os.listdir(NSCLC_path + "LUAD")
luad_Data = {}

for file in file_luad:
    name = file.split("_", 1)[-1]
    print(name)
    tmp_data = sc.read_10x_mtx(NSCLC_path + "luad/" + file, gex_only=False)
    luad_Data["luad_" + name] = preprocess(tmp_data)
    luad_Data["luad_" + name].obs["batch"] = name
    for i in range(0, 8):
        if name in patient[i]:
            luad_Data["luad_" + name].obs["patient_info"] = patient_id[i]
    if name == "data_48":
        pass
    else:
        luad_common_genes = luad_Data["luad_data_48"].var_names.intersection(luad_Data["luad_" + name].var_names)
        luad_Data["luad_data_48"] = luad_Data["luad_data_48"][:, luad_common_genes]

for file in file_luad:
    name = file.split("_", 1)[-1]
    if name == "data_48":
        pass
    else:
        luad_Data["luad_" + name] = luad_Data["luad_" + name][:, luad_common_genes]

for file in file_luad:
    name = file.split("_", 1)[-1]
    luad_Data["luad_" + name].obsm["protein_expression"] = pd.DataFrame(
        luad_Data["luad_" + name].obsm["protein_expression"],
        columns=luad_Data["luad_" + name].uns["protein_names"],
        index=luad_Data["luad_" + name].obs_names,
    )
    del luad_Data["luad_" + name].uns["protein_names"]

for file in file_luad:
    name = file.split("_", 1)[-1]
    if name == "data_48":
        pass
    elif name == "data_53":
        LUAD_adata = ad.concat([luad_Data["luad_data_48"], luad_Data["luad_" + name]], join="inner", axis=0)
        LUAD_adata.obs_names_make_unique()
    else:
        LUAD_adata = ad.concat([LUAD_adata, luad_Data["luad_" + name]], join="inner", axis=0)
        LUAD_adata.obs_names_make_unique()
        LUAD_adata.obs["batch"] = LUAD_adata.obs["batch"]
        labes_num = list(set(LUAD_adata.obs["batch"]))
        print(labes_num)
LUAD_adata.obsm["protein_expression"] = LUAD_adata.obsm["protein_expression"].fillna(0)
LUAD_adata.uns['protein_names'] = np.array(LUAD_adata.obsm["protein_expression"].columns)
LUAD_adata.obsm["protein_expression"] = np.array(LUAD_adata.obsm["protein_expression"])
LUAD_adata.uns["n_batch"] = 11

# batch numerical
class_encoder = LabelEncoder()
LUSC_adata.obs["batch_info"] = LUSC_adata.obs["batch"]
LUAD_adata.obs["batch_info"] = LUAD_adata.obs["batch"]
LUSC_adata.obs["batch"] = class_encoder.fit_transform(LUSC_adata.obs["batch"].values)
LUAD_adata.obs["batch"] = class_encoder.fit_transform(LUAD_adata.obs["batch"].values)


LUSC_adata.write("./LUSC_adata.h5ad")
LUAD_adata.write("./LUAD_adata.h5ad")
# labes_num = list(set(LUAD_adata.obs["batch"]))
