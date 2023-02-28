rm(list=ls())
setwd("E://benchmark multi-omics analysis//data_for_benchmark//public_PBMC/")

library(Seurat)
library(dplyr)
library(SingleR)
library(celldex)
library(reshape2)
library(ggplot2)
library(SeuratDisk)
library(cluster)

RNAADTcount <- Read10X_h5("pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5")
cell_info <- read.csv("truth_10X10k.csv")
# extract RNA and ADT data
RNAcount <- RNAADTcount$`Gene Expression`
proteincount <- RNAADTcount$`Antibody Capture`
proteincount <- proteincount[1:14,]
rownames(proteincount)
# Creat Seurat object
pbmc <- CreateSeuratObject(counts = RNAcount)
# Now add in the ADT data
pbmc_ADT <- CreateAssayObject(counts = proteincount)
pbmc[["ADT"]] <- pbmc_ADT
Assays(pbmc)
rownames(pbmc[["ADT"]])

DefaultAssay(pbmc)
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
VlnPlot(pbmc, features = c("nCount_ADT", "nCount_RNA","percent.mt"), ncol = 3,
        log = TRUE, pt.size = 0) + NoLegend()
pbmc <- subset(
  x = pbmc,
  subset = nCount_ADT < 3e4 &
    nCount_ADT > 1e3 &
    nCount_RNA < 25000 &
    nCount_RNA > 1000 &
    percent.mt < 18
)

# ANALYSIS
DefaultAssay(pbmc) <- "RNA"
pbmc <- NormalizeData(pbmc) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()
DefaultAssay(pbmc) <- "ADT"
# we set a dimensional reduction name to avoid overwriting the
VariableFeatures(pbmc) <- rownames(pbmc[["ADT"]])
pbmc <- NormalizeData(pbmc, normalization.method = 'CLR', margin = 2) %>% 
  ScaleData() %>% RunPCA(reduction.name = 'apca')

DefaultAssay(pbmc) <- "RNA"
SaveH5Seurat(pbmc, filename = "seuratObject_to_sacnpy.h5Seurat")
Convert("seuratObject_to_sacnpy.h5Seurat", dest = "h5ad")

pbmc@meta.data$truth_type <-  cell_info[cell_info$Barcode%in%rownames(pbmc@meta.data),]$trueType
pbmc <- pbmc[,pbmc@meta.data$truth_type != "unknown" ]
# analysis
pbmc <- FindMultiModalNeighbors(
  pbmc, reduction.list = list("pca", "apca"),
  dims.list = list(1:17, 1:10), modality.weight.name = "RNA.weight"
)# 1:10 because ADT only have 14 features
pbmc <- FindClusters(pbmc, graph.name = "wsnn", verbose = F, resolution = 0.3)
head(Idents(pbmc),5)
pbmc <- RunUMAP(pbmc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")

head(pbmc@meta.data)
pbmc@reductions$wnn.umap@cell.embeddings
need_save_seurat <- data.frame(pbmc@meta.data$truth_type, pbmc@meta.data$wsnn_res.0.3, pbmc@reductions$wnn.umap@cell.embeddings)
rownames(need_save_seurat) <- colnames(pbmc)

write.csv(need_save_seurat,file = "seurat_output_public_PBMC_add_umap.csv",row.names = T)
