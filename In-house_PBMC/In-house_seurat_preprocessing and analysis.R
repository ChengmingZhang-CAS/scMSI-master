rm(list=ls())
setwd("E://scMSI_project//data/Inhouse_PBMC/")

library(Seurat)
library(dplyr)
library(SingleR)
library(celldex)
library(reshape2)
library(ggplot2)
library(SeuratDisk)

RNADATA <- read.csv("GSM4476363_RNA_raw.csv.gz")
RNADATA <- RNADATA[!duplicated(RNADATA[,1]),]
rownames(RNADATA) <- RNADATA[,1]
RNADATA <- RNADATA[,-1]
ADTDATA <- read.csv("GSM4476364_ADT_raw.csv",row.names = 1)
cell_info <- read.csv("truth_InHouse.csv")
# cell_info <- cell_info[cell_info$trueType != "unknown",]
cell_info$Barcode <- gsub("-",".",cell_info$Barcode)
RNADATA <- RNADATA[,colnames(RNADATA) %in% cell_info$Barcode]
ADTDATA <- ADTDATA[,colnames(ADTDATA) %in% cell_info$Barcode]
RNA <- CreateSeuratObject(RNADATA)
ADT <- CreateAssayObject(ADTDATA)
pbmc <- RNA
pbmc[["ADT"]] <- ADT
Assays(pbmc)
rownames(pbmc[["ADT"]])

DefaultAssay(pbmc)
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "MT-")
VlnPlot(pbmc, features = c("nCount_ADT", "nCount_RNA","percent.mt"), ncol = 3,
        log = TRUE, pt.size = 0) + NoLegend()
pbmc <- subset(
  x = pbmc,
  subset = nCount_ADT < 2.5e4 &
    nCount_ADT > 2000 &
    nCount_RNA < 9000 &
    nCount_RNA > 1000 &
    percent.mt < 10
)
pbmc@meta.data$truth_type <-  cell_info[cell_info$Barcode%in%rownames(pbmc@meta.data),]$trueType
DefaultAssay(pbmc) <- "RNA"
pbmc <- NormalizeData(pbmc) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()
DefaultAssay(pbmc) <- "ADT"
# we set a dimensional reduction name to avoid overwriting the
VariableFeatures(pbmc) <- rownames(pbmc[["ADT"]])
pbmc <- NormalizeData(pbmc, normalization.method = 'CLR', margin = 2) %>% 
  ScaleData() %>% RunPCA(reduction.name = 'apca')
pbmc <- pbmc[,pbmc@meta.data$truth_type != "unknown" ]
DefaultAssay(pbmc) <- "RNA"
SaveH5Seurat(pbmc, filename = "seuratObject_to_sacnpy.h5Seurat")
Convert("seuratObject_to_sacnpy.h5Seurat", dest = "h5ad")

# 整合
pbmc <- FindMultiModalNeighbors(
  pbmc, reduction.list = list("pca", "apca"),
  dims.list = list(1:17, 1:8), modality.weight.name = "RNA.weight"
)# 1:8 because ADT only have 10 features
pbmc <- FindClusters(pbmc, graph.name = "wsnn", verbose = F,resolution = 0.3)
head(Idents(pbmc),5)
pbmc <- RunUMAP(pbmc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")

# p1 <- DimPlot(pbmc, reduction = 'wnn.umap', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
# p1
# 
# p1 <- DimPlot(pbmc, reduction = 'wnn.umap', label = TRUE, repel = TRUE, label.size = 4) + NoLegend()
# p2 <- DimPlot(pbmc, reduction = 'wnn.umap', group.by = 'truth_type', label = TRUE, repel = TRUE, label.size = 4) + NoLegend()
# p2 + p1
# pbmc_markers <- FindAllMarkers(pbmc, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
# 
# new.cluster.ids <- c("0"="CD4+ T cells",
#                      "1"= "NK cells",
#                      "2"= "CD4+ T cells",
#                      "3"= "CD14+ monocytes",
#                      "4"="CD8+ T cells",
#                      "5"="CD8+ T cells",
#                      "6"= "B cell")
# pbmc <- RenameIdents(pbmc, new.cluster.ids)
# p_ADTandRNA_seuratv4 <- DimPlot(pbmc, reduction = "wnn.umap", label = TRUE, pt.size = 0.5) + NoLegend()
# saveRDS(p_ADTandRNA_seuratv4,file = "p_ADTandRNA_seuratv4.rds")
# 


# 
need_save_seurat <- data.frame(pbmc@meta.data$truth_type, pbmc@meta.data$wsnn_res.0.3, pbmc@reductions$wnn.umap@cell.embeddings)
rownames(need_save_seurat) <- colnames(pbmc)
write.csv(need_save_seurat,file = "seurat_output_in_house_PBMC.csv",row.names = T)
