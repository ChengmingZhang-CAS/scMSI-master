# Contrastively generative self-expression model for single-cell and spatial multimodal data
![image](/Utilities/scMSI_overview.jpg)

Overview of scMSI model. scMSI is a deep variational autoencoder framework with a contrastive self-expressive layer, which integrates the multi-level omics data by learning the consistent sample affinity (Figure A, B). For illustrative purposes, we take CITE-seq data as an example, which allows paired measurements of RNA and surface proteins in a single cell. Suppose that $X^R∈R^(p^R×N)$ and $X^P∈R^(p^P×N)$ are represented as the matrices of RNA and protein unique molecular identifier (UMI) counts, where $p^R$ is the number of all detected genes, $p^P$ is the corresponding number of proteins, and N is the total number of cells. Categorical covariates s, such as experimental batch or donor, are optional inputs for integrating different datasets and are referred to as ‘batches’ in the text.
scMSI consists of three modular neural networks as components:
1. the encoder component;2. the self-expressive component;3. the decoder component
scMSI provides a paradigm to integrate multiple omics data even with weak relations by contrastive loss, which learns consistent affinity matrix between cells (i.e., self-expressive coefficients). scMSI uses a unified sample relationship/affinity to integrate the multi-omics data based on the contrastive self-expressive layer, which captures consistent and complementary information of different modalities and enables superior performance on the heterogeneous single-cell multimodal data.

# Installation

## Install scMSI

Installation was tested on Windows10 with Python 3.8.15 and torch 1.13.0 on a machine with one 8-core Intel(R) Xeon(R) Gold 2140B CPU addressing with 128GB RAM.
 scMSI is implemented in the Pytorch framework. Please run scMSI on CUDA if possible.

#### 1. Grab source code of scMSI

```
git clone https://github.com/yiwen-yang/scMSI.git

cd scMSI
```

#### 2. Install scMSI in the virtual environment by conda

* Firstly, install conda: https://docs.anaconda.com/anaconda/install/index.html

* Then, automatically install all used packages (described by "used_package.txt") for scMSI in a few mins.

```
conda create -n scMSI python=3.8

source activate

conda activate scMSI

pip install -r used_package.txt
```
