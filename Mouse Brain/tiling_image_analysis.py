# %%% stlearn analysis %%%

import stlearn as st
from pathlib import Path

st.settings.set_figure_params(dpi=300)
# img_path = Path("E://scMSI_project//data//V1_Breast_Cancer//data")
img_path = Path(
    "E:/scMSI_project/data/Mouse Brain Serial Section 1 (Sagittal-Anterior)/data")
# spot tile is the intermediate result of image pre-processing
TILE_PATH = Path(
    "E:/scMSI_project/data/Mouse Brain Serial Section 1 (Sagittal-Anterior)/data//image_segmentation")
TILE_PATH.mkdir(parents=True, exist_ok=True)

# output path
OUT_PATH = Path(
    "E:/scMSI_project/data/Mouse Brain Serial Section 1 (Sagittal-Anterior)/data//Mouse_Brain_output")
OUT_PATH.mkdir(parents=True, exist_ok=True)

# load data
data = st.Read10X(
    "E:/scMSI_project/data/Mouse Brain Serial Section 1 (Sagittal-Anterior)/data",
                       count_file="V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5")

# st.pp.normalize_total(data)
# st.pp.log1p(data)
# pre-processing for spot image
st.pp.tiling(data, TILE_PATH)
# this step uses deep learning model to extract high-level features from tile images
# may need few minutes to be completed
st.pp.extract_feature(data)

data.obsm['X_morphology'].shape
save_path = "E:/scMSI_project/data/Mouse Brain Serial Section 1 (Sagittal-Anterior)/data//"
data.write(save_path + "Mouse_brain_ST_adata.h5ad")
