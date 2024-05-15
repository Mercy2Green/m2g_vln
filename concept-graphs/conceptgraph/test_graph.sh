# sk-IVEhr6VCYxeJIjNJ8776C2324cEd4b0b85C2C69e1b1e06A0
# Set the env variables as follows (change the paths accordingly)
export LLAVA_PYTHON_PATH=/home/lg1/peteryu_workspace/m2g_vln/LLaVA
export LLAVA_MODEL_PATH=/home/lg1/peteryu_workspace/model/LLaVA-7B-v0

export OPENAI_API_KEY=sk-IVEhr6VCYxeJIjNJ8776C2324cEd4b0b85C2C69e1b1e06A0

export DATA_ROOT=/home/lg1/peteryu_workspace/m2g_vln/VLN-BEVBert/img_features

SCENE_NAME=17DRP5sb8fy
PKL_FILENAME=full_pcd_ram_withbg_allclasses_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub.pkl.gz # Change this to the actual output file name of the pkl.gz file

python scenegraph/build_scenegraph_cfslam.py \
    --mode extract-node-captions \
    --cachedir ${DATA_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${DATA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME} \
    --class_names_file ${DATA_ROOT}/${SCENE_NAME}/gsa_classes_ram_withbg_allclasses.json

# python scenegraph/build_scenegraph_cfslam.py \
#     --mode refine-node-captions \
#     --cachedir ${DATA_ROOT}/${SCENE_NAME}/sg_cache \
#     --mapfile ${DATA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME} \
#     --class_names_file ${DATA_ROOT}/${SCENE_NAME}/gsa_classes_ram_withbg_allclasses.json

# python scenegraph/build_scenegraph_cfslam.py \
#     --mode build-scenegraph \
#     --cachedir ${DATA_ROOT}/${SCENE_NAME}/sg_cache \
#     --mapfile ${DATA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME} \
#     --class_names_file ${DATA_ROOT}/${SCENE_NAME}/gsa_classes_ram_withbg_allclasses.json