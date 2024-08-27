# export OPENAI_API_KEY=sk-IVEhr6VCYxeJIjNJ8776C2324cEd4b0b85C2C69e1b1e06A0
# export LLAVA_PYTHON_PATH=/home/lg1/peteryu_workspace/m2g_vln/LLaVA
# export LLAVA_MODEL_PATH=/home/lg1/peteryu_workspace/model/LLaVA-7B-v0

# export CUDA_VISIBLE_DEVICES=7

# SCENE_NAME=17DRP5sb8fy
# PKL_FILENAME=full_pcd_ram_withbg_allclasses_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz  # Change this to the actual output file name of the pkl.gz file

# REPLICA_ROOT=/data0/vln_datasets/preprocessed_data/test_data/testdata_cg_gpt4_replace

# python scenegraph/build_scenegraph_cfslam.py \
#     --mode extract-node-captions \
#     --cachedir ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache \
#     --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

export LLAVA_PYTHON_PATH=/home/lg1/peteryu_workspace/m2g_vln/LLaVA
export LLAVA_MODEL_PATH=/home/lg1/peteryu_workspace/model/LLaVA-7B-v0

export OPENAI_API_KEY=sk-IVEhr6VCYxeJIjNJ8776C2324cEd4b0b85C2C69e1b1e06A0

export DATA_ROOT=/data0/vln_datasets/preprocessed_data/test_data/testdata_cg_gpt4_replace

SCENE_NAME=17DRP5sb8fy
PKL_FILENAME=full_pcd_ram_withbg_allclasses_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub.pkl.gz # Change this to the actual output file name of the pkl.gz file

# python scenegraph/build_scenegraph_cfslam.py \
#     --mode extract-node-captions \
#     --cachedir ${DATA_ROOT}/${SCENE_NAME}/sg_cache \
#     --mapfile ${DATA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME} \
#     --class_names_file ${DATA_ROOT}/${SCENE_NAME}/gsa_classes_ram_withbg_allclasses.json

python scenegraph/build_scenegraph_cfslam.py \
    --mode refine-node-captions \
    --cachedir ${DATA_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${DATA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME} \
    --class_names_file ${DATA_ROOT}/${SCENE_NAME}/gsa_classes_ram_withbg_allclasses.json

# python scenegraph/build_scenegraph_cfslam.py \
#     --mode refine-node-captions \
#     --cachedir ${DATA_ROOT}/${SCENE_NAME}/sg_cache \
#     --mapfile ${DATA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME} \
#     --class_names_file ${DATA_ROOT}/${SCENE_NAME}/gsa_classes_ram_withbg_allclasses.json