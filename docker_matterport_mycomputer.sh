# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install tensorboardX easydict transformers sklearn jsonlines torch_scatter

#! /bin/bash

export MATTERPORT_DATA_DIR=/media/m2g/Data/Datasets/dataset/v1/unzipped

export MATTERPORT_SIMULATOR_DIR=/media/m2g/Data/Datasets/m2g_vln_server/m2g_vln/Matterport3DSimulator_opencv4

export BEVBERT_DIR=/media/m2g/Data/Datasets/m2g_vln_server/m2g_vln/VLN-BEVBert

# Habitat dataset for BEV-BERT

export HABITAT_DATA_DIR=/media/m2g/Data/Datasets/dataset/Matterport3D_scene_meshes/data/scene_datasets/mp3d

# local model path

# export LOCAL_MODEL_DIR=/home/lg1/peteryu_workspace/model

# export MATTERPORT_SIMULATOR_DIR=/media/m2g/Data/Datasets/m2g_vln/Matterport3DSimulator
# export MATTERPORT_SIMULATOR_OPENCV4_DIR=/media/m2g/Data/Datasets/m2g_vln/Matterport3DSimulator_opencv4

docker run -ti --gpus '"device=0"' \
    --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans \
    --mount type=bind,source=$MATTERPORT_SIMULATOR_DIR,target=/root/mount/Matterport3DSimulator \
    --mount type=bind,source=$BEVBERT_DIR,target=/root/mount/VLN-BEVBert \
    --mount type=bind,source=$HABITAT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/scene_datasets/mp3d \
    mattersim:11.3.0-devel-ubuntu20.04

#    mattersim:9.2-devel-ubuntu18.04
# python precompute_features/grid_mp3d_clip.py --scan_dir=/root/mount/Matterport3DSimulator/data/v1/scans

# mattersim:11.8.0-cudnn8-devel-ubuntu20.04

# docker run --gpus all -it\
#     --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans \
#     --mount type=bind,source=$MATTERPORT_SIMULATOR_DIR,target=/root/mount/Matterport3DSimulator \
#     --mount type=bind,source=$BEVBERT_DIR,target=/root/mount/VLN-BEVBert \
#     mattersim:11.8.0-cudnn8-devel-ubuntu20.04