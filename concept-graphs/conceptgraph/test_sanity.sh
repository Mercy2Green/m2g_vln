# python scripts/run_slam_rgb.py \
#     --dataset_root "/data/vln_datasets/matterport3d/v1/unzipped" \
#     --dataset_config "/home/lg1/peteryu_workspace/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
#     --scene_id "1LXtFkjw3qL" \
#     --image_height 1024 \
#     --image_width 1280 \
#     --visualize

python scripts/run_slam_rgb.py \
    --dataset_root "/media/m2g/Data/Datasets/m2g_vln_server/m2g_vln/VLN-BEVBert/img_features" \
    --dataset_config "/media/m2g/Data/Datasets/m2g_vln_server/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
    --scene_id "17DRP5sb8fy" \
    --image_height 224 \
    --image_width 224 \
    --visualize