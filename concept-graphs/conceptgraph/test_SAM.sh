# # The ConceptGraphs-Detect 
# python scripts/generate_gsa_results.py \
#     --dataset_root $REPLICA_ROOT \
#     --dataset_config "/home/lg1/peteryu_workspace/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica.yaml" \
#     --scene_id "room0" \
#     --class_set "ram" \
#     --box_threshold 0.2 \
#     --text_threshold 0.2 \
#     --add_bg_classes \
#     --accumu_classes \
#     --exp_suffix withbg_allclasses



# python scripts/generate_gsa_results.py \
#     --dataset_root "/data/vln_datasets/matterport3d/v1/unzipped" \
#     --dataset_config "/home/lg1/peteryu_workspace/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
#     --scene_id "1LXtFkjw3qL" \
#     --class_set none 


# python scripts/generate_gsa_results.py \
#     --dataset_root "/data/vln_datasets/matterport3d/test" \
#     --dataset_config "/home/lg1/peteryu_workspace/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
#     --scene_id "1LXtFkjw3qL" \
#     --class_set none 

# The ConceptGraphs-Detect 
python scripts/generate_gsa_results.py \
    --dataset_root "/home/lg1/peteryu_workspace/m2g_vln/VLN-BEVBert/img_features" \
    --dataset_config "/home/lg1/peteryu_workspace/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
    --scene_id "17DRP5sb8fy" \
    --class_set "ram" \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses

    # --dataset_root "/data/vln_datasets/matterport3d/v1/unzipped" \

# python scripts/run_slam_rgb.py \
#     --dataset_root "/media/m2g/Data/Datasets/dataset/test" \
#     --dataset_config "/media/m2g/Data/Datasets/m2g_vln_server/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
#     --scene_id "1LXtFkjw3qL" \
#     --image_height 1024 \
#     --image_width 1280 \
#     --visualize