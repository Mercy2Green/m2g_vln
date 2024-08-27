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


# SCAN = "1LXtFkjw3qL"
# # The ConceptGraphs-Detect 
# python scripts/generate_gsa_results.py \
#     --dataset_root /data/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R \
#     --dataset_config "/home/lg1/peteryu_workspace/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
#     --scene_id $SCAN \
#     --class_set "ram" \
#     --box_threshold 0.2 \
#     --text_threshold 0.2 \
#     --add_bg_classes \
#     --accumu_classes \
#     --exp_suffix withbg_allclasses

    # --dataset_root "/data/vln_datasets/matterport3d/v1/unzipped" \

# python scripts/run_slam_rgb.py \
#     --dataset_root "/media/m2g/Data/Datasets/dataset/test" \
#     --dataset_config "/media/m2g/Data/Datasets/m2g_vln_server/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
#     --scene_id "1LXtFkjw3qL" \
#     --image_height 1024 \
#     --image_width 1280 \
#     --visualize

export CUDA_VISIBLE_DEVICES=0

# taskset -c 40-70 python scripts/generate_gsa_results.py \
#     --dataset_root /data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R \
#     --dataset_config "/home/lg1/peteryu_workspace/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
#     --class_set "ram" \
#     --box_threshold 0.2 \
#     --text_threshold 0.2 \
#     --add_bg_classes \
#     --accumu_classes \
#     --exp_suffix withbg_allclasses

taskset -c 40-70 python scripts/generate_gsa_results.py \
    --dataset_root /data2/vln_dataset/preprocessed_data/preprocessed_habitiat_R2R \
    --dataset_config "/home/lg1/peteryu_workspace/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
    --class_set "ram" \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses
