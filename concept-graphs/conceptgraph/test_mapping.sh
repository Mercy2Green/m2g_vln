# Using the CoceptGraphs (without open-vocab detector)
THRESHOLD=1.2

python slam/cfslam_pipeline_batch.py \
    dataset_root="/data/vln_datasets/matterport3d/test" \
    dataset_config="/home/lg1/peteryu_workspace/m2g_vln/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml" \
    scene_id="1LXtFkjw3qL" \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.95 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.1 \
    gsa_variant=none \
    class_agnostic=True \
    skip_bg=True \
    max_bbox_area_ratio=0.5 \
    save_suffix=overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub \
    merge_interval=20 \
    merge_visual_sim_thresh=0.8 \
    merge_text_sim_thresh=0.8

# python scripts/visualize_cfslam_results.py --result_path "/data/vln_datasets/matterport3d/test/1LXtFkjw3qL/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub.pkl.gz"
# python scripts/visualize_cfslam_results.py --result_path "/media/m2g/Data/Datasets/dataset/test/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub.pkl.gz"

# # On the ConceptGraphs-Detect 
# SCENE_NAMES=room0
# THRESHOLD=1.2
# python slam/cfslam_pipeline_batch.py \
#     dataset_root=$REPLICA_ROOT \
#     dataset_config=$REPLICA_CONFIG_PATH \
#     stride=5 \
#     scene_id=$SCENE_NAME \
#     spatial_sim_type=overlap \
#     mask_conf_threshold=0.25 \
#     match_method=sim_sum \
#     sim_threshold=${THRESHOLD} \
#     dbscan_eps=0.1 \
#     gsa_variant=ram_withbg_allclasses \
#     skip_bg=False \
#     max_bbox_area_ratio=0.5 \
#     save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1