# SCENE_NAMES=17DRP5sb8fy
# THRESHOLD=1.2
# python cg_process/cg_processing.py \
#     dataset_root=/media/m2g/Data/Datasets/m2g_vln_server/m2g_vln/VLN-BEVBert/img_features \
#     dataset_config=/media/m2g/Data/Datasets/BEV_HGT_VLN/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml \
#     scene_id=$SCENE_NAMES \
#     spatial_sim_type=overlap \
#     mask_conf_threshold=0.25 \
#     match_method=sim_sum \
#     sim_threshold=${THRESHOLD} \
#     dbscan_eps=0.1 \
#     gsa_variant=ram_withbg_allclasses \
#     skip_bg=False \
#     max_bbox_area_ratio=0.5 \
#     save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1 


export CUDA_VISIBLE_DEVICES=7

taskset -c 40-60 python cg_process/cg_processing.py