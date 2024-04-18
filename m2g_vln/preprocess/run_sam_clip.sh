python m2g_vln/get_object_feature.py \
    --checkpoint_file_segment /home/lg1/peteryu_workspace/model/sam_vit_h_4b8939.pth \
    --connectivity_dir /home/lg1/peteryu_workspace/m2g_vln/m2g_vln/preprocess/m2g_vln/dataset/connectivity_dir/R2R \
    --scan_dir /data/vln_datasets/matterport_skybox/v1/unzipped \
    --output_file /home/lg1/peteryu_workspace/output/object_feature/obj_feats_R2R.hdf5 \
    --output_dir /home/lg1/peteryu_workspace/output/object_feature \
    --num_workers 1 \
    --num_batches 1 \
    --class_set none 



#     SCENE_NAME=room0

# # The CoceptGraphs (without open-vocab detector)
# python scripts/generate_gsa_results.py \
#     --dataset_root $REPLICA_ROOT \
#     --dataset_config $REPLICA_CONFIG_PATH \
#     --scene_id $SCENE_NAME \
#     --class_set none \
#     --stride 5

# # The ConceptGraphs-Detect 
# CLASS_SET=ram
# python scripts/generate_gsa_results.py \
#     --dataset_root $REPLICA_ROOT \
#     --dataset_config $REPLICA_CONFIG_PATH \
#     --scene_id $SCENE_NAME \
#     --class_set $CLASS_SET \
#     --box_threshold 0.2 \
#     --text_threshold 0.2 \
#     --stride 5 \
#     --add_bg_classes \
#     --accumu_classes \
#     --exp_suffix withbg_allclasses

# parser.add_argument('--checkpoint_file_segment', default=None)
# parser.add_argument('--connectivity_dir', default='../datasets/R2R/connectivity')
# parser.add_argument('--scan_dir', default='../data/v1/scans')
# parser.add_argument('--output_file')
# parser.add_argument('--batch_size', default=4, type=int)
# parser.add_argument('--num_workers', type=int, default=1)
# parser.add_argument('--num_batches', type=int, default=2)


    # parser.add_argument(
    #     "--dataset_root", type=Path, required=True,
    # )
    # parser.add_argument(
    #     "--dataset_config", type=str, required=True,
    #     help="This path may need to be changed depending on where you run this script. "
    # )
    
    # parser.add_argument("--scene_id", type=str, default="train_3")
    
    # parser.add_argument("--start", type=int, default=0)
    # parser.add_argument("--end", type=int, default=-1)
    # parser.add_argument("--stride", type=int, default=1)

    # parser.add_argument("--desired-height", type=int, default=480)
    # parser.add_argument("--desired-width", type=int, default=640)

    # parser.add_argument("--box_threshold", type=float, default=0.25)
    # parser.add_argument("--text_threshold", type=float, default=0.25)
    # parser.add_argument("--nms_threshold", type=float, default=0.5)

    # parser.add_argument("--class_set", type=str, default="scene", 
    #                     choices=["scene", "generic", "minimal", "tag2text", "ram", "none"], 
    #                     help="If none, no tagging and detection will be used and the SAM will be run in dense sampling mode. ")
    # parser.add_argument("--add_bg_classes", action="store_true", 
    #                     help="If set, add background classes (wall, floor, ceiling) to the class set. ")
    # parser.add_argument("--accumu_classes", action="store_true",
    #                     help="if set, the class set will be accumulated over frames")

    # parser.add_argument("--sam_variant", type=str, default="sam",
    #                     choices=['fastsam', 'mobilesam', "lighthqsam"])
    
    # parser.add_argument("--save_video", action="store_true")
    
    # parser.add_argument("--device", type=str, default="cuda")
    
    # parser.add_argument("--use_slow_vis", action="store_true", 
    #                     help="If set, use vis_result_slow_caption. Only effective when using ram/tag2text. ")
    
    # parser.add_argument("--exp_suffix", type=str, default=None,
    #                     help="The suffix of the folder that the results will be saved to. ")