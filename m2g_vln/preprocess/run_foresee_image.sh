python m2g_vln/foresee_the_sim_image.py \
    --checkpoint_file_segment /home/lg1/peteryu_workspace/model/sam_vit_h_4b8939.pth \
    --connectivity_dir /home/lg1/peteryu_workspace/m2g_vln/m2g_vln/preprocess/m2g_vln/dataset/connectivity_dir/test \
    --scan_dir /data/vln_datasets/matterport_skybox/v1/unzipped \
    --output_file /home/lg1/peteryu_workspace/output/object_feature/obj_feats_R2R.hdf5 \
    --output_dir /home/lg1/peteryu_workspace/output/foresee_image 


# parser.add_argument('--checkpoint_file_segment', default=None)
# parser.add_argument('--connectivity_dir', default='../datasets/R2R/connectivity')
# parser.add_argument('--scan_dir', default='../data/v1/scans')
# parser.add_argument('--output_file')
# parser.add_argument('--batch_size', default=4, type=int)
# parser.add_argument('--num_workers', type=int, default=1)
# parser.add_argument('--num_batches', type=int, default=2)