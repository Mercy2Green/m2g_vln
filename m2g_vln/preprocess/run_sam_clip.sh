python m2g_vln/get_object_feature.py \
    --checkpoint_file_segment /media/m2g/Data/Datasets/model/sam_vit_h_4b8939.pth \
    --connectivity_dir /media/m2g/Data/Datasets/m2g_vln/m2g_vln/preprocess/m2g_vln/dataset/connectivity_dir/test \
    --scan_dir /media/m2g/Data/Datasets/dataset/v1/unzipped \
    --output_file /media/m2g/Data/Datasets/object_feature/obj_feats.hdf5 \
    --num_batches 1

# parser.add_argument('--checkpoint_file_segment', default=None)
# parser.add_argument('--connectivity_dir', default='../datasets/R2R/connectivity')
# parser.add_argument('--scan_dir', default='../data/v1/scans')
# parser.add_argument('--output_file')
# parser.add_argument('--batch_size', default=4, type=int)
# parser.add_argument('--num_workers', type=int, default=1)
# parser.add_argument('--num_batches', type=int, default=2)