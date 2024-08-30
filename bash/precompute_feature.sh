
# # precompute grid_mp3d_clip.py
# python3 precompute_features/grid_mp3d_clip.py \
#     --scan_dir /data0/vln_datasets/matterport3d/v1/unzipped \
#     --model_name /home/lg1/peteryu_workspace/model/ViT-B-16.pt \
#     --num_workers 8


# precompute grid_mp3d_imagenet.py
# python3 precompute_features/grid_mp3d_imagenet.py \
#     --scan_dir "/data0/vln_datasets/matterport3d/v1/unzipped" \
#     --num_workers 4

# # precompute grid_depth.py
# python3 precompute_features/grid_depth.py

# # precompute grid_sem.py
# python precompute_features/grid_sem.py

# # precompute grid_habitat_clip.py
# python precompute_features/grid_habitat_clip.py \
#     --num_workers 1 \
#     --model_name /home/lg1/peteryu_workspace/model/ViT-B-16.pt

# # # precompute save_habitat_img.py --img_type depth
# python precompute_features/save_habitat_img.py \
#     --img_type  depth \
#     --scan_dir  /data0/vln_datasets/mp3d/v1/tasks/mp3d \
#     --num_workers  1

# # precompute python precompute_features/save_depth_feature.py
# python precompute_features/save_depth_feature.py

# python precompute_features/save_mp3d_img.py \
#     --scan_dir  /data0/vln_datasets/matterport3d/v1/unzipped \
#     --num_workers  1

python precompute_features/save_depth_feature.py \
    --model_name resnet50 \
    --checkpoint_file /home/lg1/peteryu_workspace/model/gibson-2plus-resnet50.pth \
    --scan_dir /data0/vln_datasets/matterport3d/v1/unzipped \
    --img_db /home/lg1/peteryu_workspace/m2g_vln/VLN-BEVBert/img_features/habitat_256x256_vfov90_depth.hdf5 \
    --num_workers 1

