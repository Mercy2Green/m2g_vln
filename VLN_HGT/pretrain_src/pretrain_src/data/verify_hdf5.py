import h5py

def verify_hdf5_file(file_path, expected_viewpoint_size):
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data = f[key][...]
            if data.shape[0] != expected_viewpoint_size:
                print(f"Dataset {key} does not contain {expected_viewpoint_size} views. Found {data.shape[0]} views instead.")
            else:
                print(f"Dataset {key} contains {expected_viewpoint_size} views.")

if __name__ == '__main__':
    hdf5_file_path = '/home/lg1/lujia/VLN_HGT/pretrain_src/img_features/CLIP-ViT-B-16-views-habitat.hdf5'  # 替换为你的 HDF5 文件路径
    expected_viewpoint_size = 12  # 预期的视图数量
    verify_hdf5_file(hdf5_file_path, expected_viewpoint_size)