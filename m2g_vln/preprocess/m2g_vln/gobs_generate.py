

import gzip
import pickle
from pathlib import Path


def get_gobs4CG(scan, gt_path):  
    # input: scan, gt_path
    # output: gobs

    # scan is  a string.
    # gt_path is a list of strings, or np.array of strings, or other type of list.

    gobs = {
        "xyxy": [],
        "confidence": [],
        "class_id": [],
        "mask": [],
        "classes": [],
        "image_crops": [],
        "image_feats": [],
        "text_feats": [],
        "tagging_caption": [],
        "tagging_text_prompt": []
    }

    vp_path_list = []
    vp_path_list = gt_path

    data_dir = Path("/media/m2g/Data/Datasets/dataset/gsa_result")

    for vp in vp_path_list:

        detections_file = data_dir / f"{scan}" / f"gsa_detections_{scan}_{vp}.pkl.gz"

        with gzip.open(detections_file, "rb") as f:
            data_list = pickle.load(f)

        ## simply just append the data.
        for data in data_list:
            for key in gobs.keys():
                if key in data:
                    gobs[key].extend(data[key] if isinstance(data[key], list) else [data[key]])

    return gobs


def main():

    data_dir = Path("/media/m2g/Data/Datasets/dataset/gsa_result_17DRP5sb8fy")

    scan = "17DRP5sb8fy"
    gt_path = ["0e92a69a50414253a23043758f111cec", "00ebbf3782c64d74aaf7dd39cd561175", "08c774f20c984008882da2b8547850eb"]

    gobs = get_gobs4CG(scan, gt_path)

    print(gobs)



if __name__ == "__main__":
    main()
    


        # # load grounded SAM detections
        # gobs = None # stands for grounded SAM observations

        # color_path = Path(color_path)
        # detections_path = color_path.parent.parent / cfg.detection_folder_name / color_path.name
        # detections_path = detections_path.with_suffix(".pkl.gz")
        # color_path = str(color_path)
        # detections_path = str(detections_path)
        
        # with gzip.open(detections_path, "rb") as f:
        #     gobs = pickle.load(f)

        
        # # depth_image = Image.open(depth_path)
        # # depth_array = np.array(depth_image) / dataset.png_depth_scale
        # # depth_tensor = torch.from_numpy(depth_array).float().to(cfg['device']).T

        # # get pose, this is the untrasformed pose.
        # unt_pose = dataset.poses[idx]
        # unt_pose = unt_pose.cpu().numpy()
        
        # # Don't apply any transformation otherwise
        # adjusted_pose = unt_pose
            
        # # if idx == 71:
        # #     fg_detection_list, bg_detection_list = gobs_to_detection_list(
        # #         cfg = cfg,
        # #         image = image_rgb,
        # #         depth_array = depth_array,
        # #         cam_K = cam_K,
        # #         idx = idx,
        # #         gobs = gobs,
        # #         trans_pose = adjusted_pose,
        # #         class_names = classes,
        # #         BG_CLASSES = BG_CLASSES,
        # #         color_path = color_path,
        # #     )
        # #     for det in fg_detection_list:
        # #         o3d.visualization.draw_geometries([det['pcd']])
                
        # #     exit()
        # # else:
        # #     continue
        
        # fg_detection_list, bg_detection_list = gobs_to_detection_list(
        #     cfg = cfg,
        #     image = image_rgb,
        #     depth_array = depth_array,
        #     cam_K = cam_K,
        #     idx = idx,
        #     gobs = gobs,
        #     trans_pose = adjusted_pose,
        #     class_names = classes,
        #     BG_CLASSES = BG_CLASSES,
        #     color_path = color_path,
        # )