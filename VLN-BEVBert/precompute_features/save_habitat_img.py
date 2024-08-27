import os
import sys
import math
import json
from grpc import Compression
from tqdm import tqdm
import numpy as np
import h5py
from progressbar import ProgressBar
import torch.multiprocessing as mp
import argparse
import cv2

sys.path.insert(0, '/home/lg1/peteryu_workspace/BEV_HGT_VLN/Matterport3DSimulator_opencv4/build')  # please compile Matterport3DSimulator using cpu_only mode
import MatterSim
from utils.habitat_utils import HabitatUtils
from scipy.spatial.transform import Rotation as R

VIEWPOINT_SIZE = 36
WIDTH = 800
HEIGHT = 800
VFOV = 90
HFOV = 60

def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setRenderingEnabled(False)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]      # load all scans
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids

def get_img(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir) #MatterSim
    
    pre_scan = None
    habitat_sim = None
    for scan_id, viewpoint_id in scanvp_list:
        if scan_id != pre_scan:
            if habitat_sim != None:
                habitat_sim.sim.close()
            habitat_sim = HabitatUtils(f'/data/vln_datasets/mp3d/v1/tasks/mp3d/{scan_id}/{scan_id}.glb', 
                                       int(0), HFOV, HEIGHT, WIDTH)
            pre_scan = scan_id

        camera_intrinsics = np.array([
            [1 / np.tan(HFOV / 2.), 0., 0., 0.],
            [0., 1 / np.tan(HFOV / 2.), 0., 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]])
        
        # # create a .txt file save the camera_intrinsics, and only save fx, fy, cx, cy
        # os.makedirs(f"/root/mount/VLN-BEVBert/img_features/{scan_id}", exist_ok=True)
        # with open(f"/root/mount/VLN-BEVBert/img_features/{scan_id}/camera_intrinsics_{scan_id}.txt", 'w') as f:
        #     f.write(f"{camera_intrinsics[0, 0]} {camera_intrinsics[1, 1]} {camera_intrinsics[0, 2]} {camera_intrinsics[1, 2]}")

        transformation_matrix_list = []
        images = []
        images_name_list = []
        for ix in range(VIEWPOINT_SIZE): #Start MatterSim
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            x, y, z, h, e = state.location.x, state.location.y, state.location.z, state.heading, state.elevation
            habitat_position = [x, z-1.25, -y]
            mp3d_h = np.array([0, 2*math.pi-h, 0]) # counter-clock heading
            mp3d_e = np.array([e, 0, 0])
            rotvec_h = R.from_rotvec(mp3d_h)
            rotvec_e = R.from_rotvec(mp3d_e)
            habitat_rotation = (rotvec_h * rotvec_e).as_quat()

            # Convert quaternion to rotation matrix
            rotation_matrix = R.from_quat(habitat_rotation).as_matrix()

            # Create 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = habitat_position

            # transfor the transformation_matrix to a line vector store by row
            transformation_matrix = transformation_matrix.reshape(-1)
            transformation_matrix_list.append(transformation_matrix)

            # print(habitat_position, habitat_rotation, transformation_matrix, sep='\n')

            habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)

            if args.img_type == 'rgb':
                image = habitat_sim.render('rgb')[:, :, ::-1]
                image_name = f"{scan_id}_{viewpoint_id}_{args.img_type}_{ix}.jpg"
            elif args.img_type == 'depth':
                image = habitat_sim.render('depth')
                image_name = f"{scan_id}_{viewpoint_id}_{args.img_type}_{ix}.png"
            

            images_name_list.append(image_name)

            # # make a dir named scan_id
            # os.makedirs(f"/root/mount/VLN-BEVBert/img_features/{scan_id}", exist_ok=True)
            # cv2.imwrite(f"/root/mount/VLN-BEVBert/img_features/{scan_id}/{image_name}", image)

            images.append(image)
        images = np.stack(images, axis=0)
        out_queue.put((scan_id, viewpoint_id, images, images_name_list, transformation_matrix_list, camera_intrinsics))

        

    out_queue.put(None)

def build_img_file(args):

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=get_img,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()


    while num_finished_workers < num_workers:
        res = out_queue.get()
        if res is None:
            num_finished_workers += 1
        else:
            scan_id, viewpoint_id, images, images_name_list, transformation_matrix_list, camera_intrinsics = res
            # create a .txt file save the camera_intrinsics, and only save fx, fy, cx, cy
            os.makedirs(f"img_features/{scan_id}", exist_ok=True)
            with open(f"img_features/{scan_id}/camera_intrinsics_{scan_id}.txt", 'w') as f:
                f.write(f"{camera_intrinsics[0, 0]} {camera_intrinsics[1, 1]} {camera_intrinsics[0, 2]} {camera_intrinsics[1, 2]}")
            for idx in range(len(images_name_list)):
                cv2.imwrite(f"img_features/{scan_id}/{images_name_list[idx]}", images[idx])

            num_finished_vps += 1
            progress_bar.update(num_finished_vps)

# dataset matterport
# n_images 2358
# depth_directory undistorted_depth_images
# color_directory undistorted_color_images

# intrinsics_matrix 1076.45 0 631.116  0 1077.19 509.202  0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d0_0.png 03a8325e3b054e3fad7e1e7091f9d283_i0_0.jpg 0.90525 0.275848 0.323155 -2.99825 0.42464 -0.612795 -0.666455 -14.4532 0.0141878 0.740533 -0.67187 1.33124 0 0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d0_1.png 03a8325e3b054e3fad7e1e7091f9d283_i0_1.jpg 0.820534 -0.381542 -0.425615 -2.98374 -0.571596 -0.547236 -0.6114 -14.4543 0.000362848 0.744955 -0.667115 1.33115 0 0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d0_2.png 03a8325e3b054e3fad7e1e7091f9d283_i0_2.jpg -0.0846661 -0.653405 -0.752259 -2.97748 -0.996408 0.0548167 0.0645317 -14.4674 -0.000928868 0.755021 -0.6557 1.33093 0 0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d0_3.png 03a8325e3b054e3fad7e1e7091f9d283_i0_3.jpg -0.90513 -0.267872 -0.330125 -2.98573 -0.424975 0.591297 0.685393 -14.4794 0.0116044 0.760665 -0.649041 1.3308 0 0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d0_4.png 03a8325e3b054e3fad7e1e7091f9d283_i0_4.jpg -0.820375 0.389515 0.418642 -3.00023 0.571259 0.525712 0.630309 -14.4783 0.0254291 0.756243 -0.653796 1.33089 0 0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d0_5.png 03a8325e3b054e3fad7e1e7091f9d283_i0_5.jpg 0.0848417 0.661354 0.74526 -3.00649 0.996035 -0.076351 -0.0456356 -14.4651 0.0267202 0.746177 -0.665211 1.33112 0 0 0 1

# intrinsics_matrix 1076.01 0 635.509  0 1076.38 511.999  0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d1_0.png 03a8325e3b054e3fad7e1e7091f9d283_i1_0.jpg 0.902625 -0.00853222 0.430341 -2.99332 0.430149 -0.017951 -0.902578 -14.4636 0.0154264 0.999802 -0.0125343 1.36432 0 0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d1_1.png 03a8325e3b054e3fad7e1e7091f9d283_i1_1.jpg 0.82401 -0.00475809 -0.566553 -2.98983 -0.56657 -0.00416342 -0.824001 -14.4639 0.00156227 0.999979 -0.00612788 1.3643 0 0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d1_2.png 03a8325e3b054e3fad7e1e7091f9d283_i1_2.jpg -0.0785585 0.00907021 -0.996867 -2.98832 -0.996908 -0.000537117 0.0785572 -14.467 0.000177485 0.999958 0.009083 1.36425 0 0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d1_3.png 03a8325e3b054e3fad7e1e7091f9d283_i1_3.jpg -0.902493 0.0191241 -0.430278 -2.99031 -0.430517 -0.0106985 0.902519 -14.4699 0.0126569 0.999759 0.0178871 1.36422 0 0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d1_4.png 03a8325e3b054e3fad7e1e7091f9d283_i1_4.jpg -0.823839 0.0153494 0.566614 -2.99379 0.5662 -0.0244859 0.823903 -14.4697 0.0265208 0.999582 0.0114802 1.36424 0 0 0 1
# scan 03a8325e3b054e3fad7e1e7091f9d283_d1_5.png 03a8325e3b054e3fad7e1e7091f9d283_i1_5.jpg 0.0787463 0.00152096 0.996893 -2.9953 0.996503 -0.0281116 -0.078673 -14.4665 0.027905 0.999603 -0.00373071 1.36429 0 0 0 1
# ........


    # with h5py.File(args.output_file, 'w') as outf:
    #     while num_finished_workers < num_workers:
    #         res = out_queue.get()
    #         if res is None:
    #             num_finished_workers += 1
    #         else:
    #             scan_id, viewpoint_id, images = res
    #             key = '%s_%s'%(scan_id, viewpoint_id)
    #             if args.img_type == 'rgb':
    #                 outf.create_dataset(key, data=images, dtype='uint8', compression='gzip')
    #             elif args.img_type == 'depth':
    #                 outf.create_dataset(key, data=images, dtype='float32', compression='gzip')

    #             num_finished_vps += 1
    #             progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', default='precompute_features/connectivity')
    parser.add_argument('--scan_dir', default='/root/mount/Matterport3DSimulator/data/scene_datasets/mp3d') # mp3d scan path
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--img_type', type=str, default='rgb', choices=['rgb', 'depth'])
    args = parser.parse_args()
    if args.img_type == 'rgb':
        args.output_file = f'img_features/habitat_{HEIGHT}x{WIDTH}_vfov{VFOV}_bgr.hdf5'
    elif args.img_type == 'depth':
        args.output_file = f'img_features/habitat_{HEIGHT}x{WIDTH}_vfov{VFOV}_depth.hdf5'
    else:
        raise NotImplementedError

    build_img_file(args)