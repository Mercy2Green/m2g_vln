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

sys.path.insert(0, '/root/mount/Matterport3DSimulator/build')  # please compile Matterport3DSimulator using cpu_only mode
import MatterSim
from utils.habitat_utils import HabitatUtils
from scipy.spatial.transform import Rotation as R

from configparser import ConfigParser

VIEWPOINT_SIZE = 12
WIDTH = 224
HEIGHT = 224
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


def transfrom3D(xyzhe):
    '''
    Return (N, 4, 4) transformation matrices from (N,5) x,y,z,heading,elevation 
    '''
    theta_x = xyzhe[:,4] # elevation
    cx = np.cos(theta_x)
    sx = np.sin(theta_x)

    theta_y = xyzhe[:,3] # heading
    cy = np.cos(theta_y)
    sy = np.sin(theta_y)

    T = np.zeros([xyzhe.shape[0], 4, 4])
    T[:,0,0] =  cy
    T[:,0,1] =  sx*sy
    T[:,0,2] =  cx*sy 
    T[:,0,3] =  xyzhe[:,0] # x

    T[:,1,0] =  0
    T[:,1,1] =  cx
    T[:,1,2] =  -sx
    T[:,1,3] =  xyzhe[:,1] # y

    T[:,2,0] =  -sy
    T[:,2,1] =  cy*sx
    T[:,2,2] =  cy*cx
    T[:,2,3] =  xyzhe[:,2] # z

    T[:,3,3] =  1
    return T.astype(np.float32)


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
            habitat_sim = HabitatUtils(f'/root/mount/Matterport3DSimulator/data/scene_datasets/mp3d/{scan_id}/{scan_id}.glb', 
                                       int(0), HFOV, HEIGHT, WIDTH)
            pre_scan = scan_id

        camera_intrinsics = np.array([
            [((1 / np.tan(math.radians(HFOV) / 2.))*WIDTH) / 2, 0., WIDTH/2, 0.],
            [0., ((1 / np.tan(math.radians(HFOV) / 2.))*HEIGHT) / 2, HEIGHT/2, 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]])
        
        # # create a .txt file save the camera_intrinsics, and only save fx, fy, cx, cy
        # os.makedirs(f"/root/mount/VLN-BEVBert/img_features/{scan_id}", exist_ok=True)
        # with open(f"/root/mount/VLN-BEVBert/img_features/{scan_id}/camera_intrinsics_{scan_id}.txt", 'w') as f:
        #     f.write(f"{camera_intrinsics[0, 0]} {camera_intrinsics[1, 1]} {camera_intrinsics[0, 2]} {camera_intrinsics[1, 2]}")

        transformation_matrix_list = []
        images = []
        depths = []
        images_name_list = []
        depths_name_list = []
        for ix in range(VIEWPOINT_SIZE): #Start MatterSim
            if ix == 0:
                # sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
                sim.newEpisode([scan_id], [viewpoint_id], [0], [0])
            # elif ix % 12 == 0:
            #     sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            # assert state.viewIndex == ix

            x, y, z, h, e = state.location.x, state.location.y, state.location.z, state.heading, state.elevation
            habitat_position = [x, z-1.25, -y]
            mp3d_h = np.array([0, 2*math.pi-h, 0]) # counter-clock heading
            mp3d_e = np.array([e, 0, 0])
            rotvec_h = R.from_rotvec(mp3d_h)
            rotvec_e = R.from_rotvec(mp3d_e)
            habitat_rotation = (rotvec_h * rotvec_e).as_quat()
            habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)

            # scale = 15

            # _x =  - y * 42
            # _y =  (z-1.25) * scale # Ok, don't change
            # _z =  - x * scale # ok for now
            # _h = h # in concept graph is not counter-clock heading

            _x =  x
            _y =  - (z-1.25)
            _z =  y
            _h = h # in concept graph is not counter-clock heading 
            _e = mp3d_e[0] 
            # print("_x", _x, "_y", _y, "_z", _z)

            cx = np.cos(_e)
            sx = np.sin(_e)
            cy = np.cos(_h)
            sy = np.sin(_h)

            T = np.zeros([4, 4])
            T[0,0] = cy
            T[0,1] = sx*sy
            T[0,2] = cx*sy
            T[0,3] = _x
            T[1,0] = 0
            T[1,1] = cx
            T[1,2] = -sx
            T[1,3] = _y
            T[2,0] = -sy
            T[2,1] = cy*sx
            T[2,2] = cy*cx
            T[2,3] = _z
            T[3,3] = 1
            T = T.astype(np.float32)

            # T is camera poses, I need convert it to extrinsic matrix
            # T = np.linalg.inv(T)

            T = T.reshape(-1)
            T = T.tolist()
            transformation_matrix_list.append(T)

            image = habitat_sim.render('rgb')[:, :, ::-1]
            image_name = f"{viewpoint_id}_i_{ix}.jpg"
            
            depth = habitat_sim.render('depth')
            # each depth value is in [0, 1], we need to convert it to [0, 255]
            depth = (depth * 255).astype(np.uint8)
            depth_name = f"{viewpoint_id}_d_{ix}.png"

            images_name_list.append(image_name)
            depths_name_list.append(depth_name)

            images.append(image)
            depths.append(depth)
        images = np.stack(images, axis=0)
        out_queue.put((scan_id, viewpoint_id, images, images_name_list, depths, depths_name_list, transformation_matrix_list, camera_intrinsics))

        

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

    config = ConfigParser()

    while num_finished_workers < num_workers:
        res = out_queue.get()
        if res is None:
            num_finished_workers += 1
        else:
            scan_id, viewpoint_id, images, images_name_list, depths, depths_name_list, transformation_matrix_list, camera_intrinsics = res
            # create a .txt file save the camera_intrinsics, and only save fx, fy, cx, cy
            os.makedirs(f"/root/mount/VLN-BEVBert/img_features/{scan_id}", exist_ok=True)
            os.makedirs(f"/root/mount/VLN-BEVBert/img_features/{scan_id}/camera_parameter", exist_ok=True)
            os.makedirs(f"/root/mount/VLN-BEVBert/img_features/{scan_id}/color_image", exist_ok=True)
            os.makedirs(f"/root/mount/VLN-BEVBert/img_features/{scan_id}/depth_image", exist_ok=True)



            # Add parameters to config
            config[viewpoint_id] = {
                'scan_id': scan_id,
                'viewpoint_id': viewpoint_id,
                'images_name_list': images_name_list,
                'depths_name_list': depths_name_list,
                'poses_list': transformation_matrix_list,
                'camera_intrinsics': camera_intrinsics.tolist()  # convert numpy array to list
            }

            with open(f"/root/mount/VLN-BEVBert/img_features/{scan_id}/camera_parameter/camera_parameter_{scan_id}.conf", 'w') as f:
                # f.write(f"{camera_intrinsics[0, 0]} {camera_intrinsics[1, 1]} {camera_intrinsics[0, 2]} {camera_intrinsics[1, 2]}")
                config.write(f)

            for idx in range(len(images_name_list)):
                cv2.imwrite(f"/root/mount/VLN-BEVBert/img_features/{scan_id}/color_image/{images_name_list[idx]}", images[idx])
                cv2.imwrite(f"/root/mount/VLN-BEVBert/img_features/{scan_id}/depth_image/{depths_name_list[idx]}", depths[idx])

            

                

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

    # /root/mount/Matterport3DSimulator/data/v1/scans
    # python ./precompute_features/save_habitat_img_depth.py --num_workers 4