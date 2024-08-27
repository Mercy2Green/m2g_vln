"""
PyTorch dataset classes for datasets in the NICE-SLAM format.
Large chunks of code stolen and adapted from:
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py

Support for Replica (sequences from the iMAP paper), TUM RGB-D, NICE-SLAM Apartment.
TODO: Add Azure Kinect dataset support
"""

import abc
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from configparser import ConfigParser
import networkx as nx

from gradslam.datasets import datautils
from gradslam.geometry.geometryutils import relative_transformation
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages

from conceptgraph.utils.general_utils import to_scalar
from scipy.spatial.transform import Rotation as R

import configparser
import ast




def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def from_intrinsics_matrix(K: torch.Tensor) -> 'tuple[float, float, float, float]':
    '''
    Get fx, fy, cx, cy from the intrinsics matrix
    
    return 4 scalars
    '''
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y



class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:4",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True, # If True, the pose is relative to the first frame
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = (
            config_dict["camera_params"]["crop_size"]
            if "crop_size" in config_dict["camera_params"]
            else None
        )

        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]

        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError(
                    "Mismatch between number of color images and number of embedding files."
                )
        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()
        
        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        # print(self.poses)
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        return self.num_imgs

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.png_depth_scale
    
    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        
    def get_cam_K(self):
        '''
        Return camera intrinsics matrix K
        
        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        '''
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K
    
    def read_embedding_from_file(self, embedding_path: str):
        '''
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        '''
        raise NotImplementedError

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)
        elif ".npy" in depth_path:
            depth = np.load(depth_path)
        else:
            raise NotImplementedError

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(
            K, self.height_downsample_ratio, self.width_downsample_ratio
        )
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )


class R2RDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence, # the scan_id
        trajectory, # The viewpoint_id list
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 224,
        desired_width: Optional[int] = 224,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        relative_pose: Optional[bool] = True,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder,"camera_parameter", "camera_parameter_"+ sequence + ".conf")
        self.trajectory = trajectory # The viewpoint_id list
        #print("input_folder")

        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            relative_pose=relative_pose,
            **kwargs,
        )

    def get_filepaths(self):

        color_paths = []
        depth_paths = []

        config = configparser.ConfigParser()
        config.read(self.pose_path)
        #print("pose path, the config file path",pose_path)
        for viewpoint_id in self.trajectory:
            # Use ast.literal_eval to convert string representation of list to actual list
            images_name_list = ast.literal_eval(config.get(viewpoint_id, 'images_name_list'))

            # Append each image name to color_images list
            for image_name in images_name_list:
                color_paths.append(os.path.join(self.input_folder,"color_image", image_name))

            # Use ast.literal_eval to convert string representation of list to actual list
            depths_name_list = ast.literal_eval(config.get(viewpoint_id, 'depths_name_list'))

            # Append each depth image name to depth_images list
            for depth_image_name in depths_name_list:
                depth_paths.append(os.path.join(self.input_folder,"depth_image",depth_image_name))

        embedding_paths = []
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []

        config = configparser.ConfigParser()
        config.read(self.pose_path)
        for viewpoint_id in self.trajectory:
            data = config.get(viewpoint_id, 'poses_list')
            poses_list_str = ast.literal_eval(data)
            for pose in poses_list_str:
                _pose = np.reshape(np.array(pose), (4, 4))
                _pose_inv = np.linalg.inv(_pose)
                # print(np.linalg.inv(_pose)[:3,3])
                poses.append(torch.tensor(_pose))

        return poses

class ICLDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: Dict,
        basedir: Union[Path, str],
        sequence: Union[Path, str],
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[Union[Path, str]] = "embeddings",
        embedding_dim: Optional[int] = 512,
        embedding_file_extension: Optional[str] = "pt",
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # Attempt to find pose file (*.gt.sim)
        self.pose_path = glob.glob(os.path.join(self.input_folder, "*.gt.sim"))
        if self.pose_path == 0:
            raise ValueError("Need pose file ending in extension `*.gt.sim`")
        self.pose_path = self.pose_path[0]
        self.embedding_file_extension = embedding_file_extension
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(
                    f"{self.input_folder}/{self.embedding_dir}/*.{self.embedding_file_extension}"
                )
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []

        lines = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()

        _posearr = []
        for line in lines:
            line = line.strip().split()
            if len(line) == 0:
                continue
            _npvec = np.asarray(
                [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
            )
            _posearr.append(_npvec)
        _posearr = np.stack(_posearr)

        for pose_line_idx in range(0, _posearr.shape[0], 3):
            _curpose = np.zeros((4, 4))
            _curpose[3, 3] = 3
            _curpose[0] = _posearr[pose_line_idx]
            _curpose[1] = _posearr[pose_line_idx + 1]
            _curpose[2] = _posearr[pose_line_idx + 2]
            poses.append(torch.from_numpy(_curpose).float())

        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class ReplicaDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class ScannetDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        print(embedding_file_path)
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)



class Ai2thorDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            if self.embedding_dir == "embed_semseg":
                # embed_semseg is stored as uint16 pngs
                embedding_paths = natsorted(
                    glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.png")
                )
            else:
                embedding_paths = natsorted(
                    glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
                )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        if self.embedding_dir == "embed_semseg":
            embedding = imageio.imread(embedding_file_path) # (H, W)
            embedding = cv2.resize(
                embedding, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST
            )
            embedding = torch.from_numpy(embedding).long() # (H, W)
            embedding = F.one_hot(embedding, num_classes = self.embedding_dim) # (H, W, C)
            embedding = embedding.half() # (H, W, C)
            embedding = embedding.permute(2, 0, 1) # (C, H, W)
            embedding = embedding.unsqueeze(0) # (1, C, H, W)
        else:
            embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

class AzureKinectDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None

        # check if a file named 'poses_global_dvo.txt' exists in the basedir / sequence folder 
        if os.path.isfile(os.path.join(basedir, sequence, 'poses_global_dvo.txt')):
            self.pose_path = os.path.join(basedir, sequence, 'poses_global_dvo.txt')
            
        # if "odomfile" in kwargs.keys():
        #     self.pose_path = kwargs["odomfile"]
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        if self.pose_path is None:
            print(
                "WARNING: Dataset does not contain poses. Returning identity transform."
            )
            return [torch.eye(4).float() for _ in range(self.num_imgs)]
        else:
            # Determine whether the posefile ends in ".log"
            # a .log file has the following format for each frame
            # frame_idx frame_idx+1
            # row 1 of 4x4 transform
            # row 2 of 4x4 transform
            # row 3 of 4x4 transform
            # row 4 of 4x4 transform
            # [repeat for all frames]
            #
            # on the other hand, the "poses_o3d.txt" or "poses_dvo.txt" files have the format
            # 16 entries of 4x4 transform
            # [repeat for all frames]
            if self.pose_path.endswith(".log"):
                # print("Loading poses from .log format")
                poses = []
                lines = None
                with open(self.pose_path, "r") as f:
                    lines = f.readlines()
                if len(lines) % 5 != 0:
                    raise ValueError(
                        "Incorrect file format for .log odom file "
                        "Number of non-empty lines must be a multiple of 5"
                    )
                num_lines = len(lines) // 5
                for i in range(0, num_lines):
                    _curpose = []
                    _curpose.append(list(map(float, lines[5 * i + 1].split())))
                    _curpose.append(list(map(float, lines[5 * i + 2].split())))
                    _curpose.append(list(map(float, lines[5 * i + 3].split())))
                    _curpose.append(list(map(float, lines[5 * i + 4].split())))
                    _curpose = np.array(_curpose).reshape(4, 4)
                    poses.append(torch.from_numpy(_curpose))
            else:
                poses = []
                lines = None
                with open(self.pose_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    if len(line.split()) == 0:
                        continue
                    c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                    poses.append(torch.from_numpy(c2w))
            return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding  # .permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class RealsenseDataset(GradSLAMDataset):
    """
    Dataset class to process depth images captured by realsense camera on the tabletop manipulator 
    """
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # only poses/images/depth corresponding to the realsense_camera_order are read/used
        self.pose_path = os.path.join(self.input_folder, "poses")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(
            glob.glob(os.path.join(self.input_folder, "rgb", "*.jpg"))
        )
        depth_paths = natsorted(
            glob.glob(os.path.join(self.input_folder, "depth", "*.png"))
        )
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(os.path.join(self.pose_path, "*.npy")))
        poses = []
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        for posefile in posefiles:
            c2w = torch.from_numpy(np.load(posefile)).float()
            _R = c2w[:3, :3]
            _t = c2w[:3, 3]
            _pose = P @ c2w @ P.T
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class Record3DDataset(GradSLAMDataset):
    """
    Dataset class to read in saved files from the structure created by our
    `save_record3d_stream.py` script
    """
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "poses")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(
            glob.glob(os.path.join(self.input_folder, "rgb", "*.png"))
        )
        depth_paths = natsorted(
            glob.glob(os.path.join(self.input_folder, "depth", "*.png"))
        )
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(os.path.join(self.pose_path, "*.npy")))
        poses = []
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        for posefile in posefiles:
            c2w = torch.from_numpy(np.load(posefile)).float()
            _R = c2w[:3, :3]
            _t = c2w[:3, 3]
            _pose = P @ c2w @ P.T
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class MultiscanDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, f"{sequence}.jsonl")
        
        scene_meta = json.load(
            open(os.path.join(self.input_folder, f"{sequence}.json"), "r")
        )
        cam_K = scene_meta['streams'][0]['intrinsics']
        cam_K = np.array(cam_K).reshape(3, 3).T
        
        config_dict['camera_params']['fx'] = cam_K[0, 0]
        config_dict['camera_params']['fy'] = cam_K[1, 1]
        config_dict['camera_params']['cx'] = cam_K[0, 2]
        config_dict['camera_params']['cy'] = cam_K[1, 2]
        config_dict["camera_params"]["image_height"] = scene_meta['streams'][0]['resolution'][0]
        config_dict["camera_params"]["image_width"] = scene_meta['streams'][0]['resolution'][1]
        
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/outputs/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/outputs/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
            
        return color_paths, depth_paths, embedding_paths
        
    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        n_sampled = len(glob.glob(f"{self.input_folder}/outputs/color/*.png"))
        step = round(len(lines) / float(n_sampled))

        poses = []
        for i in range(0, len(lines), step):
            line = lines[i]
            info = json.loads(line)
            transform = np.asarray(info.get('transform'))
            transform = np.reshape(transform, (4, 4), order='F')
            transform = np.dot(transform, np.diag([1, -1, -1, 1]))
            transform = transform / transform[3][3]
            poses.append(torch.from_numpy(transform).float())
            
        return poses
        
    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class Hm3dDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/*_depth.npy"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths
    
    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/*.json"))
        
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        
        for posefile in posefiles:
            with open(posefile, 'r') as f:
                pose_raw = json.load(f)
            pose = np.asarray(pose_raw['pose'])
            
            pose = torch.from_numpy(pose).float()
            pose = P @ pose @ P.T
            
            poses.append(pose)
            
        return poses

def load_dataset_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def common_dataset_to_batch(dataset):
    colors, depths, poses = [], [], []
    intrinsics, embeddings = None, None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose, _embedding = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
        if _embedding is not None:
            if embeddings is None:
                embeddings = [_embedding]
            else:
                embeddings.append(_embedding)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    if embeddings is not None:
        embeddings = torch.stack(embeddings, dim=1)
        # # (1, NUM_IMG, DIM_EMBED, H, W) -> (1, NUM_IMG, H, W, DIM_EMBED)
        # embeddings = embeddings.permute(0, 1, 3, 4, 2)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()
    if embeddings is not None:
        embeddings = embeddings.float()
    return colors, depths, intrinsics, poses, embeddings


def get_dataset(dataconfig, basedir, sequence, **kwargs):
    config_dict = load_dataset_config(dataconfig)
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["multiscan"]:
        return MultiscanDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict['dataset_name'].lower() in ['hm3d']:
        return Hm3dDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["r2r"]:
        return R2RDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

def load_nav_graphs(connectivity_dir):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    scans = [x.strip() for x in open(os.path.join(connectivity_dir, 'scans.txt')).readlines()]
    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G

    shortest_distances = {}
    shortest_paths = {}
    for scan, G in graphs.items():  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    return graphs, shortest_distances, shortest_paths

def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    viewpoint_ids_poses = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]      # load all scans
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
            viewpoint_ids_poses.extend([(scan, x['image_id'], x['pose']) for x in data if x['included']])
            
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids, viewpoint_ids_poses

def find_paths_by_scan(scan_id, file_path):
    paths = []
    i = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if data['scan'] == scan_id:
                    i = i + 1
                    if i % 3 == 0:
                        paths.append(data['path'])
                    # i = i+1
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Could not decode JSON from file: {file_path}")
        return None

    return paths, i

def generate_subpaths(paths):
    # 创建一个新列表以避免修改原始列表
    # 简单版本，顺序循环
    all_paths = list(paths)
    
    # 遍历原始路径列表
    for path in paths:
        # 生成并添加所有可能的子路径
        for i in range(1, len(path)):
            subpath = path[:i]
            if subpath not in all_paths:  # 避免添加重复的子路径
                all_paths.append(subpath)

    return all_paths

def get_unique_seperate_viewpoints(paths):
    all_paths = list(paths)
    # for path in paths:
    # path = all_paths[0]
    all_seperate_points = []
    for path in all_paths:
        for i in range(len(path)):
            seperate_viewpoint = path[i]
            if seperate_viewpoint not in all_seperate_points:
                all_seperate_points.append(seperate_viewpoint)
    return all_seperate_points

if __name__ == "__main__":
    cfg = load_dataset_config(
        "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/dataset/dataconfigs/R2R/r2r.yaml"
    )
    json_dir =["/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_train_enc.jsonl","/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_prevalent_aug_train_enc.jsonl","/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_seen_enc.jsonl","/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_unseen_enc.jsonl"]
    # json_dir =["/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_train_enc.jsonl"]
    # paths = []
    # goal_scan = "17DRP5sb8fy"
    # for file in json_dir:
    #     _path, _path_size = find_paths_by_scan(goal_scan, file)
    #     paths.extend(_path)
    #     # print(len(_path))
    #     # print(_path_size)
    # traj = get_unique_seperate_viewpoints(paths)
    # # print("length of paths: ", len(paths))
    # traj = []
    # paths_id = 5
    # paths_lens = []
    # for i in range(paths_id):
    #     traj.extend(paths[i])
    #     paths_lens.append(len(paths[i]))


    # # generate all possible subpaths
    # print("#############################################")
    # print("seperate_viewpoints number: ", len(traj))
    # print("--------------------------------------------")
    # for i in range(len(seperate_viewpoints)):
    #     print(seperate_viewpoints[i])
    connectivity_dir = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/precompute_features/connectivity"
    scanvp_list, viewpoint_ids_poses = load_viewpoint_ids("/home/lg1/peteryu_workspace/BEV_HGT_VLN/precompute_features/connectivity")
    graphs, shortest_distances, shortest_paths = load_nav_graphs(connectivity_dir)
    # print shortest_paths size
    import pickle
    traj=[]
    goal_scan = "17DRP5sb8fy"
    path_list_dir = "/home/lg1/lujia/VLN_HGT/17DRP5sb8fy_path_list.pkl"
    with open(path_list_dir, 'rb') as f:
        path_list = pickle.load(f)
    # print(path_list[0][1])
    # #split path_list[0] into 2 parts
    # print("path_list[0] size: ", len(path_list[0]))
    
    # print(shortest_paths[goal_scan]['10c252c90fa24ef3b698c6f54d984c5c']['50c241453dfd45c1ba95b5d7191982ef'])
    # 17DRP5sb8fy
    for scan, vp in scanvp_list:
        if scan == goal_scan:
            # print("scan: ", scan)
            # print("vp size: ", len(vp))
            # print(vp)    
            # traj.append(vp)
            # pose = np.array(pose)
            # pose = pose.reshape(4, 4)
            # translation_part = pose[:3, 3]   
            # _pose_np.append(translation_part)            
            if vp in path_list[0]:
                print("chengongleyici")
                traj.append(vp)

    print("Data saved to dataset_data.pkl")
    print("traj size: ", len(traj))
    print("first element of traj: ", traj[0])
    shortest_path_list = []
    i = 0
    sum_viewpoint_ids = 0
    combined_list = []
    for j in range(1, len(traj)):
        shortest_path_ij = shortest_paths[goal_scan][traj[i]][traj[j]]
        # print("shortest_path_ij size: ", len(shortest_path_ij))
        shortest_path_list.append(shortest_path_ij)
        sum_viewpoint_ids = sum_viewpoint_ids + 2*len(shortest_path_ij)-1
        reversed_shortest_path_ij = shortest_path_ij[::-1][1:]
        new_list = shortest_path_ij + reversed_shortest_path_ij
        combined_list.extend(new_list)
    # shortest_path_0 = shortest_path_list[0]
    # reversed_shortest_path_0 = shortest_path_0[::-1][1:]
    print("shortest_path_list size: ", len(shortest_path_list))
    print("sum_viewpoint_ids: ", sum_viewpoint_ids)
    print("combined_list size: ", len(combined_list))
    dataset = R2RDataset(
        config_dict=cfg,
        basedir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/img_features",
        sequence=goal_scan,
        trajectory=traj,
        relative_pose=False,
        desired_height=224,
        desired_width=224,       
    )

    colors, depths, poses = [], [], []
    corrdinates = []
    intrinsics_np = []
    intrinsics = None
    VIEWPOINT_SIZE = 12
    colors_np = []
    depths_np = []
    poses_np = []
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose = dataset[idx]
        _intrinsics_np = intrinsics.cpu().numpy()
        # _color_np = _color.cpu().numpy()
        # _depth_np = _depth.cpu().numpy()
        _pose_np = _pose.cpu().numpy()
        # colors_np.append(_color_np)
        # depths_np.append(_depth_np)
        # poses_np.append(_pose_np)
        # R_np = _pose_np[:3, :3]
        # t_np = _pose_np[:3, 3]
        # rotation_x = R.from_euler("x", -90, degrees=True)
        # rotation_y = R.from_euler("y", 0, degrees=True)
        # rotation_z = R.from_euler("z", 90, degrees=True)
        # Rotation_ideal = rotation_z.as_matrix() @ rotation_y.as_matrix() @ rotation_x.as_matrix()
        # R_np = np.dot(Rotation_ideal, R_np)
        # t_np = np.dot(Rotation_ideal, t_np)
        # _pose_np[:3, :3] = R_np
        # _pose_np[:3, 3] = t_np
        # _pose = torch.from_numpy(_pose_np).to("cuda:4")
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
        intrinsics_np.append(_intrinsics_np)
        if (idx+1) % VIEWPOINT_SIZE == 0:
            # print("idx: ", idx)
            corrdinates.append(_pose_np[:3, 3])
    # data = {
    #     "colors": colors_np,
    #     "depths": depths_np,
    #     "poses": poses_np,
    #     "intrinsics": intrinsics_np
    # }
    data = {
        "viewpoints": corrdinates
    }
    with open('viewpoint_position_partial.pkl', 'wb') as f:
        pickle.dump(data, f)

    # print("Data saved to dataset_data.pkl")

    # colors = torch.stack(colors)
    # depths = torch.stack(depths)
    # poses = torch.stack(poses)
    # colors = colors.unsqueeze(0)
    # depths = depths.unsqueeze(0)
    # intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    # poses = poses.unsqueeze(0)
    # colors = colors.float()
    # depths = depths.float()
    # intrinsics = intrinsics.float()
    # poses = poses.float()

    # # create rgbdimages object
    # rgbdimages = RGBDImages(
    #     colors,
    #     depths,
    #     intrinsics,
    #     poses,
    #     channels_first=False,
    #     has_embeddings=False,  # KM
    # )

    # slam = PointFusion(odom="gt",dsratio=1, device="cuda:4", use_embeddings=False)
    # pointclouds, recovered_poses = slam(rgbdimages)

    # import open3d as o3d

    # print(pointclouds.colors_padded.shape)
    # pcd = pointclouds.open3d(0)

    # FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])

    # visualize_list = [FOR1, pcd]
    # iter = 0
    # for i in range(len(paths_lens)):
    #     path_length = paths_lens[i]
    #     for j in range(path_length-1):
    #         source_points = corrdinates[iter].cpu().numpy()
    #         target_points = corrdinates[iter+1].cpu().numpy()
    #         points = np.vstack((source_points, target_points))
    #         lines = [[0, 1]]
    #         colors = [[0, 1, 0]]
    #         line_set = o3d.geometry.LineSet(
    #             points=o3d.utility.Vector3dVector(points),
    #             lines=o3d.utility.Vector2iVector(lines),
    #         )
    #         line_set.colors = o3d.utility.Vector3dVector(colors)
    #         visualize_list.append(line_set)
    #         iter = iter + 1
    #     iter = iter + 1
    # for idx in corrdinates:
    #     # tensor idx to numpy.array
    #     position = idx.cpu().numpy()
    #     # print("coordinates position: ", position)

    #     visualize_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=position))

    # o3d.visualization.draw_geometries(visualize_list)    
    # import time
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # vis.add_geometry([])
    # # ctr = vis.get_view_control()
    # for idx in corrdinates:
    #     # tensor idx to numpy.array
    #     position = idx.cpu().numpy()
    #     print("coordinates position: ", position)
    #     vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=position))  
    #     vis.poll_events()
    #     vis.update_renderer()
    #     time.sleep(1.0)
    # vis.destroy_window()

