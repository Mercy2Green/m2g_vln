#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''
import os
import sys
import argparse
import numpy as np
import json
import math
import h5py
import copy
import MatterSim
from PIL import Image
import time
from progressbar import ProgressBar
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
from torch import optim
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop
from easydict import EasyDict as edict
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, build_sam, SamPredictor 
import open_clip
import supervision as sv
import clip
from pathlib import Path
import re
from typing import Any, List
import cv2
import imageio
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pickle
import gzip
from torch.utils.data import Dataset
from tqdm import trange
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption
from ultralytics import YOLO
import torchvision


try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e



# Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    
import sys

if "TAG2TEXT_PATH" in os.environ:
    TAG2TEXT_PATH = os.environ["TAG2TEXT_PATH"]

EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
# sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
sys.path.append(EFFICIENTSAM_PATH)
try:
    # from Tag2Text.models import tag2text
    # from Tag2Text import inference_tag2text, inference_ram

    from ram.models import ram_plus, ram, tag2text
    from ram import inference_tag2text, inference_ram

    import torchvision.transforms as TS
except ImportError as e:
    print("Tag2text sub-package not found. Please check your GSA_PATH. ")
    raise e

# Disable torch gradient computation
torch.set_grad_enabled(False)
    
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

FOREGROUND_GENERIC_CLASSES = [
    "item", "furniture", "object", "electronics", "wall decoration", "door"
]

FOREGROUND_MINIMAL_CLASSES = [
    "item"
]

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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

    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.5)

    parser.add_argument("--class_set", type=str, default="scene", 
                        choices=["scene", "generic", "minimal", "tag2text", "ram", "none"], 
                        help="If none, no tagging and detection will be used and the SAM will be run in dense sampling mode. ")
    parser.add_argument("--add_bg_classes", action="store_true", 
                        help="If set, add background classes (wall, floor, ceiling) to the class set. ")
    parser.add_argument("--accumu_classes", action="store_true",
                        help="if set, the class set will be accumulated over frames")
    parser.add_argument("--detector", type=str, default="dino", 
                        choices=["yolo", "dino"], 
                        help="When given classes, whether to use YOLO-World or GroundingDINO to detect objects. ")
    parser.add_argument("--sam_variant", type=str, default="sam",
                        choices=['fastsam', 'mobilesam', "lighthqsam"])
    
    # parser.add_argument("--save_video", action="store_true")
    
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--use_slow_vis", action="store_true", 
                        help="If set, use vis_result_slow_caption. Only effective when using ram/tag2text. ")
    
    parser.add_argument("--exp_suffix", type=str, default=None,
                        help="The suffix of the folder that the results will be saved to. ")
    
    # parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--checkpoint_file_segment', default=None)
    parser.add_argument('--connectivity_dir', default='../datasets/R2R/connectivity')
    parser.add_argument('--scan_dir', default='../data/v1/scans')
    parser.add_argument('--output_dir')
    # parser.add_argument('--output_file')
    parser.add_argument('--num_workers', type=int, default=1)


    return parser


# MatterSim section

VIEWPOINT_SIZE = 12 # Number of discretized views from one viewpoint

WIDTH = 224
HEIGHT = 224
VFOV = 90

def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids

def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg


def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setRenderingEnabled(True)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


# object feature model section


def compute_clip_features(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    backup_image = image.copy()
    
    image = Image.fromarray(image)
    
    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    image_feats = []
    text_feats = []

    
    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Get the preprocessed image for clip from the crop 
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

        crop_feat = clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        
        class_id = detections.class_id[idx]
        tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        crop_feat = crop_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        text_feats.append(text_feat)
        
    # turn the list of feats into np matrices
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)

    return image_crops, image_feats, text_feats


# Prompting SAM with detected boxes
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    
    if variant == "mobilesam":
        from MobileSAM.setup_mobile_sam import setup_model
        MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        
        sam_predictor = SamPredictor(mobile_sam)
        return sam_predictor

    elif variant == "lighthqsam":
        from LightHQSAM.setup_light_hqsam import setup_model
        HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
        checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=device)
        
        sam_predictor = SamPredictor(light_hqsam)
        return sam_predictor
        
    elif variant == "fastsam":
        raise NotImplementedError
    else:
        raise NotImplementedError
    


# The SAM based on automatic mask generation, without bbox prompting
def get_sam_segmentation_dense(
    variant:str, model: Any, image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    The SAM based on automatic mask generation, without bbox prompting
    
    Args:
        model: The mask generator or the YOLO model
        image: )H, W, 3), in RGB color space, in range [0, 255]
        
    Returns:
        mask: (N, H, W)
        xyxy: (N, 4)
        conf: (N,)
    '''

    if variant == "sam":
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        results = model.generate(image)
        mask = []
        xyxy = []
        conf = []
        for r in results:
            mask.append(r["segmentation"])
            r_xyxy = r["bbox"].copy()
            # Convert from xyhw format to xyxy format
            r_xyxy[2] += r_xyxy[0]
            r_xyxy[3] += r_xyxy[1]
            xyxy.append(r_xyxy)
            conf.append(r["predicted_iou"])
        mask = np.array(mask)
        xyxy = np.array(xyxy)
        conf = np.array(conf)
        return mask, xyxy, conf
    elif variant == "fastsam":
        # The arguments are directly copied from the GSA repo
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        results = model(
            image,
            imgsz=1024,
            device="cuda",
            retina_masks=True,
            iou=0.9,
            conf=0.4,
            max_det=100,
        )
        raise NotImplementedError
    else:
        raise NotImplementedError


def get_sam_mask_generator(variant:str, device: str | int) -> SamAutomaticMaskGenerator:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=12,
            points_per_batch=144,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=100,
        )
        return mask_generator
    elif variant == "fastsam":
        raise NotImplementedError
        # from ultralytics import YOLO
        # from FastSAM.tools import *
        # FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/FastSAM-x.pt")
        # model = YOLO(args.model_path)
        # return model
    else:
        raise NotImplementedError


def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> list[str]:
    '''
    Convert a text prompt from Tag2Text to a list of classes. 
    '''
    classes = text_prompt.split(',')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes


def process_ai2thor_classes(classes: List[str], add_classes:List[str]=[], remove_classes:List[str]=[]) -> List[str]:
    '''
    Some pre-processing on AI2Thor objectTypes in a scene
    '''
    classes = list(set(classes))
    
    for c in add_classes:
        classes.append(c)
        
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]

    # Split the element in classes by captical letters
    classes = [obj_class.replace("TV", "Tv") for obj_class in classes]
    classes = [re.findall('[A-Z][^A-Z]*', obj_class) for obj_class in classes]
    # Join the elements in classes by space
    classes = [" ".join(obj_class) for obj_class in classes]
    
    return classes


def compute_clip_features_sam(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    backup_image = image.copy()
    
    image = Image.fromarray(image)
    
    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    image_feats = []
    text_feats = []

    
    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Get the preprocessed image for clip from the crop 
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

        crop_feat = clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        
        class_id = detections.class_id[idx]
        tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        crop_feat = crop_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        text_feats.append(text_feat)
        
    # turn the list of feats into np matrices
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)

    return image_crops, image_feats, text_feats

def process_features(proc_id, out_queue, scanvp_list, args: argparse.Namespace):

    print('start proc_id: %d' % proc_id)
    gpu_count = torch.cuda.device_count()
    local_rank = proc_id % gpu_count + 2 # 2 to 7 (6 six gpu)
    torch.cuda.set_device('cuda:{}'.format(local_rank))
 
    results_list = []

    ### Initialize the Grounding DINO model ###
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
        device=args.device
    )

    ### Initialize the SAM model ###
    if args.class_set == "none":
        mask_generator = get_sam_mask_generator(args.sam_variant, args.device)
    else:
        sam_predictor = get_sam_predictor(args.sam_variant, args.device)
    
    ###
    # Initialize the CLIP model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    global_classes = set()

    # Initialize a YOLO-World model
    yolo_model_w_classes = YOLO('yolov8l-world.pt')  # or choose yolov8m/l-world.pt
    
    if args.class_set == "scene":
        # # Load the object meta information
        # obj_meta_path = args.dataset_root / args.scene_id / "obj_meta.json"
        # with open(obj_meta_path, "r") as f:
        #     obj_meta = json.load(f)
        # # Get a list of object classes in the scene
        # classes = process_ai2thor_classes(
        #     [obj["objectType"] for obj in obj_meta],
        #     add_classes=[],
        #     remove_classes=['wall', 'floor', 'room', 'ceiling']
        # )
        pass
    elif args.class_set == "generic":
        classes = FOREGROUND_GENERIC_CLASSES
    elif args.class_set == "minimal":
        classes = FOREGROUND_MINIMAL_CLASSES
    elif args.class_set in ["tag2text", "ram"]:
        ### Initialize the Tag2Text or RAM model ###
        
        if args.class_set == "tag2text":
            # The class set will be computed by tag2text on each image
            # filter out attributes and action categories which are difficult to grounding
            delete_tag_index = []
            for i in range(3012, 3429):
                delete_tag_index.append(i)

            specified_tags='None'
            # load model
            tagging_model = tag2text.tag2text_caption(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                                                    image_size=384,
                                                    vit='swin_b',
                                                    delete_tag_index=delete_tag_index)
            # threshold for tagging
            # we reduce the threshold to obtain more tags
            tagging_model.threshold = 0.64 
        elif args.class_set == "ram":
            tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                                         image_size=384,
                                         vit='swin_l')
            
        tagging_model = tagging_model.eval().to(args.device)
        
        # initialize Tag2Text
        tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), 
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        
        classes = None
    elif args.class_set == "none":
        classes = ['item']
    else:
        raise ValueError("Unknown args.class_set: ", args.class_set)

    if args.class_set not in ["ram", "tag2text"]:
        print("There are total", len(classes), "classes to detect. ")
    elif args.class_set == "none":
        print("Skipping tagging and detection models. ")
    else:
        print(f"{args.class_set} will be used to detect classes. ")
        
    save_name = f"{args.class_set}"
    if args.sam_variant != "sam": # For backward compatibility
        save_name += f"_{args.sam_variant}"
    if args.exp_suffix:
        save_name += f"_{args.exp_suffix}"
    
    # if args.save_video:
    #     video_save_path = args.output_dir / f"gsa_vis_{save_name}.mp4"
    #     frames = []


    count_number = 1

    for scan_id, viewpoint_id in scanvp_list:

        # print the progress for moniter in percentage
        print('scan_id: %s, viewpoint_id: %s' % (scan_id, viewpoint_id))
        print('Proc_id: %d, percentage: %f' % (proc_id, (count_number/len(scanvp_list) * 100)))
        count_number += 1

        # Loop all discretized views from this location
        images = []


        # save_name
        save_name = '%s_%s'%(scan_id, viewpoint_id)

        detections_save_path_save_name = args.output_dir + f"/{scan_id}" + f"/gsa_detections_{save_name}"

        detections_save_path_gz = detections_save_path_save_name + f".pkl.gz"
        os.makedirs(os.path.dirname(detections_save_path_gz), exist_ok=True)

        for ix in range(VIEWPOINT_SIZE):

            ### Relevant paths and load image ###
            # color_path = args.dataset_root / args.scene_id / "color" / f"{scan_id}_{viewpoint_id}_{ix}.jpg"

            # color_path = Path(color_path)
            
            vis_save_path = args.output_dir
            detections_save_path = args.output_dir
            
            os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(detections_save_path), exist_ok=True)

            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [0])
            # elif ix % 12 == 0:
            #     sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix + 12

            image = np.array(state.rgb, copy=True) # in BGR channel
            image = BGR_to_RGB(image)
            image_rgb = image #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)

            # image = np.array(state.rgb, copy=True)
            # image = Image.fromarray(image)
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image_pil = Image.fromarray(image_rgb)

            ### Tag2Text ###
            if args.class_set in ["ram", "tag2text"]:
                raw_image = image_pil.resize((384, 384))
                raw_image = tagging_transform(raw_image).unsqueeze(0).to(args.device)
                
                if args.class_set == "ram":
                    res = inference_ram(raw_image , tagging_model)
                    caption="NA"
                elif args.class_set == "tag2text":
                    res = inference_tag2text.inference(raw_image , tagging_model, specified_tags)
                    caption=res[2]

                # Currently ", " is better for detecting single tags
                # while ". " is a little worse in some case
                text_prompt=res[0].replace(' |', ',')
                
                # Add "other item" to capture objects not in the tag2text captions. 
                # Remove "xxx room", otherwise it will simply include the entire image
                # Also hide "wall" and "floor" for now...
                add_classes = ["other item"]
                remove_classes = [
                    "room", "kitchen", "office", "house", "home", "building", "corner",
                    "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
                    "apartment", "image", "city", "blue", "skylight", "hallway", 
                    "bureau", "modern", "salon", "doorway", "wall lamp"
                ]
                bg_classes = ["wall", "floor", "ceiling"]

                if args.add_bg_classes:
                    add_classes += bg_classes
                else:
                    remove_classes += bg_classes

                classes = process_tag_classes(
                    text_prompt, 
                    add_classes = add_classes,
                    remove_classes = remove_classes,
                )
                
            # add classes to global classes
            global_classes.update(classes)
            
            if args.accumu_classes:
                # Use all the classes that have been seen so far
                classes = list(global_classes)
                
            ### Detection and segmentation ###
            if args.class_set == "none":
                # Directly use SAM in dense sampling mode to get segmentation
                mask, xyxy, conf = get_sam_segmentation_dense(
                    args.sam_variant, mask_generator, image_rgb)
                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=np.zeros_like(conf).astype(int),
                    mask=mask,
                )
                image_crops, image_feats, text_feats = compute_clip_features(
                    image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)

                ### Visualize results ###
                annotated_image, labels = vis_result_fast(
                    image, detections, classes, instance_random_color=True)
                
                # cv2.imwrite(vis_save_path, annotated_image)
            else:
                if args.detector == "dino":
                    # Using GroundingDINO to detect and SAM to segment
                    detections = grounding_dino_model.predict_with_classes(
                        image=image, # This function expects a BGR image...
                        classes=classes,
                        box_threshold=args.box_threshold,
                        text_threshold=args.text_threshold,
                    )
                
                    if len(detections.class_id) > 0:
                        ### Non-maximum suppression ###
                        # print(f"Before NMS: {len(detections.xyxy)} boxes")
                        nms_idx = torchvision.ops.nms(
                            torch.from_numpy(detections.xyxy), 
                            torch.from_numpy(detections.confidence), 
                            args.nms_threshold
                        ).numpy().tolist()
                        # print(f"After NMS: {len(detections.xyxy)} boxes")

                        detections.xyxy = detections.xyxy[nms_idx]
                        detections.confidence = detections.confidence[nms_idx]
                        detections.class_id = detections.class_id[nms_idx]
                        
                        # Somehow some detections will have class_id=-1, remove them
                        valid_idx = detections.class_id != -1
                        detections.xyxy = detections.xyxy[valid_idx]
                        detections.confidence = detections.confidence[valid_idx]
                        detections.class_id = detections.class_id[valid_idx]

                        # # Somehow some detections will have class_id=-None, remove them
                        # valid_idx = [i for i, val in enumerate(detections.class_id) if val is not None]
                        # detections.xyxy = detections.xyxy[valid_idx]
                        # detections.confidence = detections.confidence[valid_idx]
                        # detections.class_id = [detections.class_id[i] for i in valid_idx]
                # elif args.detector == "yolo":
                #     # YOLO 
                #     # yolo_model.set_classes(classes)
                #     yolo_model_w_classes.set_classes(classes)
                #     yolo_results_w_classes = yolo_model_w_classes.predict(color_path)

                #     yolo_results_w_classes[0].save(vis_save_path[:-4] + "_yolo_out.jpg")
                #     xyxy_tensor = yolo_results_w_classes[0].boxes.xyxy 
                #     xyxy_np = xyxy_tensor.cpu().numpy()
                #     confidences = yolo_results_w_classes[0].boxes.conf.cpu().numpy()
                    
                #     detections = sv.Detections(
                #         xyxy=xyxy_np,
                #         confidence=confidences,
                #         class_id=yolo_results_w_classes[0].boxes.cls.cpu().numpy().astype(int),
                #         mask=None,
                #     )
                    
                if len(detections.class_id) > 0:
                    
                    ### Segment Anything ###
                    detections.mask = get_sam_segmentation_from_xyxy(
                        sam_predictor=sam_predictor,
                        image=image_rgb,
                        xyxy=detections.xyxy
                    )

                    # Compute and save the clip features of detections  
                    image_crops, image_feats, text_feats = compute_clip_features(
                        image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)
                else:
                    image_crops, image_feats, text_feats = [], [], []
                
                ### Visualize results ###
                annotated_image, labels = vis_result_fast(image, detections, classes)
                
                # save the annotated grounded-sam image
                if args.class_set in ["ram", "tag2text"] and args.use_slow_vis:
                    annotated_image_caption = vis_result_slow_caption(
                        image_rgb, detections.mask, detections.xyxy, labels, caption, text_prompt)
                    Image.fromarray(annotated_image_caption).save(vis_save_path)
                # else:
                #     # cv2.imwrite(vis_save_path, annotated_image)
            
            # if args.save_video:
            #     frames.append(annotated_image)
            
            # Convert the detections to a dict. The elements are in np.array
            results = {
                "xyxy": detections.xyxy, # data type is np.array
                "confidence": detections.confidence, # data type is np.array
                "class_id": detections.class_id, # data type is np.array
                "mask": detections.mask, # data type is np.array
                "classes": classes, # data type is list
                "image_crops": image_crops, # data type is list
                "image_feats": image_feats, # data type is np.array
                "text_feats": text_feats, # data type is np.array
            }
            
            if args.class_set in ["ram", "tag2text"]:
                results["tagging_caption"] = caption
                results["tagging_text_prompt"] = text_prompt
            
            # results_list.append(results)

            # save the detections using pickle
            # Here we use gzip to compress the file, which could reduce the file size by 500x
            with gzip.open(detections_save_path_gz, "wb") as f:
                pickle.dump(results, f)
        
        # save global classes
        with open(args.output_dir + f"/{scan_id}" + f"/gsa_classes_{save_name}.json", "w") as f:
            json.dump(list(global_classes), f)
        



def build_feature_file(args): # main funcution


    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir) #return para viewpoint_ids is a list

    # workers load data 
    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers # how many data per worker

    out_queue = mp.Queue()

    res = []

    processes = []
    for proc_id in range(num_workers): # proc_id is the index of workers
        sidx = proc_id * num_data_per_worker # start index
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker # end index

        process = mp.Process( # process into the process_features function
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args) 
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    
    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # scanvp_list = load_viewpoint_ids(args.connectivity_dir) #return para viewpoint_ids is a list

    # proc_id = 0
    # output_queue = []
    

    # process_features(proc_id, output_queue, scanvp_list, args)


    

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    parser = get_parser()
    args = parser.parse_args()
    build_feature_file(args)
    # build_feature_file_batch(args)
