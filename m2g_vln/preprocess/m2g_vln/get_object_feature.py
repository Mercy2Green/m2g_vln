#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys

#import Matterport3DSimulator.MatterSim as MatterSim
import MatterSim
import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

# from utils import load_viewpoint_ids
from tqdm import tqdm
from torch import optim

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop

from easydict import EasyDict as edict
# from model_clip import CLIP

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import build_sam, SamPredictor 

import open_clip
import supervision as sv

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

clip_config = edict({
    'patches_grid': None,
    'patches_size': 16,
    'hidden_size': 768,
    'transformer_mlp_dim': 3072,
    'transformer_num_heads': 12,
    'transformer_num_layers': 12,
    'transformer_attention_dropout_rate': 0.,
    'transformer_dropout_rate': 0.
})

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768

WIDTH = 224
HEIGHT = 224
VFOV = 60

# def build_feature_extractor(checkpoint_file=None):

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     torch.set_grad_enabled(False)
#     model = CLIP(input_resolution=224, patch_size=clip_config.patches_size, width=clip_config.hidden_size, layers=clip_config.transformer_num_layers, heads=clip_config.transformer_num_heads).to(device)

#     state_dict = torch.load(checkpoint_file, map_location='cpu')
#     model.load_state_dict(state_dict, strict=False)

   
#     model.eval()

#     img_transforms =  Compose([
#             Resize((224,224), interpolation=Image.BICUBIC),
#             ToTensor(),
#             Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#         ])

#     return model, img_transforms, device

def build_clip_feature_extractor(): # using segement anything model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the CLIP model
    # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    #     "ViT-H-14", "laion2b_s32b_b79k"
    # )

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32")
    clip_model = clip_model.to(device)
    #clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    img_transforms =  Compose([
            Resize((224,224), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    return clip_model, clip_preprocess, img_transforms, device

def compute_clip_features(image, detections, clip_model, clip_preprocess):

    # output all masks features by append to a list

    # backup_image = image.copy()

    # print(image)
    
    # image = Image.fromarray(image)
    
    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed
    
    # image_crops = []
    image_feats = []
    # text_feats = []

    
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
        
        # class_id = detections.class_id[idx]
        # tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
        # text_feat = clip_model.encode_text(tokenized_text)
        # text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        crop_feat = crop_feat.cpu().numpy().astype(np.float16)
        # text_feat = text_feat.cpu().numpy()

        # image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        # text_feats.append(text_feat)
        
    # turn the list of feats into np matrices
    image_feats = np.concatenate(image_feats, axis=0)
    # text_feats = np.concatenate(text_feats, axis=0)
    return image_feats

def build_object_mask_extractor(checkpoint_file_segment=None): # using segement anything model

    # build a Segment anything model and a CLIP model
    device_segment = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    model_segment = sam_model_registry['default'](checkpoint=checkpoint_file_segment).to(device_segment)
    #output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    #amg_kwargs = get_amg_kwargs(args)
    mask_generator = SamAutomaticMaskGenerator(
            model_segment, 
            points_per_side=12,
            points_per_batch=144,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=100,
            )

    return mask_generator, device_segment

def mask_extraction(model, image: np.ndarray):
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

def detection_generate(mask, xyxy, conf):

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=conf,
        class_id=np.zeros_like(conf).astype(int),
        mask=mask,
    )
    return detections

def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def process_features_batch(proc_id, out_queue, scanvp_list, args):

    # out_queue = []

    print('start proc_id: %d' % proc_id)
    gpu_count = torch.cuda.device_count()
    local_rank = proc_id % gpu_count
    torch.cuda.set_device('cuda:{}'.format(local_rank))
    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    #model, img_transforms, device = build_feature_extractor(args.checkpoint_file) # into  build_feature_extractor function

    clip_model, clip_preprocess, img_transforms, device = build_clip_feature_extractor() # into  build_feature_extractor function

    mask_generator, device_segment = build_object_mask_extractor(args.checkpoint_file_segment) # into  build_feature_extractor function

    count_number = 1

    for scan_id, viewpoint_id in scanvp_list:

        # print the progress for moniter in percentage
        print('scan_id: %s, viewpoint_id: %s' % (scan_id, viewpoint_id))
        print('percentage: %d/%d' % (count_number, len(scanvp_list)))
        count_number += 1

        # Loop all discretized views from this location
        images = []
        images_rgb = []

        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            if 12 <= ix and ix < 24:
                pass
            else:
                continue

            image = np.array(state.rgb, copy=True) # in BGR channel
            image = BGR_to_RGB(image)
            image = Image.fromarray(image) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            images_rgb.append(image)

            # image.save("./{scan_id}_{viewpoint_id}_{ix}.png".format(scan_id=scan_id, viewpoint_id=viewpoint_id, ix=ix))

        images = torch.stack([img_transforms(image).to(device) for image in images], 0)

        fts = []

        for k in range(0, len(images), 1):

            # print the progress for moniter in percentage
            print('progress: %d/%d' % (k, len(images)))

            # print(images[k].shape)
            # print(images_rgb[k])

            mask, xyxy, conf = mask_extraction(mask_generator, images[k])

            # print(xyxy)

            if(xyxy.size > 0):
                detections = detection_generate(mask, xyxy, conf)
                image_feats= compute_clip_features(images_rgb[k], detections, clip_model, clip_preprocess)
                # This is the place i change !!!!!!!!!!!!!
                b_fts = image_feats

                #print(image_feats)

                # b_fts = model(images[k: k+args.batch_size]) # this is my question
                b_fts = b_fts.astype(np.float16)

                fts.append(b_fts)
            else:
                # print(xyxy)
                print('no object detected')

        fts = np.concatenate(fts, 0)

        # out_queue.append((scan_id, viewpoint_id, fts))
        out_queue.put((scan_id, viewpoint_id, fts))

    out_queue.put(None)
    # return out_queue


def process_features(proc_id, out_queue, scanvp_list, args):

    print('start proc_id: %d' % proc_id)
    gpu_count = torch.cuda.device_count()
    local_rank = proc_id % gpu_count
    torch.cuda.set_device('cuda:{}'.format(local_rank))
    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    #model, img_transforms, device = build_feature_extractor(args.checkpoint_file) # into  build_feature_extractor function

    clip_model, clip_preprocess, img_transforms, device = build_clip_feature_extractor() # into  build_feature_extractor function

    mask_generator, device_segment = build_object_mask_extractor(args.checkpoint_file_segment) # into  build_feature_extractor function

    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            if 12 <= ix and ix < 24:
                pass
            else:
                continue

            image = np.array(state.rgb, copy=True) # in BGR channel
            image = BGR_to_RGB(image)
            image = Image.fromarray(image) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        images = torch.stack([img_transforms(image).to(device) for image in images], 0)

        fts = []
        for k in range(0, len(images), 1):

            # mask, xyxy, conf = get_sam_segmentation_dense(
            #     args.sam_variant, mask_generator, image_rgb)
            # detections = sv.Detections(
            #     xyxy=xyxy,
            #     confidence=conf,
            #     class_id=np.zeros_like(conf).astype(int),
            #     mask=mask,
            # )
            print(images[k])

            detections = detection_generate(images[k], mask_generator)

            image_feats= compute_clip_features(images[k], detections, clip_model, clip_preprocess)
            
            # This is the place i change !!!!!!!!!!!!!
            b_fts = image_feats

            # b_fts = model(images[k: k+args.batch_size]) # this is my question
            b_fts = b_fts.data.cpu().numpy().astype(np.float16)
            fts.append(b_fts)

        fts = np.concatenate(fts, 0)

        out_queue.put((scan_id, viewpoint_id, fts))

    out_queue.put(None)

def build_feature_file_batch(args): # main funcution
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir) #return para viewpoint_ids is a list

    # num_batches 
    num_batches = min(args.num_batches, len(scanvp_list))
    num_data_per_batch = len(scanvp_list) // num_batches # how many data per batch

    out_queue = mp.Queue()

    res = []

    for batch_id in range(num_batches):
        sidx = batch_id * num_data_per_batch # start index
        eidx = None if batch_id == num_batches - 1 else sidx + num_data_per_batch # end index

        print(sidx, eidx)

        process_features_batch(batch_id, out_queue, scanvp_list[sidx: eidx], args)


    num_finished_batches = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(maxval=len(scanvp_list))
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_batches < num_batches:
            res = out_queue.get()
            if res is None:
                num_finished_batches += 1
            else:
                scan_id, viewpoint_id, fts = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                
                data = fts
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()


def build_feature_file(args): # main funcution
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir) #return para viewpoint_ids is a list

    # workers load data 
    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers # how many data per worker

    out_queue = mp.Queue()

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
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(maxval=len(scanvp_list))
    progress_bar.start()


    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                
                data = fts
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()

    for process in processes:
        process.join()


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Runs automatic mask generation on an Matterport3D. "
        )
    )
    # parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--checkpoint_file_segment', default=None)
    parser.add_argument('--connectivity_dir', default='../datasets/R2R/connectivity')
    parser.add_argument('--scan_dir', default='../data/v1/scans')
    parser.add_argument('--output_file')
    # parser.add_argument('--batch_size', default=4, type=int)
    # parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_batches', type=int, default=2)
    args = parser.parse_args()

    # build_feature_file(args)
    build_feature_file_batch(args)
