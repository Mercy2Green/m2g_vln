"""
Build a scene graph from the segment-based map and captions from LLaVA.
"""

import gc
import gzip
import json
import os
import pickle as pkl
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Union
from textwrap import wrap

import cv2
import matplotlib.pyplot as plt

import numpy as np
import rich
import torch
import tyro
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from tqdm import tqdm, trange
from transformers import logging as hf_logging

# from mappingutils import (
#     MapObjectList,
#     compute_3d_giou_accuracte_batch,
#     compute_3d_iou_accuracte_batch,
#     compute_iou_batch,
#     compute_overlap_matrix_faiss,
#     num_points_closer_than_threshold_batch,
# )

torch.autograd.set_grad_enabled(False)
hf_logging.set_verbosity_error()

# Import OpenAI API
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ProgramArgs:
    mode: Literal[
        "extract-node-captions",
        "refine-node-captions",
        "build-scenegraph",
        "generate-scenegraph-json",
        "annotate-scenegraph",
    ]

    # Path to cache directory
    cachedir: str = "saved/room0"
    
    prompts_path: str = "prompts/gpt_prompts.json"

    # Path to map file
    mapfile: str = "saved/room0/map/scene_map_cfslam.pkl.gz"

    # Path to file storing segment class names
    class_names_file: str = "saved/room0/gsa_classes_ram.json"

    # Device to use
    device: str = "cuda:0"

    # Voxel size for downsampling
    downsample_voxel_size: float = 0.025

    # Maximum number of detections to consider, per object
    max_detections_per_object: int = 10

    # Suppress objects with less than this number of observations
    min_views_per_object: int = 2

    # List of objects to annotate (default: all objects)
    annot_inds: Union[List[int], None] = None

    # Masking option
    masking_option: Literal["blackout", "red_outline", "none"] = "none"

def load_scene_map(args, scene_map):
    """
    Loads a scene map from a gzip-compressed pickle file. This is a function because depending whether the mapfile was made using cfslam_pipeline_batch.py or merge_duplicate_objects.py, the file format is different (see below). So this function handles that case.
    
    The function checks the structure of the deserialized object to determine
    the correct way to load it into the `scene_map` object. There are two
    expected formats:
    1. A dictionary containing an "objects" key.
    2. A list or a dictionary (replace with your expected type).
    """
    
    with gzip.open(Path(args.mapfile), "rb") as f:
        loaded_data = pkl.load(f)
        
        # Check the type of the loaded data to decide how to proceed
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            scene_map.load_serializable(loaded_data["objects"])
        elif isinstance(loaded_data, list) or isinstance(loaded_data, dict):  # Replace with your expected type
            scene_map.load_serializable(loaded_data)
        else:
            raise ValueError("Unexpected data format in map file.")
        print(f"Loaded {len(scene_map)} objects")


def prjson(input_json, indent=0):
    """ Pretty print a json object """
    if not isinstance(input_json, list):
        input_json = [input_json]
        
    print("[")
    for i, entry in enumerate(input_json):
        print("  {")
        for j, (key, value) in enumerate(entry.items()):
            terminator = "," if j < len(entry) - 1 else ""
            if isinstance(value, str):
                formatted_value = value.replace("\\n", "\n").replace("\\t", "\t")
                print('    "{}": "{}"{}'.format(key, formatted_value, terminator))
            else:
                print(f'    "{key}": {value}{terminator}')
        print("  }" + ("," if i < len(input_json) - 1 else ""))
    print("]")

def crop_image_pil(image: Image, x1: int, y1: int, x2: int, y2: int, padding: int = 0) -> Image:
    """
    Crop the image with some padding

    Args:
        image: PIL image
        x1, y1, x2, y2: bounding box coordinates
        padding: padding around the bounding box

    Returns:
        image_crop: PIL image

    Implementation from the CFSLAM repo
    """
    image_width, image_height = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    image_crop = image.crop((x1, y1, x2, y2))
    return image_crop


def draw_red_outline(image, mask):
    """ Draw a red outline around the object i nan image"""
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    red_outline = [255, 0, 0]

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red outlines around the object. The last argument "3" indicates the thickness of the outline.
    cv2.drawContours(image_np, contours, -1, red_outline, 3)

    # Optionally, add padding around the object by dilating the drawn contours
    kernel = np.ones((5, 5), np.uint8)
    image_np = cv2.dilate(image_np, kernel, iterations=1)
    
    image_pil = Image.fromarray(image_np)

    return image_pil


def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
    
    image = np.array(image)
    # Verify initial dimensions
    if image.shape[:2] != mask.shape:
        print("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
        return None, None

    # Define the cropping coordinates
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    # round the coordinates to integers
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # Crop the image and the mask
    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # Verify cropped dimensions
    if image_crop.shape[:2] != mask_crop.shape:
        print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
        return None, None
    
    # convert the image back to a pil image
    image_crop = Image.fromarray(image_crop)

    return image_crop, mask_crop

def blackout_nonmasked_area(image_pil, mask):
    """ Blackout the non-masked area of an image"""
    # convert image to numpy array
    image_np = np.array(image_pil)
    # Create an all-black image of the same shape as the input image
    black_image = np.zeros_like(image_np)
    # Wherever the mask is True, replace the black image pixel with the original image pixel
    black_image[mask] = image_np[mask]
    # convert back to pil image
    black_image = Image.fromarray(black_image)
    return black_image

def plot_images_with_captions(images, captions, confidences, low_confidences, masks, savedir, idx_obj):
    """ This is debug helper function that plots the images with the captions and masks overlaid and saves them to a directory. This way you can inspect exactly what the LLaVA model is captioning which image with the mask, and the mask confidence scores overlaid."""
    
    n = min(9, len(images))  # Only plot up to 9 images
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])

        # Apply the mask to the image
        img_array = np.array(images[i])
        if img_array.shape[:2] != masks[i].shape:
            ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        else:
            green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            green_mask[masks[i]] = [0, 255, 0]  # Green color where mask is True
            ax.imshow(green_mask, alpha=0.15)  # Overlay with transparency

        title_text = f"Caption: {captions[i]}\nConfidence: {confidences[i]:.2f}"
        if low_confidences[i]:
            title_text += "\nLow Confidence"
        
        # Wrap the caption text
        wrapped_title = '\n'.join(wrap(title_text, 30))
        
        ax.set_title(wrapped_title, fontsize=12)  # Reduced font size for better fitting
        ax.axis('off')

    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(savedir / f"{idx_obj}.png")
    plt.close()



def extract_node_captions(args):
    from conceptgraph.llava.llava_model import LLaVaChat

    # NOTE: args.mapfile is in cfslam format
    from conceptgraph.slam.slam_classes import MapObjectList

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)
    
    # Scene map is in CFSLAM format
    # keys: 'image_idx', 'mask_idx', 'color_path', 'class_id', 'num_detections',
    # 'mask', 'xyxy', 'conf', 'n_points', 'pixel_area', 'contain_number', 'clip_ft',
    # 'text_ft', 'pcd_np', 'bbox_np', 'pcd_color_np'

    # Imports to help with feature extraction
    # from extract_mask_level_features import (
    #     crop_bbox_from_img,
    #     get_model_and_preprocessor,
    #     preprocess_and_encode_pil_image,
    # )

    # Load class names from the json file
    class_names = None
    with open(Path(args.class_names_file), "r") as f:
        class_names = json.load(f)
    print(class_names)

    # Creating a namespace object to pass args to the LLaVA chat object
    chat_args = SimpleNamespace()
    chat_args.model_path = os.getenv("LLAVA_MODEL_PATH")
    chat_args.conv_mode = "v0_mmtag" # "multimodal"
    chat_args.num_gpus = 1

    # rich console for pretty printing
    console = rich.console.Console()

    # Initialize LLaVA chat
    chat = LLaVaChat(chat_args.model_path, chat_args.conv_mode, chat_args.num_gpus)
    # chat = LLaVaChat(chat_args)
    print("LLaVA chat initialized...")
    query = "Describe the central object in the image."
    # query = "Describe the object in the image that is outlined in red."

    # Directories to save features and captions
    savedir_feat = Path(args.cachedir) / "cfslam_feat_llava"
    savedir_feat.mkdir(exist_ok=True, parents=True)
    savedir_captions = Path(args.cachedir) / "cfslam_captions_llava"
    savedir_captions.mkdir(exist_ok=True, parents=True)
    savedir_debug = Path(args.cachedir) / "cfslam_captions_llava_debug"
    savedir_debug.mkdir(exist_ok=True, parents=True)

    caption_dict_list = []

    for idx_obj, obj in tqdm(enumerate(scene_map), total=len(scene_map)):
        conf = obj["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]

        features = []
        captions = []
        low_confidences = []
        
        image_list = []
        caption_list = []
        confidences_list = []
        low_confidences_list = []
        mask_list = []  # New list for masks
        if len(idx_most_conf) < 2:
            continue 
        idx_most_conf = idx_most_conf[:args.max_detections_per_object]

        for idx_det in tqdm(idx_most_conf):
            # image = Image.open(correct_path).convert("RGB")
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            xyxy = obj["xyxy"][idx_det]
            class_id = obj["class_id"][idx_det]
            class_name = class_names[class_id]
            # Retrieve and crop mask
            mask = obj["mask"][idx_det]

            padding = 10
            x1, y1, x2, y2 = xyxy
            # image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=padding)
            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            if args.masking_option == "blackout":
                image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop)
            elif args.masking_option == "red_outline":
                image_crop_modified = draw_red_outline(image_crop, mask_crop)
            else:
                image_crop_modified = image_crop  # No modification

            _w, _h = image_crop.size
            if _w * _h < 70 * 70:
                # captions.append("small object")
                print("small object. Skipping LLaVA captioning...")
                low_confidences.append(True)
                continue
            else:
                low_confidences.append(False)

            # image_tensor = chat.image_processor.preprocess(image_crop, return_tensors="pt")["pixel_values"][0]
            image_tensor = chat.image_processor.preprocess(image_crop_modified, return_tensors="pt")["pixel_values"][0]

            image_features = chat.encode_image(image_tensor[None, ...].half().cuda())
            features.append(image_features.detach().cpu())

            chat.reset()
            console.print("[bold red]User:[/bold red] " + query)
            outputs = chat(query=query, image_features=image_features)
            console.print("[bold green]LLaVA:[/bold green] " + outputs)
            captions.append(outputs)
        
            # print(f"Line 274, obj['mask'][idx_det].shape: {obj['mask'][idx_det].shape}")
            # print(f"Line 276, image.size: {image.size}")
            
            # For the LLava debug folder
            conf_value = conf[idx_det]
            image_list.append(image_crop)
            caption_list.append(outputs)
            confidences_list.append(conf_value)
            low_confidences_list.append(low_confidences[-1])
            mask_list.append(mask_crop)  # Add the cropped mask

        caption_dict_list.append(
            {
                "id": idx_obj,
                "captions": captions,
                "low_confidences": low_confidences,
            }
        )

        # Concatenate the features
        if len(features) > 0:
            features = torch.cat(features, dim=0)

        # Save the feature descriptors
        torch.save(features, savedir_feat / f"{idx_obj}.pt")
        
        # Again for the LLava debug folder
        if len(image_list) > 0:
            plot_images_with_captions(image_list, caption_list, confidences_list, low_confidences_list, mask_list, savedir_debug, idx_obj)

    # Save the captions
    # Remove the "The central object in the image is " prefix from 
    # the captions as it doesnt convey and actual info
    for item in caption_dict_list:
        item["captions"] = [caption.replace("The central object in the image is ", "") for caption in item["captions"]]
    # Save the captions to a json file
    with open(Path(args.cachedir) / "cfslam_llava_captions.json", "w", encoding="utf-8") as f:
        json.dump(caption_dict_list, f, indent=4, sort_keys=False)


def save_json_to_file(json_str, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_str, f, indent=4, sort_keys=False)


def refine_node_captions(args):
    # NOTE: args.mapfile is in cfslam format
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.scenegraph.GPTPrompt import GPTPrompt

    # Load the captions for each segment
    caption_file = Path(args.cachedir) / "cfslam_llava_captions.json"
    captions = None
    with open(caption_file, "r") as f:
        captions = json.load(f)
    # loaddir_captions = Path(args.cachedir) / "cfslam_captions_llava"
    # captions = []
    # for idx_obj in range(len(os.listdir(loaddir_captions))):
    #     with open(loaddir_captions / f"{idx_obj}.pkl", "rb") as f:
    #         captions.append(pkl.load(f))

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)
    
    # load the prompt
    gpt_messages = GPTPrompt().get_json()

    TIMEOUT = 25  # Timeout in seconds

    responses_savedir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    responses_savedir.mkdir(exist_ok=True, parents=True)

    responses = []
    unsucessful_responses = 0

    # loop over every object
    for _i in trange(len(captions)):
        if len(captions[_i]) == 0:
            continue
        
        # Prepare the object prompt 
        _dict = {}
        _caption = captions[_i]
        _bbox = scene_map[_i]["bbox"]
        # _bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(scene_map[_i]["bbox"]))
        _dict["id"] = _caption["id"]
        # _dict["bbox_extent"] = np.round(_bbox.extent, 1).tolist()
        # _dict["bbox_center"] = np.round(_bbox.center, 1).tolist()
        _dict["captions"] = _caption["captions"]
        # _dict["low_confidences"] = _caption["low_confidences"]
        # Convert to printable string
        
        # Make and format the full prompt
        preds = json.dumps(_dict, indent=0)

        start_time = time.time()
    
        curr_chat_messages = gpt_messages[:]
        curr_chat_messages.append({"role": "user", "content": preds})
        chat_completion = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=curr_chat_messages,
            timeout=TIMEOUT,  # Timeout in seconds
        )
        elapsed_time = time.time() - start_time
        if elapsed_time > TIMEOUT:
            print("Timed out exceeded!")
            _dict["response"] = "FAIL"
            # responses.append('{"object_tag": "FAIL"}')
            save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
            responses.append(json.dumps(_dict))
            unsucessful_responses += 1
            continue
        
        # count unsucessful responses
        if "invalid" in chat_completion["choices"][0]["message"]["content"].strip("\n"):
            unsucessful_responses += 1
            
        # print output
        prjson([{"role": "user", "content": preds}])
        print(chat_completion["choices"][0]["message"]["content"])
        print(f"Unsucessful responses so far: {unsucessful_responses}")
        _dict["response"] = chat_completion["choices"][0]["message"]["content"].strip("\n")
        
        # save the response
        responses.append(json.dumps(_dict))
        save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
        # responses.append(chat_completion["choices"][0]["message"]["content"].strip("\n"))

    # tags = []
    # for response in responses:
    #     try:
    #         parsed = json.loads(response)
    #         tags.append(parsed["object_tag"])
    #     except:
    #         tags.append("FAIL")

    # Save the responses to a text file
    # with open(Path(args.cachedir) / "gpt-3.5-turbo_responses.txt", "w") as f:
    #     for response in responses:
    #         f.write(response + "\n")
    with open(Path(args.cachedir) / "cfslam_gpt-4_responses.pkl", "wb") as f:
        pkl.dump(responses, f)


def extract_object_tag_from_json_str(json_str):
    start_str_found = False
    is_object_tag = False
    object_tag_complete = False
    object_tag = ""
    r = json_str.strip().split()
    for _idx, _r in enumerate(r):
        if not start_str_found:
            # Searching for open parenthesis of JSON
            if _r == "{":
                start_str_found = True
                continue
            else:
                continue
        # Start string found. Now skip everything until the object_tag field
        if not is_object_tag:
            if _r == '"object_tag":':
                is_object_tag = True
                continue
            else:
                continue
        # object_tag field found. Read it
        if is_object_tag and not object_tag_complete:
            if _r == '"':
                continue
            else:
                if _r.strip() in [",", "}"]:
                    break
                object_tag += f" {_r}"
                continue
    return object_tag


def build_scenegraph(args):
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.slam.utils import compute_overlap_matrix

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    response_dir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    responses = []
    object_tags = []
    also_indices_to_remove = [] # indices to remove if the json file does not exist
    for idx in range(len(scene_map)):
        # check if the json file exists first 
        if not (response_dir / f"{idx}.json").exists():
            also_indices_to_remove.append(idx)
            continue
        with open(response_dir / f"{idx}.json", "r") as f:
            _d = json.load(f)
            try:
                _d["response"] = json.loads(_d["response"])
            except json.JSONDecodeError:
                _d["response"] = {
                    'summary': f'GPT4 json reply failed: Here is the invalid response {_d["response"]}',
                    'possible_tags': ['possible_tag_json_failed'],
                    'object_tag': 'invalid'
                }
            responses.append(_d)
            object_tags.append(_d["response"]["object_tag"])

    # # Load the responses from the json file
    # responses = None
    # # Load json file into a list
    # with open(Path(args.cachedir) / "cfslam_gpt-4_responses.pkl", "rb") as f:
    #     responses = pkl.load(f)
    # object_tags = []
    # for response in responses:
    #     # _tag = extract_object_tag_from_json_str(response)
    #     # _tag = _tag.lower().replace('"', "").strip()
    #     _d = json.loads(json.loads(response)["responses"])
    #     object_tags.append(_d["object_tag"])

    # Remove segments that correspond to "invalid" tags
    indices_to_remove = [i for i in range(len(responses)) if object_tags[i].lower() in ["fail", "invalid"]]
    # Also remove segments that do not have a minimum number of observations
    indices_to_remove = set(indices_to_remove)
    for obj_idx in range(len(scene_map)):
        conf = scene_map[obj_idx]["conf"]
        # Remove objects with less than args.min_views_per_object observations
        if len(conf) < args.min_views_per_object:
            indices_to_remove.add(obj_idx)
    indices_to_remove = list(indices_to_remove)
    # combine with also_indices_to_remove and sort the list
    indices_to_remove = list(set(indices_to_remove + also_indices_to_remove))
    
    # List of tags in original scene map that are in the pruned scene map
    segment_ids_to_retain = [i for i in range(len(scene_map)) if i not in indices_to_remove]
    with open(Path(args.cachedir) / "cfslam_scenegraph_invalid_indices.pkl", "wb") as f:
        pkl.dump(indices_to_remove, f)
    print(f"Removed {len(indices_to_remove)} segments")
    
    # Filtering responses based on segment_ids_to_retain
    responses = [resp for resp in responses if resp['id'] in segment_ids_to_retain]

    # Assuming each response dictionary contains an 'object_tag' key for the object tag.
    # Extract filtered object tags based on filtered_responses
    object_tags = [resp['response']['object_tag'] for resp in responses]


    pruned_scene_map = []
    pruned_object_tags = []
    for _idx, segmentidx in enumerate(segment_ids_to_retain):
        pruned_scene_map.append(scene_map[segmentidx])
        pruned_object_tags.append(object_tags[_idx])
    scene_map = MapObjectList(pruned_scene_map)
    object_tags = pruned_object_tags
    del pruned_scene_map
    # del pruned_object_tags
    gc.collect()
    num_segments = len(scene_map)

    for i in range(num_segments):
        scene_map[i]["caption_dict"] = responses[i]
        # scene_map[i]["object_tag"] = object_tags[i]

    # Save the pruned scene map (create the directory if needed)
    if not (Path(args.cachedir) / "map").exists():
        (Path(args.cachedir) / "map").mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        pkl.dump(scene_map.to_serializable(), f)

    print("Computing bounding box overlaps...")
    bbox_overlaps = compute_overlap_matrix(args, scene_map)

    # Construct a weighted adjacency matrix based on similarity scores
    weights = []
    rows = []
    cols = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if i == j:
                continue
            if bbox_overlaps[i, j] > 0.01:
                weights.append(bbox_overlaps[i, j])
                rows.append(i)
                cols.append(j)
                weights.append(bbox_overlaps[i, j])
                rows.append(j)
                cols.append(i)

    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_segments, num_segments))

    # Find the minimum spanning tree of the weighted adjacency matrix
    mst = minimum_spanning_tree(adjacency_matrix)

    # Find connected components in the minimum spanning tree
    _, labels = connected_components(mst)

    components = []
    _total = 0
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            indices = np.where(labels == label)[0]
            _total += len(indices.tolist())
            components.append(indices.tolist())

    with open(Path(args.cachedir) / "cfslam_scenegraph_components.pkl", "wb") as f:
        pkl.dump(components, f)

    # Initialize a list to store the minimum spanning trees of connected components
    minimum_spanning_trees = []
    relations = []
    if len(labels) != 0:
        # Iterate over each connected component
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            # Extract the subgraph for the connected component
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            # Find the minimum spanning tree of the connected component subgraph
            _mst = minimum_spanning_tree(subgraph)
            # Add the minimum spanning tree to the list
            minimum_spanning_trees.append(_mst)

        TIMEOUT = 25  # timeout in seconds

        if not (Path(args.cachedir) / "cfslam_object_relations.json").exists():
            relation_queries = []
            for componentidx, component in enumerate(components):
                if len(component) <= 1:
                    continue
                for u, v in zip(
                    minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
                ):
                    segmentidx1 = component[u]
                    segmentidx2 = component[v]
                    _bbox1 = scene_map[segmentidx1]["bbox"]
                    _bbox2 = scene_map[segmentidx2]["bbox"]

                    input_dict = {
                        "object1": {
                            "id": segmentidx1,
                            "bbox_extent": np.round(_bbox1.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox1.center, 1).tolist(),
                            "object_tag": object_tags[segmentidx1],
                        },
                        "object2": {
                            "id": segmentidx2,
                            "bbox_extent": np.round(_bbox2.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox2.center, 1).tolist(),
                            "object_tag": object_tags[segmentidx2],
                        },
                    }
                    print(f"{input_dict['object1']['object_tag']}, {input_dict['object2']['object_tag']}")

                    relation_queries.append(input_dict)

                    input_json_str = json.dumps(input_dict)

                    # Default prompt
                    DEFAULT_PROMPT = """
                    The input is a list of JSONs describing two objects "object1" and "object2". You need to produce a JSON
                    string (and nothing else), with two keys: "object_relation", and "reason".

                    Each of the JSON fields "object1" and "object2" will have the following fields:
                    1. bbox_extent: the 3D bounding box extents of the object
                    2. bbox_center: the 3D bounding box center of the object
                    3. object_tag: an extremely brief description of the object

                    Produce an "object_relation" field that best describes the relationship between the two objects. The
                    "object_relation" field must be one of the following (verbatim):
                    1. "a on b": if object a is an object commonly placed on top of object b
                    2. "b on a": if object b is an object commonly placed on top of object a
                    3. "a in b": if object a is an object commonly placed inside object b
                    4. "b in a": if object b is an object commonly placed inside object a
                    5. "none of these": if none of the above best describe the relationship between the two objects

                    Before producing the "object_relation" field, produce a "reason" field that explains why
                    the chosen "object_relation" field is the best.
                    """

                    start_time = time.time()
                    chat_completion = openai.ChatCompletion.create(
                        # model="gpt-3.5-turbo",
                        model="gpt-4",
                        messages=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
                        timeout=TIMEOUT,  # Timeout in seconds
                    )
                    elapsed_time = time.time() - start_time
                    output_dict = input_dict
                    if elapsed_time > TIMEOUT:
                        print("Timed out exceeded!")
                        output_dict["object_relation"] = "FAIL"
                        continue
                    else:
                        try:
                            # Attempt to parse the output as a JSON
                            chat_output_json = json.loads(chat_completion["choices"][0]["message"]["content"])
                            # If the output is a valid JSON, then add it to the output dictionary
                            output_dict["object_relation"] = chat_output_json["object_relation"]
                            output_dict["reason"] = chat_output_json["reason"]
                        except:
                            output_dict["object_relation"] = "FAIL"
                            output_dict["reason"] = "FAIL"
                    relations.append(output_dict)

                    # print(chat_completion["choices"][0]["message"]["content"])

            # Save the query JSON to file
            print("Saving query JSON to file...")
            with open(Path(args.cachedir) / "cfslam_object_relation_queries.json", "w") as f:
                json.dump(relation_queries, f, indent=4)

            # Saving the output
            print("Saving object relations to file...")
            with open(Path(args.cachedir) / "cfslam_object_relations.json", "w") as f:
                json.dump(relations, f, indent=4)
        else:
            relations = json.load(open(Path(args.cachedir) / "cfslam_object_relations.json", "r"))

    scenegraph_edges = []

    _idx = 0
    for componentidx, component in enumerate(components):
        if len(component) <= 1:
            continue
        for u, v in zip(
            minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
        ):
            segmentidx1 = component[u]
            segmentidx2 = component[v]
            # print(f"{segmentidx1}, {segmentidx2}, {relations[_idx]['object_relation']}")
            if relations[_idx]["object_relation"] != "none of these":
                scenegraph_edges.append((segmentidx1, segmentidx2, relations[_idx]["object_relation"]))
            _idx += 1
    print(f"Created 3D scenegraph with {num_segments} nodes and {len(scenegraph_edges)} edges")

    with open(Path(args.cachedir) / "cfslam_scenegraph_edges.pkl", "wb") as f:
        pkl.dump(scenegraph_edges, f)


def generate_scenegraph_json(args):
    from conceptgraph.slam.slam_classes import MapObjectList
    

    # Generate the JSON file summarizing the scene, if it doesn't exist already
    # or if the --recopmute_scenegraph_json flag is set
    scene_desc = []
    print("Generating scene graph JSON file...")

    # Load the pruned scene map
    scene_map = MapObjectList()
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        scene_map.load_serializable(pkl.load(f))
    print(f"Loaded scene map with {len(scene_map)} objects")

    for i, segment in enumerate(scene_map):
        _d = {
            "id": segment["caption_dict"]["id"],
            "bbox_extent": np.round(segment['bbox'].extent, 1).tolist(),
            "bbox_center": np.round(segment['bbox'].center, 1).tolist(),
            "possible_tags": segment["caption_dict"]["response"]["possible_tags"],
            "object_tag": segment["caption_dict"]["response"]["object_tag"],
            "caption": segment["caption_dict"]["response"]["summary"],
        }
        scene_desc.append(_d)
    with open(Path(args.cachedir) / "scene_graph.json", "w") as f:
        json.dump(scene_desc, f, indent=4)


def display_images(image_list):
    num_images = len(image_list)
    cols = 2  # Number of columns for the subplots (you can change this as needed)
    rows = (num_images + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = image_list[i]
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def annotate_scenegraph(args):
    from conceptgraph.slam.slam_classes import MapObjectList

    # Load the pruned scene map
    scene_map = MapObjectList()
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        scene_map.load_serializable(pkl.load(f))

    annot_inds = None
    if args.annot_inds is not None:
        annot_inds = args.annot_inds
    # If annot_inds is not None, we also need to load the annotation json file and only
    # annotate the objects that are specified in the annot_inds list
    annots = []
    if annot_inds is not None:
        annots = json.load(open(Path(args.cachedir) / "annotated_scenegraph.json", "r"))

    if annot_inds is None:
        annot_inds = list(range(len(scene_map)))

    for idx in annot_inds:
        print(f"Object {idx} out of {len(annot_inds)}...")
        obj = scene_map[idx]

        prev_annot = None
        if len(annots) >= idx + 1:
            prev_annot = annots[idx]

        annot = {}
        annot["id"] = idx

        conf = obj["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]
        print(obj.keys())

        imgs = []

        for idx_det in idx_most_conf:
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            xyxy = obj["xyxy"][idx_det]
            mask = obj["mask"][idx_det]

            padding = 10
            x1, y1, x2, y2 = xyxy
            image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=padding)
            mask_crop = crop_image_pil(Image.fromarray(mask), x1, y1, x2, y2, padding=padding)
            mask_crop = np.array(mask_crop)[..., None]
            mask_crop[mask_crop == 0] = 0.05
            image_crop = np.array(image_crop) * mask_crop
            imgs.append(image_crop)
            if len(imgs) >= 5:
                break
            # if idx_det >= 5:
            #     break

        # Display the images
        display_images(imgs)
        plt.close("all")

        # Ask the user to annotate the object
        if prev_annot is not None:
            print("Previous annotation:")
            print(prev_annot)
        annot["object_tags"] = input("Enter object tags (comma-separated): ")
        annot["colors"] = input("Enter colors (comma-separated): ")
        annot["materials"] = input("Enter materials (comma-separated): ")

        if prev_annot is not None:
            annots[idx] = annot
        else:
            annots.append(annot)

        go_on = input("Continue? (y/n): ")
        if go_on == "n":
            break

    # Save the annotations
    with open(Path(args.cachedir) / "annotated_scenegraph.json", "w") as f:
        json.dump(annots, f, indent=4)


def generate_object_edges_by_rules(args):
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.slam.utils import compute_overlap_matrix

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    # Also remove segments that do not have a minimum number of observations
    indices_to_remove = set([])
    for obj_idx in range(len(scene_map)):
        conf = scene_map[obj_idx]["conf"]
        # Remove objects with less than args.min_views_per_object observations
        if len(conf) < args.min_views_per_object:
            indices_to_remove.add(obj_idx)
    indices_to_remove = list(indices_to_remove)
    # combine with also_indices_to_remove and sort the list
    # indices_to_remove = list(set(indices_to_remove + also_indices_to_remove))
    indices_to_remove = list(set(indices_to_remove))
    
    # List of tags in original scene map that are in the pruned scene map
    segment_ids_to_retain = [i for i in range(len(scene_map)) if i not in indices_to_remove]
    with open(Path(args.cachedir) / "cfslam_scenegraph_invalid_indices.pkl", "wb") as f:
        pkl.dump(indices_to_remove, f)
    print(f"Removed {len(indices_to_remove)} segments")
    
    pruned_scene_map = []
    # pruned_object_tags = []
    for _idx, segmentidx in enumerate(segment_ids_to_retain):
        pruned_scene_map.append(scene_map[segmentidx])
        # pruned_object_tags.append(object_tags[_idx])
    scene_map = MapObjectList(pruned_scene_map)
    # object_tags = pruned_object_tags
    del pruned_scene_map
    # del pruned_object_tags
    gc.collect()
    num_segments = len(scene_map)

    # for i in range(num_segments):
    #     scene_map[i]["caption_dict"] = responses[i]
    #     # scene_map[i]["object_tag"] = object_tags[i]

    # Save the pruned scene map (create the directory if needed)
    if not (Path(args.cachedir) / "map").exists():
        (Path(args.cachedir) / "map").mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        pkl.dump(scene_map.to_serializable(), f)

    print("Computing bounding box overlaps...")
    bbox_overlaps = compute_overlap_matrix(args, scene_map)

    # Construct a weighted adjacency matrix based on similarity scores
    weights = []
    rows = []
    cols = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if i == j:
                continue
            if bbox_overlaps[i, j] > 0.01:
                weights.append(bbox_overlaps[i, j])
                rows.append(i)
                cols.append(j)
                weights.append(bbox_overlaps[i, j])
                rows.append(j)
                cols.append(i)

    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_segments, num_segments))

    # Find the minimum spanning tree of the weighted adjacency matrix
    mst = minimum_spanning_tree(adjacency_matrix)

    # Find connected components in the minimum spanning tree
    _, labels = connected_components(mst)

    components = []
    _total = 0
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            indices = np.where(labels == label)[0]
            _total += len(indices.tolist())
            components.append(indices.tolist())

    with open(Path(args.cachedir) / "cfslam_scenegraph_components.pkl", "wb") as f:
        pkl.dump(components, f)

    # Initialize a list to store the minimum spanning trees of connected components
    minimum_spanning_trees = []
    relations = []
    if len(labels) != 0:
        # Iterate over each connected component
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            # Extract the subgraph for the connected component
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            # Find the minimum spanning tree of the connected component subgraph
            _mst = minimum_spanning_tree(subgraph)
            # Add the minimum spanning tree to the list
            minimum_spanning_trees.append(_mst)

        if not (Path(args.cachedir) / "cfslam_object_relations.json").exists():
            relation_queries = []
            for componentidx, component in enumerate(components):
                if len(component) <= 1:
                    continue
                for u, v in zip(
                    minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
                ):
                    segmentidx1 = component[u]
                    segmentidx2 = component[v]
                    _bbox1 = scene_map[segmentidx1]["bbox"]
                    _bbox2 = scene_map[segmentidx2]["bbox"]

                    input_dict = {
                        "object1": {
                            "id": segmentidx1,
                            "bbox_extent": np.round(_bbox1.extent, 4).tolist(),
                            "bbox_center": np.round(_bbox1.center, 4).tolist(),
                            "object_volume": np.round(_bbox1.volume(), 4),
                            # "object_tag": object_tags[segmentidx1],
                        },
                        "object2": {
                            "id": segmentidx2,
                            "bbox_extent": np.round(_bbox2.extent, 4).tolist(),
                            "bbox_center": np.round(_bbox2.center, 4).tolist(),
                            "object_volume": np.round(_bbox2.volume(), 4),
                            # "object_tag": object_tags[segmentidx2],
                        },
                    }
                    # print(f"{input_dict['object1']['object_volume']}, {input_dict['object2']['object_volume']}")

                    relation_queries.append(input_dict)

                    # input_json_str = json.dumps(input_dict)
                    
                    output_dict = input_dict
                    # Example usage within the provided code snippet
                    # Assuming _bbox1 and _bbox2 are the bounding boxes from the code excerpt
                    if is_inside(_bbox1, _bbox2) and is_bigger(_bbox2, _bbox1):
                        # print(f"a in b")
                        output_dict["object_relation"] = "a in b"
                    elif is_inside(_bbox2, _bbox1) and is_bigger(_bbox1, _bbox2):
                        # print(f"b in a")     
                        output_dict["object_relation"] = "b in a"          
                    elif is_attached(_bbox1, _bbox2) and is_bigger(_bbox2, _bbox1):
                        # print("a on b")
                        output_dict["object_relation"] = "a on b"
                    elif is_attached(_bbox2, _bbox1) and is_bigger(_bbox1, _bbox2):
                        # print("b on a")
                        output_dict["object_relation"] = "b on a"
                    else:
                        # print("none of these")
                        output_dict["object_relation"] = "none of these"

                    relations.append(output_dict)

            # Save the query JSON to file
            print("Saving query JSON to file...")
            with open(Path(args.cachedir) / "cfslam_object_relation_queries.json", "w") as f:
                json.dump(relation_queries, f, indent=4)

            # Saving the output
            print("Saving object relations to file...")
            with open(Path(args.cachedir) / "cfslam_object_relations.json", "w") as f:
                json.dump(relations, f, indent=4)
        else:
            relations = json.load(open(Path(args.cachedir) / "cfslam_object_relations.json", "r"))

    scenegraph_edges = []

    _idx = 0
    for componentidx, component in enumerate(components):
        if len(component) <= 1:
            continue
        for u, v in zip(
            minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
        ):
            segmentidx1 = component[u]
            segmentidx2 = component[v]
            
            if _idx % 10 == 0: # for debug
                save_edges_image_with_relation(args, scene_map, segmentidx1, segmentidx2, relations[_idx]["object_relation"])
            # print(f"{segmentidx1}, {segmentidx2}, {relations[_idx]['object_relation']}")
            if relations[_idx]["object_relation"] != "none of these":
                scenegraph_edges.append((segmentidx1, segmentidx2, relations[_idx]["object_relation"]))
            _idx += 1
    print(f"Created 3D scenegraph with {num_segments} nodes and {len(scenegraph_edges)} edges")

    with open(Path(args.cachedir) / "cfslam_scenegraph_edges.pkl", "wb") as f:
        pkl.dump(scenegraph_edges, f)
        
    results = {'objects': scene_map.to_serializable()}
    scene_map_save_path = Path(args.cachedir) / "cfslam_scenegraph_nodes.pkl.gz"
    with gzip.open(scene_map_save_path, "wb") as f:
        pkl.dump(results, f)
        
        
        
def get_image_crop(scene_map, obj_idx, idx_det=0):
   
    obj = scene_map[obj_idx]
    image = Image.open(obj["color_path"][idx_det]).convert("RGB")
    xyxy = obj["xyxy"][idx_det]
    # Retrieve and crop mask
    mask = obj["mask"][idx_det]
    conf = obj["conf"][idx_det]

    padding = 10
    x1, y1, x2, y2 = xyxy
    # image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=padding)
    image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
    
    _w, _h = image_crop.size
    if _w * _h < 70 * 70:
        # captions.append("small object")
        # print("small object. Skipping LLaVA captioning...")
        low_confidences = True
    else:
        low_confidences = False
        
        
    return image_crop, mask_crop, conf, low_confidences

def save_edges_image_with_relation(args, scene_map, obj1_idx, obj2_idx, relation):
    
    obj1_image_crop, obj1_mask_crop, obj1_confidence, obj1_low_confidences = get_image_crop(scene_map, obj1_idx)
    obj2_image_crop, obj2_maks_crop, obj2_confidence, obj2_low_confidences = get_image_crop(scene_map, obj2_idx)
    
    savedir_debug = Path(args.cachedir) / "cfslam_object_relations_debug"
    savedir_debug.mkdir(exist_ok=True, parents=True)
    
    image_list = [obj1_image_crop, obj2_image_crop]
    mask_list = [obj1_mask_crop, obj2_maks_crop]
    caption_list = [relation, relation]
    # confidences_list=[obj1_confidence, obj2_confidence]
    confidences_list = [scene_map[obj1_idx]["bbox"].volume(), scene_map[obj2_idx]["bbox"].volume()]
    
    low_confidences_list=[False, False]
    idx_obj = f"{obj1_idx}_{obj2_idx}"
    plot_images_with_captions(image_list, caption_list, confidences_list, low_confidences_list, mask_list, savedir_debug, idx_obj)
            

def is_inside(bbox1, bbox2):
    """
    Check if bbox1 is inside bbox2.
    """
    # Calculate corners of bbox1
    corners1 = np.array([
        bbox1.center + np.array([dx * bbox1.extent[0] / 2, dy * bbox1.extent[1] / 2, dz * bbox1.extent[2] / 2])
        for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)
    ])
    
    # Transform corners1 to bbox2's coordinate system
    relative_corners = corners1 - bbox2.center
    
    # Check if all corners are within bbox2's extent
    inside = np.all(np.abs(relative_corners) <= bbox2.extent / 2, axis=1)
    
    return np.all(inside)

def is_on_top(bboxA, bboxB):
    # Step 1: Check vertical alignment
    vertical_distance = (bboxA.center[1] - bboxA.extent[1] / 2) - (bboxB.center[1] + bboxB.extent[1] / 2)
    if vertical_distance > 0 and vertical_distance < 0.5:  # Assuming a small threshold for "on"
        # Step 2: Check horizontal overlap
        for axis in [0, 2]:  # Assuming 0:x-axis, 1:y-axis(vertical), 2:z-axis for a y-up system
            distance = abs(bboxA.center[axis] - bboxB.center[axis])
            max_extent = (bboxA.extent[axis] + bboxB.extent[axis]) / 2
            if distance >= max_extent:
                return False  # No horizontal overlap in this axis
        return True  # Passed both checks
    return False

def is_attached(bbox1, bbox2, threshold=0.1):
    """
    Check if bbox1 is attached to bbox2 within a certain threshold.
    The threshold determines how close the front faces of the bounding boxes need to be.
    """
    # Calculate the distance between the each faces of the two bounding boxes
    distance_x = np.abs(bbox1.center[0] - bbox2.center[0]) - (bbox1.extent[0]/2 + bbox2.extent[0]/2)
    distance_y = np.abs(bbox1.center[1] - bbox2.center[1]) - (bbox1.extent[1]/2 + bbox2.extent[1]/2)
    distance_z = np.abs(bbox1.center[2] - bbox2.center[2]) - (bbox1.extent[2]/2 + bbox2.extent[2]/2)

    if distance_x <= threshold or distance_y <= threshold or distance_z <= threshold:
        return True
    else:
        return False
    
def is_attached_old(bbox1, bbox2, threshold=0.1):
    """
    Check if bbox1 is attached to bbox2 within a certain threshold.
    The threshold determines how close the front faces of the bounding boxes need to be.
    """
    # Calculate the distance between the each faces of the two bounding boxes
    distance_x = np.abs(bbox1.center[0] - bbox2.center[0]) - (bbox1.extent[0]/2 + bbox2.extent[0]/2)
    distance_y = np.abs(bbox1.center[1] - bbox2.center[1]) - (bbox1.extent[1]/2 + bbox2.extent[1]/2)
    distance_z = np.abs(bbox1.center[2] - bbox2.center[2]) - (bbox1.extent[2]/2 + bbox2.extent[2]/2)
    

    # Check if the bounding boxes are aligned vertically and horizontally
    aligned_x = np.abs(bbox1.center[0] - bbox2.center[0]) <= (bbox1.extent[0]/2 + bbox2.extent[0]/2)
    aligned_y = np.abs(bbox1.center[1] - bbox2.center[1]) <= (bbox1.extent[1]/2 + bbox2.extent[1]/2)
    aligned_z = np.abs(bbox1.center[2] - bbox2.center[2]) <= (bbox1.extent[2]/2 + bbox2.extent[2]/2)
    
    if aligned_x or aligned_y or aligned_z:
        if distance_x <= threshold or distance_y <= threshold or distance_z <= threshold:
            return True
        else:
            return False
    else:
        return False

def is_bigger(bbox1, bbox2):
    """
    Check if bbox1 is bigger than bbox2.
    """
    return bbox1.volume() > bbox2.volume()


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

def batch_process(test_flag=False, scans_txt_name = 'scans.txt'):
    import shutil

    scan_info_dir = '/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity'
    DATA_ROOT = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R"
    PKL_FILENAME = "full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz"
    
    remove_exist_flag = True # BE AWARE OF THIS SETTING !!!!!!
    # remove_exist_flag = False
    
    
    if test_flag:
        test_DATA_ROOT = "/data0/vln_datasets/preprocessed_data/test_data/testdata_cg_gpt4_replace"
        DATA_ROOT = test_DATA_ROOT
        test_PKL_FILENAME = "full_pcd_ram_withbg_allclasses_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz"
        PKL_FILENAME = test_PKL_FILENAME
        remove_exist_flag = True


    with open(os.path.join(scan_info_dir, scans_txt_name)) as f:
        scans = [x.strip() for x in f]      # load all scans
        
    missing_dir = []
    
    for scan in tqdm(scans, desc="Processing scans"):
        args = ProgramArgs(
            mode="build-scenegraph",
            cachedir=f"{DATA_ROOT}/{scan}/sg_cache",
            mapfile=f"{DATA_ROOT}/{scan}/pcd_saves/{PKL_FILENAME}",
            class_names_file=f"{DATA_ROOT}/{scan}/gsa_classes_ram_withbg_allclasses.json"  
        )
        if not os.path.exists(args.mapfile):
            missing_dir.append(scan)
            continue
        try:
            os.mkdir(args.cachedir)
            print("Processing scan: ", scan)
            generate_object_edges_by_rules(args)
        except FileExistsError:
            if remove_exist_flag:
                print("Remove existing cache dir: ", args.cachedir)
                shutil.rmtree(args.cachedir)
                os.mkdir(args.cachedir)
                print("Processing scan: ", scan)
                generate_object_edges_by_rules(args)
            else:
                print("Processing scan: ", scan)
                generate_object_edges_by_rules(args)
    print(missing_dir)


def get_edges_from_file(scan, data_root = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R/", obj_file_name="cfslam_scenegraph_nodes.pkl.gz", edges_file_name="cfslam_scenegraph_edges.pkl"):

    with gzip.open(f"{data_root}/{scan}/sg_cache/{obj_file_name}", 'rb') as f:
        objs_gz = pkl.load(f)
    ojbs = objs_gz['objects']
    
    with open(f"{data_root}/{scan}/sg_cache/{edges_file_name}", 'rb') as f:
        edges = pkl.load(f)

    return ojbs, edges

def get_all_edges_from_file(
    data_root = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R", 
    obj_file_name="cfslam_scenegraph_nodes.pkl.gz",
    edges_file_name="cfslam_scenegraph_edges.pkl", 
    scans_allocation_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/scans_allocation"):

    # with open(os.path.join(scans_allocation_dir, 'all_scans.txt')) as f:
    with open(os.path.join(scans_allocation_dir, 'test_scan.txt')) as f:  
        scans = [x.strip() for x in f]    

    with open(os.path.join(scans_allocation_dir, 'eval_scans.txt')) as f:
        eval_scans = [x.strip() for x in f] 
    
    train_scans = list(set(scans) - set(eval_scans))
    
    dict_scan_edges = {}
    for scan in tqdm(train_scans):
        print("Loading scan: ", scan)
        with gzip.open(f"{data_root}/{scan}/sg_cache/{obj_file_name}", 'rb') as f:
            objs_gz = pkl.load(f)
        objs = [obj["bbox_np"] for obj in objs_gz['objects']]

        with open(f"{data_root}/{scan}/sg_cache/{edges_file_name}", 'rb') as f:
            edges = pkl.load(f)
        dict_scan_edges[scan] = (objs, edges)
    return dict_scan_edges

def get_all_edges_from_file_specific_scans(
    data_root = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R", 
    obj_file_name="cfslam_scenegraph_nodes.pkl.gz",
    edges_file_name="cfslam_scenegraph_edges.pkl", 
    scans_allocation_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity",
    scans_txt_name = 'scans.txt',):

    # with open(os.path.join(scans_allocation_dir, 'all_scans.txt')) as f:
    with open(os.path.join(scans_allocation_dir, scans_txt_name)) as f:  
        scans = [x.strip() for x in f]    

    dict_scan_edges = {}
    for scan in tqdm(scans):
        print("Loading scan: ", scan)
        with gzip.open(f"{data_root}/{scan}/sg_cache/{obj_file_name}", 'rb') as f:
            objs_gz = pkl.load(f)
        objs = [obj["bbox_np"] for obj in objs_gz['objects']]

        with open(f"{data_root}/{scan}/sg_cache/{edges_file_name}", 'rb') as f:
            edges = pkl.load(f)
        dict_scan_edges[scan] = (objs, edges)
    return dict_scan_edges


def get_edges_from_hdf5(hdf5_path="/data0/vln_datasets/preprocessed_data/edges_hdf5", hdf5_file_name="edges.hdf5"):
    import h5py
    edges_dict = {}
    hdf5_path = os.path.join(hdf5_path, hdf5_file_name)
    
    with h5py.File(hdf5_path, 'r') as f:
        for scan in f.keys():
            # Initialize an empty list to store combined edges for the current scan
            combined_edges = []
            
            # Load the 'objs' dataset as is
            objs = f[scan]['objs'][:]
            
            # Check if the 'edges_integers' and 'edges_strings' datasets exist
            if 'edges_integers' in f[scan] and 'edges_strings' in f[scan]:
                edges_integers = f[scan]['edges_integers'][:]
                edges_strings = f[scan]['edges_strings'][:]
                
                # Combine integers and strings back into the original mixed structure
                for i in range(len(edges_integers)):
                    combined_edge = list(edges_integers[i]) + [edges_strings[i].decode('utf-8')]
                    combined_edges.append(combined_edge)
            else:
                # Fallback if the original 'edges' dataset exists without splitting
                combined_edges = f[scan]['edges'][:]
            
            # Store the loaded data in the dictionary
            edges_dict[scan] = (objs, combined_edges)
    return edges_dict

def get_edges_from_dict(dict_scan_edges, scan):
    objs_bbox_list, edges_relationship = dict_scan_edges[scan]
    objs = [{"bbox_np": obj} for obj in objs_bbox_list]
    return objs, edges_relationship

def find_correspondence(objs, obj_clip):
    # # Example usage
    # correspondence_dict = find_correspondence(objs, obj_clip)
    # print(correspondence_dict)
    import torch
    correspondence_dict = {}
    
    for i, obj in enumerate(objs):
        for j, clip_tensor in enumerate(obj_clip):
            # Assuming you're checking for exact equality
            if torch.equal(torch.tensor(obj["clip_ft"]), torch.tensor(clip_tensor)):
                # If exact match, add to correspondence dictionary
                # Key: Index in objs, Value: Index in obj_clip
                correspondence_dict[i] = j
                break  # Assuming one-to-one correspondence, break after finding the first match
    return correspondence_dict

def find_correspondence_similarity(objs, obj_clip, similarity_threshold=0.99):
    # # Example usage
    # correspondence_dict_similarity = find_correspondence_similarity(objs, obj_clip)
    # print(correspondence_dict_similarity)
    import torch
    from torch.nn.functional import cosine_similarity
    correspondence_dict = {}
    
    for i, obj in enumerate(objs):
        for j, clip_tensor in enumerate(obj_clip):
            # Ensure tensors are flattened (cosine_similarity expects inputs of shape (1, N))
            obj_ft = torch.tensor(obj["clip_ft"])
            clip_tensor = torch.tensor(clip_tensor)
            similarity = cosine_similarity(obj_ft.flatten().unsqueeze(0), clip_tensor.flatten().unsqueeze(0))
            print(similarity)
            if similarity >= similarity_threshold:
                correspondence_dict[i] = j
                break  # Assuming one-to-one correspondence based on similarity
        print("another")

    return correspondence_dict

def calculate_center(bbox_corners):
    """Calculate the center of a bbox given its eight corners."""
    return np.mean(bbox_corners, axis=0)

def find_correspondece_similarity_bboxcenter(objs, obj_pos, similarity_threshold=0.2, decimals=5):
    import numpy as np
    """Find correspondence based on the similarity of bbox centers."""
    correspondence_dict = {}
    
    # Calculate centers for each bbox in objs
    objs_centers = [calculate_center(obj["bbox_np"]) for obj in objs]
    
    # Iterate over each center in obj_pos
    for i, pos_center in enumerate(obj_pos):
        # Compare with each center in objs
        for j, obj_center in enumerate(objs_centers):
            # Calculate Euclidean distance to determine similarity
            distance = np.linalg.norm(np.around(pos_center, decimals=decimals) - np.around(obj_center, decimals=decimals))
            # If distance is within the threshold, add to correspondence_dict
            if distance < similarity_threshold:
                correspondence_dict[j] = i
                break  # Assuming one-to-one correspondence, stop after the first match
    return correspondence_dict


def get_obj_edges(scan, obj_clip, obj_pos):
    
    edges_objects = None
    edges_relationship = None
    edges_relationship_extend = None

    # objs, edges_relationship = get_edges_from_file(scan)
    # dict_scan_edges = get_all_edges_from_file()
    dict_scan_edges = get_edges_from_hdf5()
    
    objs, edges_relationship = get_edges_from_dict(dict_scan_edges, scan)
    
    # correspondence_dict = find_correspondence_similarity(objs, obj_clip)
    correspondence_dict = find_correspondece_similarity_bboxcenter(objs, obj_pos)
    # Correspondec_dict is a dict, the key is the index of the objs, and the value is the index of obj_pos.
    
    edges = []  
    for relationship in edges_relationship:
        edge = []
        if relationship[0] in correspondence_dict and relationship[1] in correspondence_dict and relationship[2] != "none of these":
            edge.append(correspondence_dict[relationship[0]])
            edge.append(correspondence_dict[relationship[1]])
            edge.append(relationship[2])
            edges.append(edge)
        else:
            continue
        
    # edges_objs is edges first two columns
    edges_objects = [edge[:2] for edge in edges]
    # edges_relationship is the third column
    edges_relationship = [edge_encode(edge[2], False) for edge in edges]
    
    # exntend
    _edges_objects_extend = [[edge[1], edge[0]] for edge in edges]
    _edges_relationship_extend =  [edge_encode(edge[2], True) for edge in edges]
    
    edges_objects_extend = []
    edges_relationship_extend = []
    edges_objects_extend.extend(edges_objects)
    edges_objects_extend.extend(_edges_objects_extend)
    edges_relationship_extend.extend(edges_relationship)
    edges_relationship_extend.extend(_edges_relationship_extend)
      
    # return edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend
    return np.array(edges_objects), np.array(edges_relationship), np.array(edges_objects_extend), np.array(edges_relationship_extend)

def edge_encode(edge_relationship, revert_flag = False):
    
    if revert_flag == False:
        if edge_relationship == "a on b":
            encoded_relationship = 0
        elif edge_relationship == "a in b":
            encoded_relationship = 1
        elif edge_relationship == "b on a":
            encoded_relationship = 2
        elif edge_relationship == "b in a":
            encoded_relationship = 3
        elif edge_relationship == "none of these":
            encoded_relationship = 4
    elif revert_flag == True:
        if edge_relationship =="a on b":
            encoded_relationship = 2
        elif edge_relationship == "a in b":
            encoded_relationship = 3
        elif edge_relationship == "b on a":
            encoded_relationship = 0
        elif edge_relationship == "b in a":
            encoded_relationship = 1
        elif edge_relationship == "none of these":
            encoded_relationship = 4
    
    return encoded_relationship

def test_get_obj_edges(scan="17DRP5sb8fy"):
    import h5py
    
    cg_file_path = "/home/lg1/lujia/VLN_HGT/pretrain_src/datasets/cg_data/cgobj_clip.hdf5"
    path_mapping_file = "/home/lg1/lujia/VLN_HGT/pretrain_src/datasets/cg_data/all_paths.json"
    
    with h5py.File(cg_file_path, 'r') as f:
        batch_obj_clips = f[f'{scan}_obj_clip_ft'][...].astype(np.float32)
        paths_map_indices = f[f'{scan}_paths_indices'][...].astype(np.float32)
        batch_obj_pos = f[f'{scan}_obj_pos'][...].astype(np.float32)
    

    # read the paths_indices from json
    with open(path_mapping_file, 'r') as f:
        all_paths_indices = json.load(f)
    # get the object list and pos information
    # get path index from a list by input the element
    paths_indices = all_paths_indices[scan]
    
    path = paths_indices[0] # For test
    
    path_indx = paths_indices.index(path)
    # get the path_map_index
    path_map_index = paths_map_indices[path_indx]
    # get the obj_clip_np
    obj_clip, obj_pos = get_scene_map(batch_obj_clips, batch_obj_pos, path_map_index)

    # edges_objects, edges_relationship, edges_relationship_extend = get_obj_edges(scan, obj_clip)
    edges_ojects, edges_relationship, edges_objects_extend, edges_relationship_extend = get_obj_edges(scan, obj_clip, obj_pos)
    
    print("Fine")

def get_scene_map(batch_obj_clip_np, batch_obj_pos_np, path_map_index):

        obj_indices = np.arange(batch_obj_clip_np.shape[0])
        valid_obs_combination = np.where(path_map_index >= 0, path_map_index, 0).astype(int)
        selected_obj_clip = batch_obj_clip_np[obj_indices, valid_obs_combination,:]
        # mask = path_map_index >= 0
        mask = path_map_index > 0

        obj_clip = selected_obj_clip[mask]
        obj_pos = batch_obj_pos_np[mask]
        return obj_clip, obj_pos

def edges_pkls_to_hdf5(
    data_root = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R",
    obj_file_name="cfslam_scenegraph_nodes.pkl.gz",
    edges_file_name="cfslam_scenegraph_edges.pkl",
    data_edges_hdf5 = "/data0/vln_datasets/preprocessed_data/edges_hdf5"
    ):
    import h5py
    
    if not os.path.exists(data_edges_hdf5):
        os.mkdir(data_edges_hdf5)
    
    ## This is for train data
    # edges_dict = get_all_edges_from_file(
    #     data_root, 
    #     obj_file_name,
    #     edges_file_name, 
    #     scans_allocation_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/scans_allocation")
    
    # This is for test data, I used to call the test data as eval data.
    edges_dict = get_all_edges_from_file_specific_scans(
        data_root,
        obj_file_name,
        edges_file_name,
        scans_allocation_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity",
        scans_txt_name="eval_scans.txt"
    )
    
    # create a hdf5 dataset to store edges_dict in data_deges_hdf5
    # check if the edges.hdf5 is exist, if it is, delet it  
    if os.path.exists(os.path.join(data_edges_hdf5, "edges.hdf5")):
        os.remove(os.path.join(data_edges_hdf5, "edges.hdf5"))
        
    with h5py.File(os.path.join(data_edges_hdf5, "edges.hdf5"), 'w') as f:
        for scan in edges_dict.keys():
            scan_group = f.create_group(scan)
            # Assuming edges_dict[scan][0] is not mixed and doesn't cause issues
            objs = scan_group.create_dataset("objs", data=edges_dict[scan][0])
            
            # Splitting the mixed list into integers and strings
            edges_integers = np.array([item[:2] for item in edges_dict[scan][1]], dtype=np.int32)
            edges_strings = np.array([item[2] for item in edges_dict[scan][1]], dtype=h5py.special_dtype(vlen=str))
            
            # Creating separate datasets for integers and strings
            edges_ints_ds = scan_group.create_dataset("edges_integers", data=edges_integers)
            edges_strs_ds = scan_group.create_dataset("edges_strings", data=edges_strings)
            
    if os.path.exists(os.path.join(data_edges_hdf5, "edges.hdf5")):
        print("The hdf5 dataset of edges is created successfully")

def main():
    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)
    
    # print using masking option
    print(f"args.masking_option: {args.masking_option}")

    if args.mode == "extract-node-captions":
        extract_node_captions(args)
    elif args.mode == "refine-node-captions":
        refine_node_captions(args)
    elif args.mode == "build-scenegraph":
        # build_scenegraph(args)
        generate_object_edges_by_rules(args)
    elif args.mode == "generate-scenegraph-json":
        generate_scenegraph_json(args)
    elif args.mode == "annotate-scenegraph":
        annotate_scenegraph(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
def test_load_pkl_file():
    
    edges_file_addr = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R/17DRP5sb8fy/sg_cache/cfslam_scenegraph_edges.pkl"
    obj_file_addr = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R/17DRP5sb8fy/sg_cache/cfslam_scenegraph_nodes.pkl.gz"
    
    with open(edges_file_addr, 'rb') as f:
        edges = pkl.load(f)

    with gzip.open(obj_file_addr, 'rb') as f:
        objs_gz = pkl.load(f)
    objs = objs_gz['objects']
        
    print(edges)
    print(objs)

if __name__ == "__main__":
    # main()

    # test_load_pkl_file()
    # test_get_obj_edges()
    
    
    ## STEP 1 : Using batch_process generate the edge files
    # batch_process(scans_txt_name="eval_scans.txt")
    
    ## STEP 2 : Using edges_pkls_to_hdf5 to convert the pkls to hdf5
    edges_pkls_to_hdf5(
        data_root = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R",
        obj_file_name="cfslam_scenegraph_nodes.pkl.gz",
        edges_file_name="cfslam_scenegraph_edges.pkl",
        data_edges_hdf5 = "/data2/vln_dataset/test_m2g_processed"
    )
