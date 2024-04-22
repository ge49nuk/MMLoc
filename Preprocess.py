import os
import json
import numpy as np
from utils import utils
import cv2
import torchvision.transforms as transforms
import torch
from config import Config
from tqdm import tqdm

def init_rgb_dict(cfg, bop_path, used_rgb_ids):
    mu_rgb, std_rgb = cfg["rgb_mean_std"]
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mu_rgb, std_rgb),
            ],
        )
    
    rgb_dict = {}
    for rgb_id in tqdm(used_rgb_ids, desc="Loading rgb data"):
        folder =  str(rgb_id//1000).zfill(6)
        RGB_PATH = os.path.join(bop_path, folder, "rgb")
        fn = str(rgb_id%1000).zfill(6) + ".jpg"
        image = cv2.imread(os.path.join(RGB_PATH, fn), cv2.IMREAD_COLOR)/255.
        rgb_dict[rgb_id] = transform(image)
    return rgb_dict

def init_mesh_dict(mesh_data_path, used_obj_ids):
    mesh_dict = {}
    for obj_id in tqdm(used_obj_ids, desc="Loading mesh data"):
        file = "obj_" + str(obj_id).zfill(6) + ".pt"
        mesh = torch.load(os.path.join(mesh_data_path, file))
        vertices = mesh["vertices"]
        mesh_dict[obj_id] = vertices
    return mesh_dict

def load_data(cfg, bop_path, split):
    print(f"Loading {split} data...")
    data = []
    used_rgb_ids = set()
    used_obj_ids = set()
    bbox_mean, bbox_std = cfg["bbox_mean_std"] 

    bop_scenes = os.listdir(bop_path)
    train_end = int(cfg["train_val_split"] * len(bop_scenes))
    if split == "train":
        bop_scenes = bop_scenes[:train_end]
    else:
        bop_scenes = bop_scenes[train_end:]
        
    for folder in bop_scenes: # BOP scenes
        with open(os.path.join(bop_path, folder, "scene_gt.json"), "r") as f:
            scene_gt = json.load(f)
        with open(os.path.join(bop_path, folder, "scene_gt_info.json"), "r") as f2:
            scene_gt_info = json.load(f2)
        keys = list(scene_gt.keys())

        # Iterate through images
        for key in keys:
            gt_row = scene_gt[key]
            if key not in scene_gt_info:
                continue
            gt_info_row = scene_gt_info[key]

            # Iterate through queried objects
            for i, object in enumerate(gt_row):
                gt_info_object = gt_info_row[i]
                if gt_info_object["visib_fract"] < cfg["min_visib_fract"]:
                    continue
                # initialize dataset sample
                data_dict = {"rgb_id": int(folder)*1000 + int(key)}
                data_dict["translation"] = object["cam_t_m2c"]
                data_dict["rotation"] = object["cam_R_m2c"]
                bbox = np.array(gt_info_object["bbox_visib"], dtype=np.float32)
                bbox = (bbox - bbox_mean) / bbox_std
                data_dict["bbox"] = bbox.astype(np.float32)
                data_dict["obj_id"] = object["obj_id"]
                data.append(data_dict)
                
                used_rgb_ids.add(data_dict["rgb_id"])
                used_obj_ids.add(data_dict["obj_id"])
    
    print(f"{split} dataset: {len(data)} samples, {len(used_obj_ids)} meshes, {len(used_rgb_ids)} images")
    return data, used_obj_ids, used_rgb_ids

def main():
    cfg = Config().data
    bop_path = os.path.join(cfg["data_path"], "bop_data/train_pbr")
    mesh_data_path = os.path.join(cfg["data_path"], "mesh_data")
    
    data, used_obj_ids, used_rgb_ids = load_data(cfg, bop_path, "train")
    rgb_dict = init_rgb_dict(cfg, bop_path, used_rgb_ids)
    mesh_dict = init_mesh_dict(mesh_data_path, used_obj_ids)
    # rgb_dict = init_rgb_dict(cfg, bop_path, used_rgb_ids)
    dataset = {"Data":data, "RGB":rgb_dict, "XYZ":mesh_dict}
    torch.save(dataset, os.path.join(cfg["data_path"], "dataset_train.pt"))
    
    data, used_obj_ids, used_rgb_ids = load_data(cfg, bop_path, "val")
    mesh_dict = init_mesh_dict(mesh_data_path, used_obj_ids)
    rgb_dict = init_rgb_dict(cfg, bop_path, used_rgb_ids)
    dataset = {"Data":data, "RGB":rgb_dict, "XYZ":mesh_dict}
    torch.save(dataset, os.path.join(cfg["data_path"], "dataset_val.pt"))

if __name__=="__main__":
    main()