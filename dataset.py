import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import cv2
import json
from transform import jitter, rot, flip, flip_bbox

class GeneralDataset(Dataset):
    def __init__(self, split, config):      
        self.bop_path = os.path.join(config["data_path"], "bop_data/train_pbr")
        self.cfg = config
        self.split = split
        self.init_dataset(split)
        
    def init_dataset(self, split):
        print(f"Loading {split} data...")
        self.data = []
        used_rgb_ids = set()
        used_obj_ids = set()
        bbox_mean, bbox_std = self.cfg["bbox_mean_std"] 
        
        bop_scenes = os.listdir(self.bop_path)
        train_end = int(self.cfg["train_val_split"] * len(bop_scenes))
        if split == "train":
            bop_scenes = bop_scenes[:train_end]
        else:
            bop_scenes = bop_scenes[train_end:]
            
        for folder in bop_scenes: # BOP scenes
            with open(os.path.join(self.bop_path, folder, "scene_gt.json"), "r") as f:
                scene_gt = json.load(f)
            with open(os.path.join(self.bop_path, folder, "scene_gt_info.json"), "r") as f2:
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
                    if gt_info_object["visib_fract"] < self.cfg["min_visib_fract"]:
                        continue
                    # initialize dataset sample
                    data_dict = {"rgb_id": int(folder)*1000 + int(key)}
                    data_dict["translation"] = object["cam_t_m2c"]
                    data_dict["rotation"] = object["cam_R_m2c"]
                    bbox = np.array(gt_info_object["bbox_visib"], dtype=np.float32)
                    bbox = (bbox - bbox_mean) / bbox_std
                    data_dict["bbox"] = bbox.astype(np.float32)
                    data_dict["obj_id"] = object["obj_id"]
                    self.data.append(data_dict)
                    
                    used_rgb_ids.add(data_dict["rgb_id"])
                    used_obj_ids.add(data_dict["obj_id"])
        
        print(f"{split} dataset: {len(self.data)} samples, {len(used_obj_ids)} meshes, {len(used_rgb_ids)} images")        
    
    def load_rgb(self, rgb_id):
        mu_rgb, std_rgb = self.cfg["rgb_mean_std"]
        transform = v2.Compose(
            [
                v2.ToTensor(),
                v2.Resize(224),
                v2.ConvertImageDtype(torch.float32),
                v2.Normalize(mu_rgb, std_rgb),
            ],
        )
        
        folder =  str(rgb_id//1000).zfill(6)
        RGB_PATH = os.path.join(self.bop_path, folder, "rgb")
        fn = str(rgb_id%1000).zfill(6) + ".jpg"
        image = cv2.imread(os.path.join(RGB_PATH, fn), cv2.IMREAD_COLOR)/255.
        return transform(image)
    
    def load_mesh(self, obj_id):
        file = "obj_" + str(obj_id).zfill(6) + ".pt"
        mesh = torch.load(os.path.join(self.cfg["data_path"], "mesh_data", file))
        vertices = mesh["vertices"]
        return vertices        
    
    def _get_augmentation_matrix(self, do_jitter=False, do_flip=False, do_rot=True):
        m = np.eye(3)
        if do_jitter:
            m = np.matmul(m, jitter())
        if do_flip:
            flip_m = flip(axis=-1)
            m *= flip_m
        if do_rot:
            rot_m = rot()
            m = np.matmul(m, rot_m)
        return m.astype(np.float32)
    
    def pc_norm(self, pc):
        """ taken from https://github.com/qizekun/ReCon/blob/main/datasets/ShapeNet55Dataset.py """
        centroid = torch.mean(pc, axis=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        rgb = self.load_rgb(data_dict["rgb_id"])
        verts = self.load_mesh(data_dict["obj_id"])

        if self.split == "train" and self.cfg["aug"]:
            do_flip = np.random.choice([True, False])
            transform = v2.Compose([v2.RandomPhotometricDistort()])
            rgb = transform(rgb)
            if do_flip:
                hor_flip = np.random.choice([True, False])
                if hor_flip:
                    rgb = v2.functional.horizontal_flip(rgb)
                    data_dict["bbox"] = flip_bbox(self.cfg, data_dict["bbox"], 0)
                else:
                    rgb = v2.functional.vertical_flip(rgb)
                    data_dict["bbox"] = flip_bbox(self.cfg, data_dict["bbox"], 1)
            aug_matrix = self._get_augmentation_matrix(do_flip=do_flip)
            verts = np.matmul(verts, aug_matrix) 
            
        data_dict["rgb"] = rgb
        data_dict["vertices"] = self.pc_norm(verts)
        
        return data_dict
    
    def __len__(self):
        return len(self.data)
