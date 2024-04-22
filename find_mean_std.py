import os
import numpy as np
import cv2
import json
from config import Config
from tqdm import tqdm

def main(cfg):
    bop_path = os.path.join(cfg["data_path"], "bop_data/train_pbr")
    bboxes = []
    means_rgb = []
    stds_rgb = []
    
    bop_scenes = os.listdir(bop_path)
    train_end = int(cfg["train_val_split"] * len(bop_scenes))
    bop_scenes = bop_scenes[:train_end]
            
    for folder in tqdm(bop_scenes, desc="Loading BOP-data"): # BOP scenes
        
        used_rgb_ids = set()
        
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
                used_rgb_ids.add(int(folder)*1000 + int(key))
                bboxes.append(np.array(gt_info_object["bbox_visib"], dtype=np.float32))
        
        # Calc mean, var of folder
        rgb_images = []
        for rgb_id in used_rgb_ids:
            folder =  str(rgb_id//1000).zfill(6)
            rgb_path = os.path.join(bop_path, folder, "rgb")
            fn = str(rgb_id%1000).zfill(6) + ".jpg"
            image = cv2.imread(os.path.join(rgb_path, fn), cv2.IMREAD_COLOR)
            rgb_images.append(image.reshape(-1,3)/255.)
        
        means_rgb.append(np.mean(rgb_images, axis=(0,1)))
        stds_rgb.append(np.std(rgb_images, axis=(0,1)))

    mean_bbox = np.mean(bboxes, axis=0)
    std_bbox = np.std(bboxes, axis=0)
    
    mean_rgb = np.mean(means_rgb, axis=0)
    std_rgb = np.mean(stds_rgb, axis=0)
    
    formatter = lambda x: f"{x:.5f}"
    
    mean_str = np.array2string(mean_rgb, separator=', ', formatter={'all': formatter})[1:-1]
    std_str = np.array2string(std_rgb, separator=', ', formatter={'all': formatter})[1:-1]
    rgb_str = f"(({mean_str}), ({std_str}))"
    print(f"RGB: {rgb_str}")
    mean_str = np.array2string(mean_bbox, separator=', ', formatter={'all': formatter})[1:-1]
    std_str = np.array2string(std_bbox, separator=', ', formatter={'all': formatter})[1:-1]
    bbox_str = f"(({mean_str}), ({std_str}))"
    print("bounding box:", bbox_str)

if __name__=="__main__":
    cfg = Config()
    main(cfg.data)
    