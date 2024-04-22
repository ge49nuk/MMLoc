import cv2
import numpy as np
from config import Config
from torchvision import transforms
import torch

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

def visualize_bbox(rgb, bbox_pred, bbox_gt=None, filename="bbox.png", normalized=True):
    if normalized:
        rgb, bbox_pred, bbox_gt = denormalize(rgb, bbox_pred, bbox_gt)
        
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1,2,0))
    rgb = cv2.cvtColor(rgb, cv2.IMREAD_COLOR)

    x, y, width, height = bbox_pred.astype(np.uint8)
    cv2.rectangle(rgb, (x, y), (x + width, y + height), (255, 0, 0), 2)
    if bbox_gt is not None:
        x, y, width, height = bbox_gt.astype(np.uint8)
        cv2.rectangle(rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.imwrite(filename, rgb)
    
def denormalize(rgb, bbox_pred, bbox_gt):
    cfg = Config()
    rgb_mean, rgb_std = cfg.data["rgb_mean_std"]
    inv_normalize = transforms.Normalize(
        mean=[-rgb_mean[0] / rgb_std[0], -rgb_mean[1] / rgb_std[1], -rgb_mean[2] / rgb_std[2]],
        std=[1 / rgb_std[0], 1 / rgb_std[1], 1 / rgb_std[2]]
    )
    rgb = inv_normalize(rgb).cpu().numpy()
    rgb = (rgb * 255).astype(np.uint8)
    
    bbox_mean, bbox_std = cfg.data["bbox_mean_std"]
    bbox_pred = (bbox_pred.cpu() * torch.tensor(bbox_std) + torch.tensor(bbox_mean)).numpy()
    bbox_gt = (bbox_gt.cpu() * torch.tensor(bbox_std) + torch.tensor(bbox_mean)).numpy()
    return rgb, bbox_pred, bbox_gt
    