import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import GeneralDataset
import torch

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.dataset = GeneralDataset

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset("train", self.cfg)
            self.val_set = self.dataset("val", self.cfg)
        if stage == "test" or stage is None:
            self.test_set = self.dataset("test", self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.cfg["batch_size"], shuffle=False, persistent_workers=True,
            pin_memory=True, collate_fn=_sparse_collate_fn, num_workers= self.cfg["num_workers"]
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.cfg["batch_size"],
            pin_memory=True, collate_fn=_sparse_collate_fn, num_workers= self.cfg["num_workers"]
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.cfg["batch_size"],
            pin_memory=True, collate_fn=_sparse_collate_fn, num_workers= self.cfg["num_workers"]
        )

def _sparse_collate_fn(batch):
    data = {}
    rgb_id = []
    rgb = []
    translation = []
    rotation = []
    bbox = []
    obj_id = []
    vertices = []
    rgb_id_to_rgb = {}
    obj_id_to_pts = {}

    for i, b in enumerate(batch):
        rgb_id.append(b["rgb_id"])
        translation.append(torch.tensor(b["translation"]))
        rotation.append(torch.tensor(b["rotation"]))
        bbox.append(torch.tensor(b["bbox"]))
        obj_id.append(b["obj_id"])
        if b["rgb_id"] not in rgb_id_to_rgb:
            rgb_id_to_rgb[b["rgb_id"]] = len(rgb)
            rgb.append(b["rgb"])
        if b["obj_id"] not in obj_id_to_pts:
            obj_id_to_pts[b["obj_id"]] = len(vertices)
            vertices.append(b["vertices"])

    data["rgb_id"] = rgb_id
    data["rgb"] = torch.stack(rgb)
    data["translation"] = torch.stack(translation)
    data["rotation"] = torch.stack(rotation)
    data["bbox"] = torch.stack(bbox)
    data["obj_id"] = obj_id
    data["vertices"] = torch.stack(vertices)
    data["obj_id_to_pts"] = obj_id_to_pts
    data["rgb_id_to_rgb"] = rgb_id_to_rgb
    return data
