import pytorch_lightning as pl
import torch
from cnn import CNN
from utils import utils

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.cnn = CNN(dropout_p=cfg["dropout_p"])
        self.training_step_losses = torch.tensor([])
        self.val_step_losses = torch.tensor([])
        
        # For testing
        self.counts = torch.tensor([])
        self.iou50 = 0
        self.iou90 = 0
        self.avg_ious = torch.tensor([])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.cnn.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #         optimizer=optimizer,
        #         step_size=20,
        #         gamma=0.1,
        #     )
        # return [optimizer], [scheduler] 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.1,
            patience=self.cfg["scheduler_patience"],
            min_lr=1e-10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "frequency": 1,
            },
        }

    def forward(self, input_dict):
        return self.cnn(input_dict)

    def training_step(self, batch, _):
        bs = len(batch["obj_id"])
        output = self(batch)
        self.training_step_losses = torch.cat((self.training_step_losses, output["loss"].cpu().unsqueeze(0)))
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log(
            "train_loss",
            output["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bs,
        )
        self.log(
                'learning_rate', 
                lr, on_epoch=True, 
                on_step=False, 
                prog_bar=True, 
                logger=True, 
                batch_size=bs
        )
        return output["loss"]

    def validation_step(self, batch, _):
        bs = len(batch["obj_id"])
        output = self(batch)
        self.log(
            "val_loss",
            output["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bs,
        )
        self.val_step_losses = torch.cat((self.val_step_losses, output["loss"].cpu().unsqueeze(0)))
        return output["loss"]

    def test_step(self, batch, _):
        bs = len(batch["obj_id"])
        output = self(batch)
        for i, pred in enumerate(output["preds"]):
            gt = batch["bbox"][i]
            rgb_id = batch["rgb_id"][i]
            rgb_index = batch["rgb_id_to_rgb"][rgb_id]
            rgb = batch["rgb"][rgb_index]
            utils.visualize_bbox(rgb, pred, gt, f"output/bbox_{i}.png")
        exit(0)
        ious = output["ious"]
        self.counts = torch.cat((self.counts, torch.tensor([ious.shape[0]])))
        self.avg_ious = torch.cat((self.avg_ious, torch.mean(ious).unsqueeze(0)))
        iou50 = ious[ious>=0.5]
        iou90 = ious[ious>=0.9]
        self.iou50 += iou50.shape[0]
        self.iou90 += iou90.shape[0]
        
        self.log(
            "test_iou",
            torch.mean(ious),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bs,
        )
        return output["loss"]
    
    def on_train_epoch_end(self):
        loss = torch.mean(self.training_step_losses).item()
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.training_step_losses = torch.tensor([])
        # self.logger.experiment.log({"train_loss": loss}, step=self.trainer.current_epoch)
        # self.logger.experiment.log({"lr": lr}, step=self.trainer.current_epoch)
        
    def on_validation_epoch_end(self):
        loss = torch.mean(self.val_step_losses).item()
        self.val_step_losses = torch.tensor([])
        # self.logger.experiment.log({"val_loss": loss}, step=self.trainer.current_epoch)
        
    def on_test_epoch_end(self):
        print("Testing finished!\nResults:")
        counts = self.counts
        avg_ious = self.avg_ious
        num_samples = torch.sum(counts) 
        avg_iou = torch.sum(counts * avg_ious) / num_samples
        iou50 = self.iou50 / num_samples
        iou90 = self.iou90 / num_samples
        print(f"Average IoU: {avg_iou:.3f}")
        print(f"IoU 50: {iou50:.3f}")
        print(f"IoU 90: {iou90:.3f}")
        
    def inference(self, vertices, rgb):
        bbox = self.cnn(vertices, rgb)
        return bbox
