from model import Model
from datamodule import DataModule
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os
from datetime import datetime
from config import Config
from pytorch_lightning.profilers import AdvancedProfiler

def main():
    torch.set_float32_matmul_precision('medium')
    cfg = Config()
    model = Model(cfg.model)
    data_module = DataModule(config=cfg.data)
    
    wandb_logger = loggers.WandbLogger(project="MA-Thesis")
    ckpt_path = os.path.join(cfg.data["data_path"], "checkpoints", wandb_logger.experiment.id)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=ckpt_path)
    
    profiler = AdvancedProfiler(filename="profiler_out")
    trainer = pl.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], **cfg.trainer, profiler=profiler)
    
    trainer.fit(model=model, datamodule=data_module)

if __name__=="__main__":
    main()
    