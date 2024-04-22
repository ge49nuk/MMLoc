from model import Model
from datamodule import DataModule
import pytorch_lightning as pl
import torch
from config import Config

def main():
    torch.set_float32_matmul_precision('medium')
    cfg = Config()
    model = Model(cfg.model)
    data_module = DataModule(config=cfg.data)
    
    # wandb_logger = loggers.WandbLogger(project="MA-Thesis")
    # wandb_logger.watch(model)
    trainer = pl.Trainer()
    trainer.test(model=model, ckpt_path=cfg.ckpt, datamodule=data_module)

if __name__=="__main__":
    main()
    