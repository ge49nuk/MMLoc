class Config:
    def __init__(self):

        self.ckpt = "MA-Thesis/url239tk/checkpoints/epoch=99-step=400.ckpt"

        self.data = {
            "data_path": "/dl/volatile/students/projects/llms_for_code/master_thesis/",
            "batch_size": 256,
            "num_workers": 8,
            "train_val_split": 0.8,
            "rgb_mean_std": ((0.31502984, 0.33784471, 0.35396426), (0.25393445, 0.26602962, 0.27868257)),
            "bbox_mean_std": ((50.874825,  54.959476, 121.10628,  113.959465), (50.31403, 48.536953, 49.873207, 50.1974)),
            "min_visib_fract": 0.35,
            "aug": True,
        }

        self.model = {
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "scheduler_patience": 8,
            "dropout_p" : 0.0,
        }

        self.trainer = {
            "check_val_every_n_epoch": 10,
            "max_epochs": 350,
        }