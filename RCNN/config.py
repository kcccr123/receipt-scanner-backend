import os
import yaml
from datetime import datetime
import numpy as np

class ConfigFile:
    def __init__(self, name: str, path: str,  h: int = 32, w: int = 128, 
                 bs: int = 64, lr: float = 0.002, epoch: int = 100): #mean: np.ndarray, std: np.ndarray,
        
        self.name = name
        self.model_path = os.path.join(path, datetime.strftime(datetime.now(), "%Y%m%d%H%M")).replace("\\","/")
        self.height = h
        self.width = w
        self.batch_size = bs
        self.lr = lr
        self.train_epochs = epoch
        self.max_txt_len = 0
        self.vocab = ""
        # self.mean = mean
        # self.std = std

    def to_dict(self):
        return {
            'name': self.name,
            'model_path': self.model_path,
            'height': self.height,
            'width': self.width,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'train_epochs': self.train_epochs,
            'max_txt_len': self.max_txt_len,
            'vocab': self.vocab,
            # 'mean': self.mean,
            # 'std': self.std
        }

    def save(self):
        name = self.name
        if self.model_path is None:
            raise Exception("Model path is not specified")

        # create directory if not exist
        if not os.path.exists(self.model_path):
            print("config saved to" + self.model_path)
            os.makedirs(self.model_path)

        with open(os.path.join(self.model_path, name+".yml"), "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def load(configs_path: str):
        with open(configs_path, "r") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        config = ConfigFile()
        
        for key, value in configs.items():
            setattr(config, key, value)

        return config