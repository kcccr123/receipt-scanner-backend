import typing 
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op
from itertools import groupby
from datetime import datetime
import yaml

class ConfigFile:
    def __init__(self, name: str, path: str, h: int = 32, w: int = 128, 
                 bs: int = 64, lr: float = 0.002, epoch: int = 100):
        
        self.name = name
        self.model_path = os.path.join(path, datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.height = h
        self.width = w
        self.batch_size = bs
        self.learning_rate = lr
        self.train_epochs = epoch
        self.max_text_length = 0
        self.vocab = ""

    def save(self):
        name = self.name
        if self.model_path is None:
            raise Exception("Model path is not specified")

        # create directory if not exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        with open(os.path.join(self.model_path, name), "w") as f:
            yaml.dump(self.serialize(), f)

    def load(configs_path: str):
        with open(configs_path, "r") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        config = ConfigFile()
        
        for key, value in configs.items():
            setattr(config, key, value)

        return config


