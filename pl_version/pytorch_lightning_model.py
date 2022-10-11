import pytorch_lightning as pl
import torch


class PlPointerGenerator(pl.LightningModule):
    def __init__(self, torch_model, args):
        """args is a json object"""
        super(PlPointerGenerator, self).__init__()
