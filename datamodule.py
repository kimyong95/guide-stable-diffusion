import torch

from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import lightning as L
from torch.utils.data import DataLoader

class DummyDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 1
        self.dummy_dataloader = DataLoader(TensorDataset(torch.empty(1)))

    def train_dataloader(self):
        return self.dummy_dataloader

    def val_dataloader(self):
        return self.dummy_dataloader

    def test_dataloader(self):
        return self.dummy_dataloader

    def predict_dataloader(self):
        return self.dummy_dataloader
