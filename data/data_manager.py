import torch

from data.avenue_dataset import AvenueDataset
from data.avenue_dataset_ae import AvenueDatasetAE
from data.shanghai_dataset import ShanghaiDataset
from data.shanghai_dataset_ae import ShanghaiDatasetAE


class DataManager:
    def __init__(self, config):
        self.config = config

    def get_train_test_dataloaders_avenue(self):
        dataset_train = AvenueDataset(self.config, train=True)
        dataset_test = AvenueDataset(self.config, train=False)

        train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                   batch_size=self.config['batch_size'],
                                                   shuffle=True,
                                                   pin_memory=False)

        test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=self.config['batch_size'],
                                                  shuffle=False, pin_memory=False)
        return train_loader, test_loader

    def get_train_test_dataloaders_shanghai(self):
        dataset_train = ShanghaiDataset(self.config, train=True)
        dataset_test = ShanghaiDataset(self.config, train=False)

        train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                   batch_size=self.config['batch_size'],
                                                   shuffle=True,
                                                   pin_memory=False)

        test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=self.config['batch_size'],
                                                  shuffle=False, pin_memory=False)
        return train_loader, test_loader

    def get_dataloaders_avenue_AE(self):

        dataset_train = AvenueDatasetAE(self.config)

        train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                   batch_size=self.config['batch_size'],
                                                   shuffle=True,
                                                   pin_memory=True)

        return train_loader

    def get_dataloaders_shanghai_AE(self):
        dataset_train = ShanghaiDatasetAE(self.config)

        train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                   batch_size=self.config['batch_size'],
                                                   shuffle=True,
                                                   pin_memory=True)

        return train_loader
