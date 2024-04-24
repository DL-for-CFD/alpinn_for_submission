import torchvision
import torch.utils.data
from torch.utils.data import Dataset
import torch
import numpy as np
import os

batch_size_G = 2
batch_size_D = 2

class ResultsDataset(Dataset):
    def __init__(self, np_results_file_dir):

        self.results_file_dir = np_results_file_dir
        self.results_files = os.listdir(self.results_file_dir)
        self.results_files = [os.path.join(self.results_file_dir, file) for file in self.results_files if file.endswith('.npy')]
        self.results_files.sort()

    def __len__(self):
        return len(self.results_files)
    
    def __getitem__(self, idx):
        # load npy file
        results = np.load(self.results_files[idx])
        # convert to torch tensor
        results = torch.from_numpy(results)
        t, c, h, w = results.shape
        results = results.view(t*c, h, w)
        results = torchvision.transforms.functional.resize(results, (128, 128*3))
        # return results
        return results

class InputDataset(Dataset):
    def __init__(self, np_input_file_dir):

        self.input_file_dir = np_input_file_dir
        self.input_files = os.listdir(self.input_file_dir)
        self.input_files = [os.path.join(self.input_file_dir, file) for file in self.input_files if file.endswith('.npy')]
        self.input_files.sort()

    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # load npy file
        input = np.load(self.input_files[idx])
        # convert to torch tensor
        input = torch.from_numpy(input)
        input = torchvision.transforms.functional.resize(input, (128, 128*3))
        # return input
        return input

def get_netG_dataloader(netG_dataroot = '/localdata/rzhangbq/RL_pretraining_data/results_2/mu_0.1_rho_4.0'):
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset for generator
    netG_dataset = ResultsDataset(netG_dataroot)
    # Create the dataloader for generator
    netG_dataloader = torch.utils.data.DataLoader(netG_dataset, batch_size=batch_size_G,
                                            shuffle=True)
    return netG_dataloader


def get_netD_dataloader(netD_dataroot = '/localdata/rzhangbq/RL_pretraining_data/inputs'):
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset for discriminator
    netD_dataset = InputDataset(netD_dataroot)
    # Create the dataloader for discriminator
    netD_dataloader = torch.utils.data.DataLoader(netD_dataset, batch_size=batch_size_D,
                                            shuffle=True)
    return netD_dataloader