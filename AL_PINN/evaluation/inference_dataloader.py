# Data loader for group processing inference data

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ValidationDataset(Dataset):
    def __init__(self, data_folder, file_names):
        self.file_list = [os.path.join(data_folder, file) for file in os.listdir(data_folder)]
        self.file_names = file_names

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        sample = np.load(file_path)
        return torch.from_numpy(sample).float(), self.file_names[index]

        # file_path = self.file_list[index]
        # sample = torch.load(file_path)
        # return sample.to(torch.device("cuda")), self.file_names[index]

def collate_fn(data):
    # data is a list of tuples, where each tuple contains a single sample
    # Extract the samples and stack them into a batch
    samples = [sample[0] for sample in data]
    batch_samples = torch.stack(samples, dim=0).squeeze(1)

    batch_file_names = [sample[1] for sample in data]
    return batch_samples, batch_file_names

# Set the batch size and number of workers for data loading
batch_size = 4
num_workers = 4
def get_validation_dataloader(validation_data_root = '//data/zxiaoal/Validation_Dataset/data'):
    file_names = [file for file in os.listdir(validation_data_root) if file.endswith('.npy')]
    # file_names = [file for file in os.listdir(validation_data_root) if file.endswith('.pth')]
    # Create the validation dataset
    validation_dataset = ValidationDataset(validation_data_root, file_names)

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers = num_workers,
        collate_fn=collate_fn
    )
    return validation_dataloader