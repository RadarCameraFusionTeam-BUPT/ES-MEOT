import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

class RoadPlaneDataset(TensorDataset):
    def __init__(self, folder_path):
        super().__init__()
        # Store the input and output tensors as attributes
        file_path = os.path.join(folder_path, 'output-PlanePoint.npy')
        if not os.path.exists(file_path):
            raise FileNotFoundError('output-PlanePoint.npy not found in {}'.format(folder_path))
        file_data = np.load(file_path, allow_pickle=True)
        self.data = np.vstack(file_data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a tuple of input and output tensors given an index
        return self.data[idx][:2], self.data[idx][2:]