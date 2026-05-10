import torch
import numpy as np
import pickle
from torch.utils.data import Dataset

class TemporalMultiplexDataset(Dataset):
def init(self, data_dir, temporal_window=4, edge_types=None):
self.data_dir = data_dir
self.temporal_window = temporal_window
self.edge_types = edge_types or ['coauthorship', 'supervision', 'project', 'course']
with open(f"{data_dir}/snapshots.pkl", "rb") as f:
self.snapshots = pickle.load(f)
self.samples = [(self.snapshots[t-self.temporal_window:t], self.snapshots[t])
for t in range(self.temporal_window, len(self.snapshots))]
def len(self):
return len(self.samples)
def getitem(self, idx):
return self.samples[idx]
