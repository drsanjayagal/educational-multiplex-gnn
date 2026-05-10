import yaml
import random
import numpy as np
import torch

def set_seed(seed):
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def load_config(path):
with open(path, "r") as f:
return yaml.safe_load(f)
