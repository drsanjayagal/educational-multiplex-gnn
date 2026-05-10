#!/usr/bin/env python
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.models.tgat import TGAT
from src.data.multiplex_dataset import TemporalMultiplexDataset

def main():
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--data_dir", default="data/synthetic")
args = parser.parse_args()
with open(args.config, "r") as f:
config = yaml.safe_load(f)
dataset = TemporalMultiplexDataset(args.data_dir, temporal_window=config["data"]["temporal_window"])
loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
model = TGAT(node_feat_dim=16, **config["model"])
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
print("Training started (placeholder - implement full loop)")

Full training loop would go here
torch.save(model.state_dict(), "model_final.pt")

if name == "main":
main()
