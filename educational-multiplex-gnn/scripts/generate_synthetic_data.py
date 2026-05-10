#!/usr/bin/env python
import argparse
import numpy as np
import networkx as nx
import pickle
import os
import csv
from tqdm import tqdm

def generate_smallworld(n, density, p=0.1):
m = int(n * density / 2)
return nx.watts_strogatz_graph(n, m, p)

def main():
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="data/synthetic")
parser.add_argument("--num_nodes", type=int, default=378)
parser.add_argument("--num_semesters", type=int, default=12)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
np.random.seed(args.seed)
os.makedirs(args.output, exist_ok=True)
snapshots = []
base_density = 0.048
for t in tqdm(range(args.num_semesters)):
density = max(0.01, base_density + 0.005 * np.random.randn())
G = generate_smallworld(args.num_nodes, density)
adj = nx.to_numpy_array(G)
node_feat = np.random.randn(args.num_nodes, 16)
snapshots.append({"adjacency": adj, "node_features": node_feat, "time": t})
with open(os.path.join(args.output, "snapshots.pkl"), "wb") as f:
pickle.dump(snapshots, f)
print(f"Saved {args.num_semesters} snapshots to {args.output}")

if name == "main":
main()
