#!/usr/bin/env python
import argparse
import yaml
import subprocess
import os

def main():
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
with open(args.config, "r") as f:
config = yaml.safe_load(f)
print("Running full pipeline with config:", config)

Generate synthetic data if needed
if config["data"].get("synthetic", False):
subprocess.run(["python", "scripts/generate_synthetic_data.py", "--output", "data/synthetic", "--num_nodes", str(config["data"]["synthetic_params"]["num_nodes"]), "--num_semesters", str(config["data"]["synthetic_params"]["num_semesters"]), "--seed", str(config.get("seed", 42))])

Train model
subprocess.run(["python", "scripts/train_tgat.py", "--config", args.config, "--data_dir", "data/synthetic"])
print("Pipeline completed.")

if name == "main":
main()
