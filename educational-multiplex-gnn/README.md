# Educational Multiplex GNN

**Author:** Dr. Sanjay Agal (Parul University, India)  

Implementation of "A Multiplex Graph Neural Network Framework for Educational Collaboration and Influence Analytics".

## Quick Start

```bash
# Create conda environment
conda env create -f environment.yml
conda activate edu-gnn
pip install -e .

# Generate synthetic data
python scripts/generate_synthetic_data.py --output data/synthetic/

# Train TGAT
python scripts/train_tgat.py --config configs/tgat_default.yaml
See full documentation in the paper.
