from setuptools import setup, find_packages
setup(
name="educational_multiplex_gnn",
version="1.0.0",
author="Dr. Sanjay Agal",
description="Temporal multiplex GNN for educational collaboration analytics",
packages=find_packages(where="src"),
package_dir={"": "src"},
install_requires=["torch>=2.1.0", "torch-geometric>=2.3.0", "numpy>=1.24.0", "scikit-learn>=1.3.0", "pyyaml>=6.0", "tqdm>=4.65.0"],
python_requires=">=3.10",
)
