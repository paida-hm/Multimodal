## Project Overview

This repository contains the implementation of a **multimodal contrastive learning framework for molecular representation learning**, integrating:

- **1D SMILES sequences**
- **2D molecular graphs**
- **3D geometric conformations**

The project supports:

1. **Pretraining** on the **QM9** dataset.  
2. **Fine-tuning** on 6 separate benchmark datasets as well as QM9 for property prediction (regression and classification).

The goal is to leverage complementary molecular information across sequential, topological, and geometric levels for improved generalization in molecular property prediction tasks.

## Features
- Multimodal NT-Xent contrastive learning
- Integration of **SMILES-BERT** (1D), **PNA** (2D), and **3D models**
- Support for GPU training
- Pretrained checkpoints and training logs (via Git LFS or external links)
Due to platform upload limitations, the 1D SMILES encoder (SMILES-BERT) must be downloaded separately.  
Download it from HuggingFace:  
[https://huggingface.co/unikei/bert-base-smiles](https://huggingface.co/unikei/bert-base-smiles)

## Dependencies

- Python >= 3.8  
- PyTorch >= 2.0  
- RDKit  
- HuggingFace Transformers  
- Other dependencies listed in `requirements.txt`

## Installation
```bash
# Clone repository
git clone git@github.com:paida-hm/Multimodal_Code.git
cd Multimodal_Code

# Install dependencies
conda create -n multimodal python=3.7
conda activate multimodal
pip install -r requirements.txt

# Install Git LFS for large files
git lfs install
