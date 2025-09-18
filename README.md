# Structure-Aware Contrastive Learning with Fine-Grained Binding Representations for Drug Discovery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

We present SaBAN-DTI, a sequence-based Drug-Target Interaction (DTI) framework that injects structural signals while preserving screening speed and accuracy. Our approach represents proteins using a structure-aware vocabulary that pairs each residue token with a compact descriptor of local geometry. By pretraining a large protein language model on sequences annotated with these descriptors, our encoder learns structural context while operating on plain sequence inputs. Small molecules are encoded with SELFIES to guarantee validity and preserve chemical semantics. Both encoders employ attention-based aggregation that maintains input resolution and produces interpretable importance maps over binding-relevant regions. A contrastive learning objective aligns drug and protein embeddings by drawing together regions corresponding to true binding interfaces.

## Key Contributions

1. **Structure-aware protein sequence representation** that augments each residue token with a compact local-geometry descriptor, enabling structural context learning from plain sequences
2. **Attention-based aggregation module** that preserves resolution, focuses on binding-relevant regions, and produces interpretable importance maps
3. **High-performance DTI prediction framework** that outperforms existing baselines in both accuracy and speed, supporting large-scale virtual screening for drug discovery

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.7+ (for GPU acceleration)
- PyTorch 2.0+

### Download Pretrained Models

Download pretrained protein encoder (SaProt):
```bash
wget https://huggingface.co/westlake-repl/SaProt_650M_AF2/resolve/main/pytorch_model.bin
```

Download Foldseek binary for structure encoding:
```bash
wget https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view?usp=sharing
chmod +x foldseek
```

## Usage

### Training

Train the model with default parameters:
```bash
python train.py \
    --data_path dataset/your_dataset.csv \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4
```

For k-fold cross-validation:
```bash
python train.py \
    --data_path dataset/your_dataset.csv \
    --k_folds 5 \
    --batch_size 32 \
    --epochs 100
```

### Evaluation

Evaluate on DUDE and LIT-PCBA benchmarks:
```bash
python evaluate.py \
    --checkpoint checkpoint_best.pth \
    --dataset both \
    --data-path dataset/ \
    --output-dir test_results/
```

### Inference

For single prediction:
```python
from model import DTIModel
from transformers import EsmTokenizer, AutoTokenizer
import torch

# Load model
model = DTIModel()
model.load_state_dict(torch.load('checkpoint_best.pth'))
model.eval()

# Load tokenizers
prot_tokenizer = EsmTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
drug_tokenizer = AutoTokenizer.from_pretrained("HUBioDataLab/SELFormer")

# Prepare inputs
protein_sequence = "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASE..."
drug_selfies = "[C][C][=C][C][=C][C][=C][Ring1][=Branch1]"

# Predict
with torch.no_grad():
    score = model(protein_sequence, drug_selfies)
    probability = torch.sigmoid(score).item()
```

## Dataset Format

### Input CSV Format

The input CSV should contain the following columns:
- `Protein`: Raw protein sequence
- `Seq`: Structure-aware protein sequence (from SaProt)
- `SMILES`: Drug SMILES notation
- `selfies`: Drug SELFIES notation
- `label`: Binary interaction label (0 or 1)

Example:
```csv
Protein,Seq,SMILES,selfies,label
MGLSDGEWQLVLNVWGK...,MpGpLpSpDpGpEpWpQp...,CC(=O)Oc1ccccc1C(=O)O,[C][C][=Branch1][C][=O]...,1
```

### Structure-Aware Sequence Generation

To generate structure-aware sequences from PDB/CIF files:

```python
from saprot import get_structure_aware_sequence

# Load structure file
structure_file = "protein.cif"  # or .pdb

# Generate structure-aware sequence
sa_sequence = get_structure_aware_sequence(structure_file, foldseek_path="./foldseek")
```

## Model Architecture

### Components

1. **Protein Encoder**: SaProt-based transformer with learned aggregation layer
2. **Drug Encoder**: SELFormer-based transformer with attention pooling  
3. **Interaction Predictor**: Bilinear attention network (BAN) with multi-head attention
4. **Contrastive Learning**: CLIP-based alignment of drug-protein embeddings

### Key Parameters

- Protein embedding dimension: 1280
- Drug embedding dimension: 768
- Latent dimension: 1024
- Number of attention heads: 8
- Dropout rate: 0.05

## Repository Structure

```
SaBAN-DTI/
├── model.py           # Core DTI model implementation
├── modules.py         # Core modules
├── dataset.py         # Data processing and loading
├── train.py          # Training script
├── evaluate.py       # Evaluation metrics and benchmarking
├── test_cp3a4.py     # Specific target testing for DrugCLIP
└── README.md         # This file
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

We thank the authors of SaProt, SELFormer, and SELFIES for making their models and code publicly available.
