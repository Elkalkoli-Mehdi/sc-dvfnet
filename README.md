# SC-DVFNet 

Semantically-Constrained Multi-Scale Deformable Registration Network  
for Bi-Temporal Remote Sensing Image Registration.

Target journals: IEEE TGRS / ISPRS JPRS

## Architecture

SC-DVFNet combines:

- Swin-T Transformer backbone
- Feature Pyramid Network
- Semantic segmentation branch
- Linear cross-task attention
- Multi-head DVF estimation (Affine, Projective, Dense)
- Uncertainty-aware spatial fusion

## Installation

```bash
pip install torch torchvision
