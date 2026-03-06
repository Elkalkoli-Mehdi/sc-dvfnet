# SC-DVFNet

Semantically-Constrained Multi-Scale Deformable Registration Network  
for Bi-Temporal Remote Sensing Image Registration.

Target journals: **IEEE TGRS / ISPRS JPRS**

---

## Architecture

SC-DVFNet combines:

- Transformer or CNN backbone (Swin-T / ResNet34)
- Feature Pyramid Network (FPN)
- Semantic segmentation branch
- Linear cross-task attention
- Multi-head DVF estimation (Affine, Projective, Dense)
- Change-aware registration
- Spatial transformer warping

---

## Installation

```bash
pip install torch torchvision
