# SC-DVFNet

**Semantically-Constrained Multi-Scale Deformable Registration Network  
for Bi-Temporal Remote Sensing Image Registration**

Target journals: **IEEE TGRS / ISPRS Journal of Photogrammetry and Remote Sensing**

---

## Overview

Remote sensing platforms continuously acquire large volumes of Earth observation imagery from satellites, aerial systems, and UAVs. Images of the same geographical area are often captured at different times, from different sensors, and under varying viewing conditions. As a result, geometric misalignments frequently occur between images representing the same scene.

**Image registration** aims to align such images within a common coordinate system so that corresponding pixels represent the same physical location on the Earth's surface.

SC-DVFNet is a deep learning framework designed to perform **accurate bi-temporal remote sensing image registration** by combining:

- multi-scale feature extraction
- semantic constraints
- deformable geometric modeling
- end-to-end differentiable warping

The proposed framework estimates **displacement vector fields (DVF)** to model geometric transformations between images.

---

## Repository Structure
```
sc-dvfnet
│
├── README.md
├── figures
│     └── scdvfnet_architecture.png
│
└── models
      ├── sc_dvfnet_swin_t_v5.py
      └── sc_dvfnet_v41_resnet34.py
```

---

## Model Variants

This repository provides **two implementations** of SC-DVFNet.

### 1️⃣ Swin Transformer Version

File: `models/sc_dvfnet_swin_t_v5.py`

Main characteristics:

- Swin Transformer backbone
- hierarchical feature extraction
- Feature Pyramid Network (FPN)
- semantic segmentation branch
- change detection map
- cross-task attention
- multi-head DVF estimation
- uncertainty-aware spatial fusion

This version provides **higher modeling capacity** and better performance on complex remote sensing scenes.

---

### 2️⃣ ResNet34 Version

File: `models/sc_dvfnet_v41_resnet34.py`

Main characteristics:

- Siamese ResNet34 encoder
- multi-scale feature fusion via FPN
- semantic branch with structural attention
- hierarchical DVF composition
- affine → projective → dense transformation refinement
- residual confidence gating mechanism

This version is **lighter and computationally cheaper** compared to the transformer-based model.

---

## Architecture

SC-DVFNet combines several modules:

### Feature Extraction
- Swin Transformer or ResNet34 backbone
- Siamese architecture for bi-temporal images

### Multi-Scale Representation
- Feature Pyramid Network (FPN)

### Semantic Guidance
- segmentation prediction
- change map estimation

### Cross-Task Attention
- semantic features guide geometric alignment

### Multi-Head Transformation Estimation

Three geometric transformation heads:

- **Affine head** (global alignment)
- **Projective head** (perspective distortion)
- **Dense deformation head** (local non-rigid motion)

### Spatial Fusion

Adaptive combination of displacement fields using learned confidence weights.

### Differentiable Warping

A spatial transformer warps the source image using the predicted displacement field.

---

## Architecture Diagram

![SC-DVFNet Architecture](figures/scdvfnet_architecture.png)

> The architecture consists of a siamese feature encoder, a multi-scale feature pyramid, a semantic branch for structural guidance, and multiple displacement field heads that model different geometric transformations.

---

## Installation

Clone the repository:
```bash
git clone https://github.com/Elkalkoli-Mehdi/sc-dvfnet.git
cd sc-dvfnet
```

Install dependencies:
```bash
pip install torch torchvision
```

Optional packages:
```bash
pip install numpy matplotlib tqdm
```

---

## Usage

Run the Swin Transformer model:
```bash
python models/sc_dvfnet_swin_t_v5.py
```

Run the ResNet34 model:
```bash
python models/sc_dvfnet_v41_resnet34.py
```

---

## Citation

If you use this code in your research, please cite:
```bibtex
@article{elkalkoli2026scdvfnet,
  title={SC-DVFNet: Semantically-Constrained Multi-Scale Deformable Registration Network for Bi-Temporal Remote Sensing Image Registration},
  author={Elkalkoli, Mehdi},
  year={2026}
}
```

---

## Author

**Mehdi Elkalkoli**

Research interests:
- Remote sensing
- Computer vision
- Image registration
- Deep learning for geospatial analysis

---

## License

MIT License
