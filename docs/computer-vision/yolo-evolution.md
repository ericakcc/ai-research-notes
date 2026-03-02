# YOLO Evolution Comparison

## Architecture Overview

| Version | Year | Backbone | Neck | Head | Anchor | NMS | Key Innovation |
|---------|------|----------|------|------|--------|-----|----------------|
| v1 | 2016 | GoogLeNet-like | None | Coupled | No | Yes | Grid-based single-stage detection |
| v2 | 2017 | Darknet-19 | None | Coupled | Yes | Yes | Anchor boxes, batch norm, multi-scale |
| v3 | 2018 | Darknet-53 | FPN | Coupled | Yes | Yes | Multi-scale prediction, residual blocks |
| v4 | 2020 | CSPDarknet-53 | SPP + PANet | Coupled | Yes | Yes | Bag of Freebies/Specials, Mosaic aug |
| v5 | 2020 | CSPDarknet | PANet | Coupled | Yes | Yes | AutoAnchor, Focus layer, engineering |
| v6 | 2022 | EfficientRep | RepPAN | Decoupled | Yes | Yes | Re-parameterization, industrial focus |
| v7 | 2022 | E-ELAN | ELAN-PANet | Coupled | Yes | Yes | E-ELAN, auxiliary head training |
| v8 | 2023 | CSPDarknet | C2f + PANet | Decoupled | No | Yes | Anchor-free, C2f module |
| v9 | 2024 | GELAN | GELAN | Decoupled | No | Yes | PGI, information bottleneck solution |
| v10 | 2024 | CSPDarknet | - | Decoupled | No | No | NMS-free, consistent dual assignments |
| v11 | 2024 | C3k2 | SPPF + C2PSA | Decoupled | No | Yes | C3k2 block, attention (C2PSA) |
| v12 | 2025 | R-ELAN | A2 + ELAN | Decoupled | No | Yes | Attention-centric, Area Attention, FlashAttention |
| YOLO26 | 2026 | - | - | Decoupled | No | No | NMS-free native, MuSGD, 5-task unified, edge optimized |

## Key Paradigm Shifts

### 1. Anchor-Based → Anchor-Free (v1-v7 → v8+)

- v1-v7: Predefined anchor box shapes (need k-means tuning)
- v8+: Direct center + offset prediction (simpler, more generalizable)

### 2. Coupled → Decoupled Head (v1-v5 → v6+)

- Coupled: Single head for both classification and regression
- Decoupled: Separate branches (classification needs semantic features, regression needs spatial features)

### 3. CNN-Only → Attention Integration (v1-v11 → v12+)

- v1-v11: Pure CNN-based feature extraction
- v12: Attention-centric design with Area Attention (A2) module
- YOLO26: Further refinements for edge deployment

### 4. NMS-Required → NMS-Free (v1-v9 → v10, YOLO26)

- Traditional: Requires NMS post-processing (non-deterministic latency)
- NMS-free: End-to-end detection with consistent dual assignments (deterministic, deployment-friendly)

## Performance Trends (COCO val)

| Version | Size | mAP@50 | mAP@50:95 | Latency (ms) | Params (M) |
|---------|------|--------|-----------|--------------|------------|
| v8-n | 640 | - | - | - | 3.2 |
| v9-t | 640 | - | - | - | - |
| v10-n | 640 | - | - | - | - |
| v11-n | 640 | - | - | - | 2.6 |
| v12-n | 640 | - | - | - | - |
| YOLO26-n | 640 | - | - | - | - |

## Industrial Relevance

| Feature | Application |
|---------|-------------------|
| Multi-scale detection (FPN/PANet) | Defects range from micro-bubbles to large cracks |
| Anchor-free detection | No need to tune anchors for each defect type |
| Decoupled head | Better accuracy for both classification and localization |
| NMS-free inference | Deterministic latency for production line real-time inspection |
| Attention mechanisms | Better feature extraction for subtle defects (scratches, striae) |
| Model quantization support | Edge deployment on NVIDIA Jetson for in-line inspection |
