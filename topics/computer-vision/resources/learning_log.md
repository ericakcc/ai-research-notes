# Learning Progress Log

## Phase 1: Foundations

### YOLOv1 (Redmon et al., 2016)

- [x] First pass: Abstract, Conclusion, Figures
- [x] Second pass: Architecture details (grid, boxes, classes)
- [x] Understand: Loss function design (localization + confidence + classification)
- [ ] Implementation: Grid-based detection head in PyTorch
- [ ] Implementation: IoU calculation + NMS
- [ ] Checkpoint: Detect objects on VOC subset

### YOLOv2 (Redmon & Farhadi, 2017)

- [x] First pass: Improvements over v1
- [x] Understand: Anchor boxes via k-means clustering
- [x] Understand: Batch normalization, multi-scale training
- [ ] Implementation: Add anchor boxes + passthrough layer
- [ ] Checkpoint: Compare v1 vs v2 performance

---

## Phase 2: Maturity

### YOLOv3 (Redmon & Farhadi, 2018)

- [ ] First pass: Architecture overview
- [ ] Understand: Darknet-53 backbone (residual connections)
- [ ] Understand: FPN for 3-scale prediction
- [ ] Implementation: Darknet-53 + FPN in PyTorch

### YOLOv4 (Bochkovskiy et al., 2020)

- [ ] First pass: Bag of Freebies vs Bag of Specials
- [ ] Understand: CSPDarknet-53 structure
- [ ] Understand: PANet (bidirectional feature fusion)
- [ ] Implementation: Mosaic augmentation
- [ ] Implementation: CSP block

### YOLOv5 (Ultralytics, 2020)

- [ ] Review: Ultralytics documentation + architecture blog posts
- [ ] Understand: AutoAnchor, Focus layer, training pipeline
- [ ] Hands-on: Train YOLOv5 on COCO128 as baseline
- [ ] Checkpoint: Darknet-53 + FPN vs YOLOv5 comparison

---

## Phase 3: Industrial

### YOLOv6 (Li et al., 2022)

- [ ] First pass: RepVGG backbone, industrial deployment focus
- [ ] Understand: Re-parameterization (train multi-branch → merge to single)
- [ ] Understand: EfficientRep backbone design

### YOLOv7 (Wang et al., 2022)

- [ ] First pass: E-ELAN and auxiliary head training
- [ ] Understand: Re-parameterizable convolutions
- [ ] Understand: Compound model scaling strategy

### YOLOv8 (Ultralytics, 2023)

- [ ] Review: Ultralytics documentation + architecture analysis
- [ ] Understand: Anchor-free detection, decoupled head
- [ ] Understand: C2f module vs CSP
- [ ] Hands-on: Train YOLOv8 on NEU Surface Defect Dataset
- [ ] Hands-on: Export PyTorch → ONNX → TensorRT
- [ ] Checkpoint: First defect detection experiment complete

---

## Phase 4: Frontier

### YOLOv9 (Wang et al., 2024)

- [ ] First pass: PGI (Programmable Gradient Information)
- [ ] Understand: GELAN architecture design
- [ ] Understand: Information bottleneck problem in deep networks

### YOLOv10 (Wang et al., 2024)

- [ ] First pass: NMS-free end-to-end detection
- [ ] Understand: Consistent dual assignments (one-to-many + one-to-one)
- [ ] Understand: Why NMS-free matters for industrial deployment

### YOLOv11 (Ultralytics, 2024)

- [ ] Review: C3k2 block, SPPF improvements, C2PSA attention
- [ ] Compare: v8 → v11 architectural changes

### YOLOv12 (Tian et al., 2025)

- [ ] First pass: Attention-centric YOLO design
- [ ] Understand: Area Attention (A2) module
- [ ] Understand: R-ELAN (Residual Efficient Layer Aggregation)
- [ ] Understand: FlashAttention integration in detection

### YOLO26 (Ultralytics, 2026)

- [ ] Review: Ultralytics YOLO26 docs + architecture blog
- [ ] Understand: Native NMS-free inference
- [ ] Understand: MuSGD optimizer (SGD + Muon hybrid)
- [ ] Understand: 5-task unified support
- [ ] Hands-on: Run YOLO26 on defect detection tasks
- [ ] Checkpoint: Cross-version benchmark table complete (v8-v12, YOLO26)

---

## Phase 5: Defect Detection

### U-Net (Ronneberger et al., 2015)

- [ ] First pass: Encoder-decoder + skip connections
- [ ] Implementation: U-Net from scratch in PyTorch
- [ ] Hands-on: Train on MVTec AD segmentation masks
- [ ] Checkpoint: Dice coefficient > 0.8 on test set

### PatchCore (Roth et al., 2022)

- [ ] First pass: Memory bank anomaly detection
- [ ] Understand: Coreset subsampling strategy
- [ ] Hands-on: Run PatchCore on MVTec AD (15 categories)
- [ ] Checkpoint: AUROC comparison table

### Glass Defect YOLO (2025)

- [ ] First pass: YOLO for optical lens defect detection
- [ ] Understand: Domain-specific modifications for glass inspection

### Pipeline Integration

- [ ] Build hybrid pipeline: PatchCore screening → YOLO classification
- [ ] Add U-Net segmentation for defect area measurement
- [ ] Implement data augmentation for glass defect simulation
- [ ] Model quantization: FP16/INT8 TensorRT export
- [ ] End-to-end benchmark: throughput (images/sec)
- [ ] Checkpoint: Complete defect detection system demo
