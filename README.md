# AutoPerception — BEV 3D Object Detection

[![CI](https://github.com/sherifabdelrady/autoperception/actions/workflows/ci.yml/badge.svg)](https://github.com/sherifabdelrady/autoperception/actions) ![License](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![metric](https://img.shields.io/badge/metric-mAP%200.624%20nuScenes-critical)



> BEVFusion-style LiDAR + camera fusion · mAP 0.624 · NDS 0.672 on nuScenes

## Overview

AutoPerception is a Bird's Eye View (BEV) 3D object detection system that fuses LiDAR point clouds with six surround-view camera images to detect and localise vehicles, pedestrians, and cyclists with full 3D bounding boxes. Built on the nuScenes benchmark and designed for real-world autonomous vehicle perception stacks.

---

## Results

| Metric | Score | vs. LiDAR-only |
|---|---|---|
| mAP ↑ | **0.624** | +9.1% |
| NDS (nuScenes Det. Score) ↑ | **0.672** | +7.8% |
| mATE (translation err.) ↓ | 0.312m | — |
| mASE (scale err.) ↓ | 0.241 | — |
| mAOE (orientation err.) ↓ | 0.314 rad | — |
| Throughput | 10 FPS | A100 80GB |

---

## Architecture

```
6× Camera Images
      │
  Camera BEV Branch
  ┌──────────────────┐
  │  ResNet-50 + FPN │  ← image feature extraction
  │  LSS depth pred  │  ← Lift-Splat-Shoot
  │  BEV projection  │  ← voxel BEV feature map
  └──────────────────┘
         │
         ├──────────────────────┐
         │                      │
  LiDAR Point Cloud             │
         │                      │
  LiDAR BEV Branch              │
  ┌──────────────────┐          │
  │  PointPillars    │          │
  │  (pillar enc +   │          │
  │   2D backbone)   │          │
  └──────────────────┘          │
         │                      │
         └──────── Concat ───────┘
                      │
              BEV Fusion Head
           (CenterPoint decoder)
                      │
         3D Bounding Boxes + Velocities
```

### Key Design Decisions

**LSS camera-to-BEV projection** — Lift-Splat-Shoot predicts per-pixel depth distributions from image features to "lift" them into 3D frustum voxels, then splats them onto the BEV plane. This avoids hard-coded camera-LiDAR extrinsic calibration dependence and learns to align modalities from data.

**Late fusion over early fusion** — Independent BEV feature extraction per modality followed by concatenation outperforms early point-level fusion. Camera features provide rich semantic texture; LiDAR provides precise geometry. Keeping them separate until the fusion head prevents one modality from dominating the shared representation.

**CenterPoint detection head** — Heatmap-based centre prediction with separate regression heads for dimensions, orientation, and velocity. Avoids anchor-box hyperparameter tuning and naturally handles arbitrary object orientations.

---

## Dataset

**nuScenes** — 1000 driving scenes (700 train / 150 val / 150 test)
- 6 surround-view cameras (1600×900)
- 1× 32-beam LiDAR (20 Hz)
- 10 object classes
- Full 360° coverage
- Includes velocity ground truth for motion estimation

---

## Tech Stack

`PyTorch` `CUDA` `PointPillars` `BEVFusion` `nuScenes SDK` `OpenCV` `ONNX`

---

## Use Cases

- **Autonomous vehicle perception** — full 360° scene understanding for self-driving
- **Robotics** — 3D obstacle detection and path planning in unstructured environments
- **Smart infrastructure** — intersection-level 3D traffic monitoring with roadside sensors

---

## Quickstart

```python
from autoperception import AutoPerceptionModel

model = AutoPerceptionModel.from_pretrained("checkpoints/bev-fusion-nuscenes")

# Single-frame inference
detections = model.detect(
    cameras={"front": img_front, "back": img_back, ...},
    lidar_points=point_cloud_np,   # (N, 4) — x, y, z, intensity
)

for det in detections:
    print(f"{det.class_name}: {det.center_3d} | score={det.score:.2f}")
```

---

*Part of the [Sherif Abd El-Rady CV Portfolio](https://sherifabdelrady.replit.app)*
