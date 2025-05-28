# 3D-Image-Segmentation-of-Liver-Tumor

This repository contains a fully explainable and modular pipeline for liver tumor segmentation in 3D CT scans using classical image processing techniques. The project avoids the use of neural networks, focusing instead on intensity-based region growing, morphological operations, and anatomical constraints.

## Project Structure

```
.
├── 0745/                       # Contains the full CT scan data and ground truth segmentations in DICOM format
├── attempts/                   # Folder containing various segmentation attempts and experimental runs
├── Assignment.ipynb           # Main Jupyter notebook with the full segmentation pipeline, visualizations, and evaluation
├── utils.py                   # Utility functions for plotting, filtering, and morphological operations
├── mip_liver_tumor.gif        # Animated 3D rotation showing liver and tumor masks
├── overlay_segmentation.png   # Overlay of predicted vs ground truth segmentation masks (MIP view)
├── overlay_tumor_masks.png    # Slice-wise overlay of predicted vs ground truth tumor masks
└── README.md
```

## How to Use

1. Open `Assignment.ipynb` to view and execute the segmentation pipeline.
2. Use `utils.py` for helper functions related to filtering, plotting, and morphological operations.
3. Refer to `overlay_segmentation.png` and `overlay_tumor_masks.png` for visual comparison between predicted and true masks.
4. The `mip_liver_tumor.gif` provides a 3D visual rotation of liver and tumor masks.

## Highlights

- No deep learning required — fully classical, interpretable pipeline.
- Region-wise tumor processing for improved control and accuracy.
- Evaluation includes voxel-level and tumor-wise metrics.
- Visualizations and metrics all generated from real clinical DICOM data.

---

*Developed as part of the Medical Image Processing (MIP) course.*
