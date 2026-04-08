# NeuroScan — Brain Stroke Detection System

A machine learning pipeline for detecting and classifying brain strokes from MRI scans, with a Flask web interface for real-time inference.

## Features

- **Binary classification** — Stroke vs Normal using an ensemble ML model (SVM + Random Forest + Logistic Regression)
- **Stroke subtype classification** — Ischemic vs Hemorrhagic vs Mixed, based on segmentation masks
- **Image preprocessing** — Grayscale conversion, CLAHE contrast enhancement, median blur denoising
- **Skull stripping** — Isolates brain tissue using Otsu thresholding + largest connected component
- **Multi-feature extraction** — HOG, LBP, intensity statistics, hemisphere asymmetry, gradient features, and region-based features
- **Flask web server** — Upload MRI scan and get annotated result with confidence score
- **Heuristic fallback** — Works without a trained model for development/testing

## Project Structure

```
neuroscan-stroke-detection/
├── preprocess.py       # Image preprocessing (resize, denoise, CLAHE)
├── segmentation.py     # Skull stripping + stroke region segmentation
├── features.py         # Feature extraction (HOG, LBP, symmetry, gradient, etc.)
├── detection.py        # Stroke detection + subtype classification
├── model.py            # Model training (SVM ensemble + cross-validation)
├── main.py             # CLI entry point for training and evaluation
├── server.py           # Flask web server
├── app.py              # App configuration / utilities
├── utils.py            # Helper functions (image loading, normalization)
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates for Flask UI
│   └── index.html      # (add your frontend here)
└── dataset/            # Place your MRI dataset here
    ├── stroke/         # MRI images labelled as stroke
    └── normal/         # MRI images labelled as normal
```

## Prerequisites

- Python 3.8+
- pip

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/neuroscan-stroke-detection.git
cd neuroscan-stroke-detection
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python main.py --train --dataset dataset/
```

### Train with hyperparameter tuning
```bash
python main.py --train --tune --dataset dataset/
```

### Predict on a single image
```bash
python main.py --image path/to/scan.jpg
```

### Evaluate on a labelled dataset
```bash
python main.py --dataset dataset/
```

### Run the web server
```bash
python server.py
# Open http://localhost:5000
```

## ML Pipeline

1. **Preprocessing** — Resize to 128×128, median blur, CLAHE contrast enhancement
2. **Skull stripping** — Otsu threshold + morphological ops + largest connected component
3. **Segmentation** — Separate ischemic (dark) and hemorrhagic (bright) candidate masks using intensity statistics (±1.5 std from mean)
4. **Feature extraction** — HOG + LBP + intensity stats + hemisphere asymmetry + gradient features + stroke-type features + region stats
5. **Classification** — Soft-voting ensemble (SVM + RF + LR), trained with 3× class weight on stroke class to minimize missed detections
6. **Subtype classification** — dark/bright region ratio determines ischemic vs hemorrhagic

## Tech Stack

- Python, OpenCV, scikit-learn, scikit-image, NumPy, Flask
- Models: SVM (RBF), Random Forest, Logistic Regression (soft-voting ensemble)

## Dataset Structure

Place your MRI images in the `dataset/` folder:
```
dataset/
  stroke/   ← MRI scans confirmed as stroke
  normal/   ← Normal MRI scans
```
Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`
