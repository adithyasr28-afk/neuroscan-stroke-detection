"""
main.py — Stroke detection pipeline entry point.

USAGE:
  # Predict on a single image (uses pre-trained model if available):
  python main.py --image path/to/scan.jpg

  # Evaluate on a labelled dataset:
  python main.py --dataset path/to/dataset

  # Train the model first, then evaluate:
  python main.py --train --dataset path/to/dataset

  # Train + tune hyperparameters:
  python main.py --train --tune --dataset path/to/dataset

DATASET STRUCTURE:
  dataset/
    stroke/   <- MRI scans with stroke
    normal/   <- Normal MRI scans
"""

import os
import argparse
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from preprocess import preprocess_image
from segmentation import skull_strip, segment_stroke
from features import extract_features
from detection import detect_stroke, load_model


# ─────────────────────────────────────────────
# Single-image prediction
# ─────────────────────────────────────────────

def process_image(path, show=False):
    """
    Full pipeline for a single image.
    Returns a result dict, or None if the image cannot be loaded.
    """
    raw = cv2.imread(path)
    if raw is None:
        print(f"[ERROR] Cannot load image: {path}")
        return None

    feats = extract_features(raw, is_path=False)
    if feats is None:
        print(f"[ERROR] Feature extraction failed: {path}")
        return None

    result = detect_stroke(feats)

    label = "STROKE" if result["stroke"] else "NORMAL"
    method = result["method"].upper()
    conf = result["confidence"]
    print(f"{os.path.basename(path):40s} → {label}  (conf={conf:.2f}, method={method})")

    if show:
        _visualize(raw, feats, result)

    return result


def _visualize(raw, feats, result):
    """
    Draw bounding boxes around suspect regions and show prediction.
    """
    processed = preprocess_image(raw)
    brain = skull_strip(processed)
    mask = segment_stroke(brain)

    import cv2 as _cv2
    contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)

    output = raw.copy()
    color = (0, 0, 255) if result["stroke"] else (0, 200, 0)

    for cnt in contours:
        if _cv2.contourArea(cnt) > 80:
            x, y, w, h = _cv2.boundingRect(cnt)
            _cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    label = f"{'STROKE' if result['stroke'] else 'NORMAL'} ({result['confidence']:.0%})"
    _cv2.putText(output, label, (10, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    _cv2.imshow("Stroke Detection", output)
    _cv2.waitKey(0)
    _cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Full dataset evaluation
# ─────────────────────────────────────────────

def evaluate_dataset(dataset_path):
    """
    Run prediction on a labelled dataset and print metrics.
    """
    y_true, y_pred, y_conf = [], [], []

    for label_name, label_val in [("normal", 0), ("stroke", 1)]:
        folder = os.path.join(dataset_path, label_name)
        if not os.path.isdir(folder):
            print(f"[WARN] Missing folder: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]

        for fname in files:
            path = os.path.join(folder, fname)
            result = process_image(path)

            if result is None:
                continue

            y_true.append(label_val)
            y_pred.append(int(result["stroke"]))
            y_conf.append(result["confidence"])

    if not y_true:
        print("[ERROR] No images processed.")
        return

    print("\n" + "=" * 50)
    print("DATASET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total images: {len(y_true)}")
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"Accuracy:     {accuracy:.2%}")

    try:
        auc = roc_auc_score(y_true, y_conf)
        print(f"ROC-AUC:      {auc:.3f}")
    except Exception:
        pass

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Stroke"]))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("              Normal  Stroke")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Normal       {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"  Stroke       {cm[1][0]:4d}    {cm[1][1]:4d}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Stroke Detection")
    parser.add_argument("--image", help="Path to a single MRI image to predict")
    parser.add_argument("--dataset", default="dataset", help="Path to labelled dataset folder")
    parser.add_argument("--train", action="store_true", help="Train ML model before evaluating")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning during training")
    parser.add_argument("--show", action="store_true", help="Show visualisation window")
    args = parser.parse_args()

    # 1. Train if requested
    if args.train:
        from model import train
        print("\n=== TRAINING MODE ===")
        train(args.dataset, tune=args.tune)

    # 2. Single image prediction
    if args.image:
        print("\n=== SINGLE IMAGE PREDICTION ===")
        result = process_image(args.image, show=args.show)
        if result:
            print(f"\nResult: {'STROKE DETECTED' if result['stroke'] else 'NO STROKE'}")
            print(f"Confidence: {result['confidence']:.1%}  |  Method: {result['method']}")

    # 3. Dataset evaluation (only if not just training, or dataset explicitly given)
    elif not args.train or args.dataset:
        print("\n=== DATASET EVALUATION ===")
        evaluate_dataset(args.dataset)
