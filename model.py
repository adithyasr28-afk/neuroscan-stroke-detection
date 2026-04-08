"""
model.py — Train and save the stroke detection ML model.

Uses:
  - SVM with RBF kernel (strong baseline for image feature classification)
  - StandardScaler for feature normalization
  - Cross-validation for honest performance estimates
  - GridSearchCV for hyperparameter tuning

Run this script once to train and save the model:
    python model.py --dataset path/to/dataset

Expected dataset structure:
    dataset/
      stroke/   <- MRI images labeled as stroke
      normal/   <- MRI images labeled as normal
"""




'''
import os
import argparse
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from features import extract_features

MODEL_PATH = "stroke_model.joblib"


def load_dataset(dataset_path):
    """
    Walk the dataset folder and extract features + labels from all images.
    """
    X, y, paths = [], [], []

    for label_name, label_val in [("normal", 0), ("stroke", 1)]:
        folder = os.path.join(dataset_path, label_name)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]

        print(f"  Loading {len(files)} '{label_name}' images...")

        for fname in files:
            path = os.path.join(folder, fname)
            feats = extract_features(path, is_path=True)

            if feats is None:
                print(f"  [SKIP] Could not process: {path}")
                continue

            X.append(feats)
            y.append(label_val)
            paths.append(path)

    return np.array(X), np.array(y), paths


def train(dataset_path, save_path=MODEL_PATH, tune=False):
    """
    Load data, train SVM model, evaluate with cross-validation, and save.
    """
    print(f"\n=== Loading dataset from: {dataset_path} ===")
    X, y, _ = load_dataset(dataset_path)

    if len(X) == 0:
        print("[ERROR] No images loaded. Check dataset path and structure.")
        return None

    print(f"\nLoaded {len(X)} images | {np.sum(y == 1)} stroke | {np.sum(y == 0)} normal")
    print(f"Feature vector size: {X.shape[1]}")

    # Pipeline: normalize → classify
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced",  # handles imbalanced datasets
                    probability=True))
    ])

    # Cross-validation for honest accuracy estimate
    print("\n=== 5-Fold Cross-Validation ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    print(f"F1 scores per fold: {np.round(scores, 3)}")
    print(f"Mean F1: {scores.mean():.3f} ± {scores.std():.3f}")

    # Optional: hyperparameter tuning
    if tune:
        print("\n=== Hyperparameter Tuning (GridSearchCV) ===")
        param_grid = {
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", "auto", 0.001, 0.01]
        }
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
        grid.fit(X, y)
        print(f"Best params: {grid.best_params_}")
        print(f"Best F1: {grid.best_score_:.3f}")
        pipeline = grid.best_estimator_
    else:
        pipeline.fit(X, y)

    # Final evaluation on full training set (for reference)
    y_pred = pipeline.predict(X)
    print("\n=== Training Set Evaluation ===")
    print(classification_report(y, y_pred, target_names=["Normal", "Stroke"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Save model
    joblib.dump(pipeline, save_path)
    print(f"\nModel saved to: {save_path}")
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stroke detection model")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset folder")
    parser.add_argument("--model", default=MODEL_PATH, help="Output model path")
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV tuning")
    args = parser.parse_args()

    train(args.dataset, save_path=args.model, tune=args.tune)
'''



"""
model.py — Train and save the stroke detection ML model.

Optimised for stroke RECALL — minimising missed strokes.
Uses:
  - SVM with RBF kernel
  - StandardScaler
  - Stroke class weighted 3x (missing a stroke is more costly than a false alarm)
  - Cross-validation scored on stroke recall
  - Optional GridSearchCV tuning

Run:
    python model.py --dataset path/to/dataset
    python model.py --dataset path/to/dataset --tune
"""

import os
import argparse
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, recall_score

from features import extract_features

MODEL_PATH = "stroke_model.joblib"


def load_dataset(dataset_path):
    X, y, paths = [], [], []

    for label_name, label_val in [("normal", 0), ("stroke", 1)]:
        folder = os.path.join(dataset_path, label_name)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]

        print(f"  Loading {len(files)} '{label_name}' images...")

        for fname in files:
            path  = os.path.join(folder, fname)
            feats = extract_features(path, is_path=True)
            if feats is None:
                print(f"  [SKIP] {path}")
                continue
            X.append(feats)
            y.append(label_val)
            paths.append(path)

    return np.array(X), np.array(y), paths


def build_pipeline():
    """
    Voting ensemble of SVM + Random Forest + Logistic Regression.
    All three use class_weight to penalise missed strokes 3x.
    """
    class_w = {0: 1, 1: 3}

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", C=1.0, gamma="scale",
                       class_weight=class_w, probability=True))
    ])
    rf  = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
                       n_estimators=200, class_weight=class_w, random_state=42))
    ])
    lr  = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
                       class_weight=class_w, max_iter=1000, random_state=42))
    ])

    return VotingClassifier(
        estimators=[("svm", svm), ("rf", rf), ("lr", lr)],
        voting="soft"   # average probabilities
    )


def train(dataset_path, save_path=MODEL_PATH, tune=False):
    print(f"\n=== Loading dataset from: {dataset_path} ===")
    X, y, _ = load_dataset(dataset_path)

    if len(X) == 0:
        print("[ERROR] No images loaded.")
        return None

    print(f"\nLoaded {len(X)} images | {np.sum(y==1)} stroke | {np.sum(y==0)} normal")
    print(f"Feature vector size: {X.shape[1]}")

    pipeline      = build_pipeline()
    stroke_recall = make_scorer(recall_score, pos_label=1)
    cv            = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validation scored on STROKE RECALL
    print("\n=== 5-Fold Cross-Validation (scored on Stroke Recall) ===")
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=stroke_recall)
    print(f"Stroke Recall per fold: {np.round(scores, 3)}")
    print(f"Mean Recall: {scores.mean():.3f} ± {scores.std():.3f}")

    if tune:
        print("\n=== Hyperparameter Tuning ===")
        param_grid = {
            "svm__clf__C":     [0.1, 1, 10],
            "svm__clf__gamma": ["scale", "auto"]
        }
        grid = GridSearchCV(pipeline, param_grid, cv=cv,
                            scoring=stroke_recall, n_jobs=-1, verbose=1)
        grid.fit(X, y)
        print(f"Best params: {grid.best_params_}")
        print(f"Best Recall: {grid.best_score_:.3f}")
        pipeline = grid.best_estimator_
    else:
        pipeline.fit(X, y)

    y_pred = pipeline.predict(X)
    print("\n=== Training Set Evaluation ===")
    print(classification_report(y, y_pred, target_names=["Normal", "Stroke"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n  Missed strokes (FN): {cm[1][0]} / {np.sum(y==1)}")

    joblib.dump(pipeline, save_path)
    print(f"\nModel saved to: {save_path}")
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--model",   default=MODEL_PATH)
    parser.add_argument("--tune",    action="store_true")
    args = parser.parse_args()
    train(args.dataset, save_path=args.model, tune=args.tune)