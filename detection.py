"""
detection.py — Stroke detection using the trained ML model.

Falls back to a rule-based heuristic if no model file is found,
so the pipeline still works during development/testing.
"""
'''
import os
import joblib
import numpy as np

MODEL_PATH = "stroke_model.joblib"

# Cache the loaded model in module scope (avoid re-loading on every call)
_model = None


def load_model(path=MODEL_PATH):
    global _model
    if _model is None:
        if not os.path.exists(path):
            return None
        _model = joblib.load(path)
        print(f"[Model] Loaded from {path}")
    return _model


def detect_stroke(features, threshold=0.5):
    """
    Predict stroke from an extracted feature vector.

    Parameters:
        features (np.ndarray): 1D feature vector from features.extract_features()
        threshold (float): probability cutoff (default 0.5)

    Returns:
        dict with keys:
            - "stroke" (bool): True if stroke detected
            - "confidence" (float): probability of stroke class
            - "method" (str): "ml" or "heuristic"
    """
    model = load_model()

    if model is not None:
        # ML prediction
        feats_2d = features.reshape(1, -1)
        proba = model.predict_proba(feats_2d)[0]
        stroke_prob = proba[1]  # probability of class "stroke" (label 1)

        return {
            "stroke": stroke_prob >= threshold,
            "confidence": float(stroke_prob),
            "method": "ml"
        }

    else:
        # Fallback: rule-based heuristic using region and symmetry features
        # Features are ordered: [hog..., lbp..., intensity(8), symmetry(2), region(6)]
        # Symmetry features are at positions [-8] and [-7], region at [-6:]
        asymmetry = features[-8]        # mean asymmetry
        num_regions = features[-6]      # number of suspect regions
        max_area = features[-4]         # largest suspect region
        mean_intensity = features[-3]   # mean intensity in regions

        score = 0.0
        if asymmetry > 15:
            score += 0.4
        if num_regions > 3:
            score += 0.3
        if max_area > 200:
            score += 0.2
        if mean_intensity < 60:
            score += 0.1

        return {
            "stroke": score >= 0.5,
            "confidence": score,
            "method": "heuristic"
        }
'''


"""
detection.py — Stroke detection + subtype classification.

Two-stage pipeline:
  Stage 1: Binary  — Stroke vs Normal        (ML model)
  Stage 2: Subtype — Ischemic vs Hemorrhagic (segmentation-based)

Falls back to heuristic if no trained model is found.
"""

import os
import joblib
import numpy as np

MODEL_PATH = "stroke_model.joblib"
_model     = None


def load_model(path=MODEL_PATH):
    global _model
    if _model is None:
        if not os.path.exists(path):
            return None
        _model = joblib.load(path)
        print(f"[Model] Loaded from {path}")
    return _model


def detect_stroke(features, dark_mask=None, bright_mask=None, threshold=0.35):
    """
    Predict stroke and determine subtype.

    Parameters:
        features    (np.ndarray): feature vector from features.extract_features()
        dark_mask   (np.ndarray): ischemic candidate mask from segment_stroke()
        bright_mask (np.ndarray): hemorrhagic candidate mask from segment_stroke()
        threshold   (float): confidence cutoff — lowered to 0.35 to reduce missed strokes

    Returns dict with keys:
        stroke       (bool)  — True if stroke detected
        confidence   (float) — model's stroke probability
        method       (str)   — "ml" or "heuristic"
        stroke_type  (str)   — "ischemic" | "hemorrhagic" | "mixed" | "normal"
        dark_area    (int)   — pixel count of ischemic regions
        bright_area  (int)   — pixel count of hemorrhagic regions
        type_confidence (str) — how confident the subtype call is
    """
    model    = load_model()
    is_stroke = False
    confidence = 0.0
    method     = "heuristic"

    # ── Stage 1: Binary classification ───────────────────────────────────
    if model is not None:
        feats_2d   = features.reshape(1, -1)
        proba      = model.predict_proba(feats_2d)[0]
        confidence = float(proba[1])
        is_stroke  = confidence >= threshold
        method     = "ml"
    else:
        # Heuristic fallback using feature offsets
        # Feature order tail: [..., gradient(5), type(6), region(6)]
        # type features: dark_area, bright_area, dark_mean, bright_mean, dark_frac, bright_frac
        dark_frac   = features[-8]   # dark_fraction
        bright_frac = features[-7]   # bright_fraction
        asymmetry   = features[-14]  # symmetry mean (before gradient block)
        num_regions = features[-6]

        score = 0.0
        if asymmetry   > 15:  score += 0.35
        if dark_frac   > 0.1: score += 0.25
        if bright_frac > 0.1: score += 0.25
        if num_regions > 2:   score += 0.15

        confidence = min(score, 1.0)
        is_stroke  = confidence >= threshold

    # ── Stage 2: Subtype classification ──────────────────────────────────
    stroke_type     = "normal"
    dark_area       = 0
    bright_area     = 0
    type_confidence = "N/A"

    if is_stroke:
        if dark_mask is not None and bright_mask is not None:
            # Use segmentation masks directly — most accurate
            from segmentation import classify_stroke_type
            stroke_type, dark_area, bright_area = classify_stroke_type(
                dark_mask, bright_mask
            )
        else:
            # Fall back to feature vector values (indices -11 and -10 in type block)
            # type_feats order: dark_area, bright_area, dark_mean, bright_mean, dark_frac, bright_frac
            # region_feats(6) at end → type_feats start at -12
            dark_area_feat   = float(features[-12])
            bright_area_feat = float(features[-11])
            total = dark_area_feat + bright_area_feat

            if total < 50:
                stroke_type = "normal"
            else:
                ratio = bright_area_feat / total
                if ratio > 0.65:
                    stroke_type = "hemorrhagic"
                elif ratio < 0.35:
                    stroke_type = "ischemic"
                else:
                    stroke_type = "mixed"

            dark_area   = int(dark_area_feat)
            bright_area = int(bright_area_feat)

        # Confidence label for subtype
        total = dark_area + bright_area
        if total > 0:
            dominant_ratio = max(dark_area, bright_area) / total
            if dominant_ratio > 0.80:
                type_confidence = "High"
            elif dominant_ratio > 0.65:
                type_confidence = "Moderate"
            else:
                type_confidence = "Low (mixed)"

    return {
        "stroke":          is_stroke,
        "confidence":      confidence,
        "method":          method,
        "stroke_type":     stroke_type,
        "dark_area":       dark_area,
        "bright_area":     bright_area,
        "type_confidence": type_confidence
    }