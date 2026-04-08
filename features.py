'''import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.measure import shannon_entropy
from preprocess import preprocess_image
from segmentation import skull_strip, segment_stroke, extract_region_features


def extract_hog_features(img):
    """
    HOG (Histogram of Oriented Gradients) — captures edge/shape patterns
    commonly used in medical image classification.
    """
    features, _ = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=None
    )
    return features


def extract_lbp_features(img, num_points=24, radius=3):
    """
    Local Binary Pattern — captures texture patterns.
    """
    lbp = local_binary_pattern(img, num_points, radius, method="uniform")
    n_bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_intensity_features(img, brain):
    """
    Global and regional intensity statistics from the brain region.
    """
    brain_pixels = brain[brain > 0]

    if len(brain_pixels) == 0:
        return np.zeros(8)

    return np.array([
        np.mean(brain_pixels),
        np.std(brain_pixels),
        np.median(brain_pixels),
        np.percentile(brain_pixels, 25),
        np.percentile(brain_pixels, 75),
        float(np.sum(brain_pixels < 50)) / len(brain_pixels),   # dark fraction
        float(np.sum(brain_pixels > 200)) / len(brain_pixels),  # bright fraction
        shannon_entropy(brain_pixels)
    ])


def extract_symmetry_features(brain):
    """
    Asymmetry between left and right brain hemispheres.
    Strokes often cause asymmetry in intensity distribution.
    """
    h, w = brain.shape
    left = brain[:, :w // 2].astype(np.float32)
    right = brain[:, w // 2:].astype(np.float32)
    right_flipped = np.fliplr(right)

    min_w = min(left.shape[1], right_flipped.shape[1])
    left = left[:, :min_w]
    right_flipped = right_flipped[:, :min_w]

    diff = np.abs(left - right_flipped)
    asymmetry = np.mean(diff)
    asymmetry_std = np.std(diff)

    return np.array([asymmetry, asymmetry_std])


def extract_features(img_path_or_array, is_path=True):
    """
    Main feature extraction pipeline.
    Returns a 1D numpy array of combined features for ML classification.
    
    Parameters:
        img_path_or_array: file path (str) or already-loaded numpy image
        is_path: True if first argument is a file path
    """
    if is_path:
        raw = cv2.imread(img_path_or_array)
        if raw is None:
            return None
    else:
        raw = img_path_or_array

    # Preprocessing
    processed = preprocess_image(raw)
    if processed is None:
        return None

    brain = skull_strip(processed)
    mask = segment_stroke(brain)
    region_stats = extract_region_features(brain, mask)

    # Feature groups
    hog_feats = extract_hog_features(processed)       # shape/edge features
    lbp_feats = extract_lbp_features(processed)       # texture features
    intensity_feats = extract_intensity_features(processed, brain)  # intensity stats
    symmetry_feats = extract_symmetry_features(brain) # asymmetry (stroke indicator)

    # Region-based features (as array)
    region_feats = np.array([
        region_stats["num_regions"],
        region_stats["total_area"],
        region_stats["max_area"],
        region_stats["mean_intensity"],
        region_stats["min_intensity"],
        region_stats["mean_aspect_ratio"]
    ])

    # Combine all features
    all_features = np.concatenate([
        hog_feats,
        lbp_feats,
        intensity_feats,
        symmetry_feats,
        region_feats
    ])

    return all_features
'''

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.measure import shannon_entropy
from preprocess import preprocess_image
from segmentation import skull_strip, segment_stroke, extract_region_features


def extract_hog_features(img):
    """HOG — captures edge/shape patterns."""
    features, _ = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=None
    )
    return features


def extract_lbp_features(img, num_points=24, radius=3):
    """LBP — captures texture patterns."""
    lbp    = local_binary_pattern(img, num_points, radius, method="uniform")
    n_bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_intensity_features(img, brain):
    """Global intensity statistics from the brain region."""
    brain_pixels = brain[brain > 0]
    if len(brain_pixels) == 0:
        return np.zeros(8)
    return np.array([
        np.mean(brain_pixels),
        np.std(brain_pixels),
        np.median(brain_pixels),
        np.percentile(brain_pixels, 25),
        np.percentile(brain_pixels, 75),
        float(np.sum(brain_pixels < 50))  / len(brain_pixels),   # dark fraction
        float(np.sum(brain_pixels > 200)) / len(brain_pixels),   # bright fraction
        shannon_entropy(brain_pixels)
    ])


def extract_symmetry_features(brain):
    """
    Hemisphere asymmetry — strokes disrupt normal left/right symmetry.
    """
    h, w = brain.shape
    left          = brain[:, :w // 2].astype(np.float32)
    right         = brain[:, w // 2:].astype(np.float32)
    right_flipped = np.fliplr(right)

    min_w         = min(left.shape[1], right_flipped.shape[1])
    diff          = np.abs(left[:, :min_w] - right_flipped[:, :min_w])

    return np.array([np.mean(diff), np.std(diff)])


def extract_gradient_features(img):
    """
    Gradient magnitude — ischemic regions are abnormally smooth (low gradient),
    hemorrhagic regions have sharp high-gradient boundaries.
    """
    grad_x    = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y    = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return np.array([
        np.mean(magnitude),
        np.std(magnitude),
        np.percentile(magnitude, 10),                            # low gradient  → ischemic
        np.percentile(magnitude, 90),                            # high gradient → hemorrhagic
        float(np.sum(magnitude < 5)) / magnitude.size           # fraction of very smooth regions
    ])


def extract_stroke_type_features(brain, dark_mask, bright_mask):
    """
    Features that directly encode the ischemic vs hemorrhagic signal.
    These are used both for the binary stroke/normal classifier AND
    for the subtype classifier.

    Returns 6 features:
        dark_pixel_count    — size of ischemic candidate region
        bright_pixel_count  — size of hemorrhagic candidate region
        dark_mean_intensity — how dark the dark region is
        bright_mean_intensity — how bright the bright region is
        dark_fraction       — dark pixels as fraction of brain
        bright_fraction     — bright pixels as fraction of brain
    """
    brain_area  = max(np.count_nonzero(brain), 1)
    dark_area   = int(np.sum(dark_mask   > 0))
    bright_area = int(np.sum(bright_mask > 0))

    dark_pixels   = brain[dark_mask   > 0]
    bright_pixels = brain[bright_mask > 0]

    dark_mean   = float(np.mean(dark_pixels))   if len(dark_pixels)   > 0 else 0.0
    bright_mean = float(np.mean(bright_pixels)) if len(bright_pixels) > 0 else 0.0

    return np.array([
        dark_area,
        bright_area,
        dark_mean,
        bright_mean,
        dark_area   / brain_area,
        bright_area / brain_area
    ])


def extract_features(img_path_or_array, is_path=True):
    """
    Full feature extraction pipeline.
    Returns a 1D numpy array for ML classification.
    """
    if is_path:
        raw = cv2.imread(img_path_or_array)
        if raw is None:
            return None
    else:
        raw = img_path_or_array

    processed = preprocess_image(raw)
    if processed is None:
        return None

    brain                          = skull_strip(processed)
    combined, dark_mask, bright_mask = segment_stroke(brain)
    region_stats                   = extract_region_features(brain, combined)

    hog_feats        = extract_hog_features(processed)
    lbp_feats        = extract_lbp_features(processed)
    intensity_feats  = extract_intensity_features(processed, brain)
    symmetry_feats   = extract_symmetry_features(brain)
    gradient_feats   = extract_gradient_features(processed)
    type_feats       = extract_stroke_type_features(brain, dark_mask, bright_mask)

    region_feats = np.array([
        region_stats["num_regions"],
        region_stats["total_area"],
        region_stats["max_area"],
        region_stats["mean_intensity"],
        region_stats["min_intensity"],
        region_stats["mean_aspect_ratio"]
    ])

    return np.concatenate([
        hog_feats,        # shape/edge
        lbp_feats,        # texture
        intensity_feats,  # intensity statistics
        symmetry_feats,   # hemisphere asymmetry
        gradient_feats,   # gradient (ischemic vs hemorrhagic signal)
        type_feats,       # stroke type–specific features  ← new
        region_feats      # region properties
    ])