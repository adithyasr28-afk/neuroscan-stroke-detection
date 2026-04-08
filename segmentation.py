'''
import cv2
import numpy as np


def skull_strip(img):
    """
    Remove skull/background to isolate brain tissue.
    Uses Otsu thresholding + morphology + largest connected component.
    """
    # Otsu thresholding — auto-selects best threshold
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill holes with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Keep only the largest connected component (the brain)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

    if num_labels < 2:
        # Fallback: return the image as-is if no components found
        return img

    largest_label = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    brain_mask = (labels == largest_label).astype("uint8") * 255

    # Fill any internal holes in the brain mask
    floodfill = brain_mask.copy()
    h, w = brain_mask.shape
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(floodfill, mask_ff, (0, 0), 255)
    floodfill_inv = cv2.bitwise_not(floodfill)
    brain_mask = brain_mask | floodfill_inv

    brain = cv2.bitwise_and(img, img, mask=brain_mask)
    return brain


''def segment_stroke(brain):
    """
    Detect potential stroke/lesion regions using adaptive thresholding
    and morphological cleanup.
    """
    if np.count_nonzero(brain) == 0:
        return np.zeros_like(brain)

    # Adaptive threshold on non-zero brain pixels
    seg = cv2.adaptiveThreshold(
        brain, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=3
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    return cleaned

def segment_stroke(brain):
    """
    Detect stroke/lesion regions only.
    Strokes appear as abnormally dark (ischemic) or bright (hemorrhagic)
    regions compared to surrounding normal tissue.
    """
    if np.count_nonzero(brain) == 0:
        return np.zeros_like(brain)

    # Only work within the brain region
    brain_pixels = brain[brain > 0]
    mean_val = np.mean(brain_pixels)
    std_val  = np.std(brain_pixels)

    # Ischemic stroke → unusually dark patches (below mean - 1.5 std)
    dark_thresh  = max(0,   int(mean_val - 1.5 * std_val))
    # Hemorrhagic stroke → unusually bright patches (above mean + 1.5 std)
    bright_thresh = min(255, int(mean_val + 1.5 * std_val))

    dark_mask   = cv2.inRange(brain, 1, dark_thresh)
    bright_mask = cv2.inRange(brain, bright_thresh, 255)
    combined    = cv2.bitwise_or(dark_mask, bright_mask)

    # Remove noise — small isolated pixels are not lesions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=2)
    closed = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed
def extract_region_features(brain, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    region_areas        = []
    region_intensities  = []
    region_aspect_ratios = []

    brain_area = np.count_nonzero(brain)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Skip tiny noise blobs AND blobs that cover >40% of the brain
        # (those are background leakage, not lesions)
        if area < 100 or (brain_area > 0 and area / brain_area > 0.4):
            continue

        region_mask = np.zeros_like(mask)
        cv2.drawContours(region_mask, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(brain, mask=region_mask)[0]

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 1.0

        region_areas.append(area)
        region_intensities.append(mean_intensity)
        region_aspect_ratios.append(aspect_ratio)

    if not region_areas:
        return {
            "num_regions": 0, "total_area": 0, "max_area": 0,
            "mean_intensity": 0, "min_intensity": 0, "mean_aspect_ratio": 1.0
        }

    return {
        "num_regions":      len(region_areas),
        "total_area":       sum(region_areas),
        "max_area":         max(region_areas),
        "mean_intensity":   np.mean(region_intensities),
        "min_intensity":    np.min(region_intensities),
        "mean_aspect_ratio": np.mean(region_aspect_ratios)
    }
'''
'''def extract_region_features(brain, mask):
    """
    Extract features from detected regions for use in the ML pipeline.
    Returns a dict of per-image region statistics.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    region_areas = []
    region_intensities = []
    region_aspect_ratios = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue

        region_mask = np.zeros_like(mask)
        cv2.drawContours(region_mask, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(brain, mask=region_mask)[0]

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 1.0

        region_areas.append(area)
        region_intensities.append(mean_intensity)
        region_aspect_ratios.append(aspect_ratio)

    if not region_areas:
        return {
            "num_regions": 0,
            "total_area": 0,
            "max_area": 0,
            "mean_intensity": 0,
            "min_intensity": 0,
            "mean_aspect_ratio": 1.0
        }

    return {
        "num_regions": len(region_areas),
        "total_area": sum(region_areas),
        "max_area": max(region_areas),
        "mean_intensity": np.mean(region_intensities),
        "min_intensity": np.min(region_intensities),
        "mean_aspect_ratio": np.mean(region_aspect_ratios)
    }
'''


import cv2
import numpy as np


def skull_strip(img):
    """
    Remove skull/background to isolate brain tissue.
    Uses Otsu thresholding + morphology + largest connected component.
    """
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

    if num_labels < 2:
        return img

    largest_label = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    brain_mask = (labels == largest_label).astype("uint8") * 255

    # Fill internal holes
    floodfill = brain_mask.copy()
    h, w = brain_mask.shape
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(floodfill, mask_ff, (0, 0), 255)
    floodfill_inv = cv2.bitwise_not(floodfill)
    brain_mask = brain_mask | floodfill_inv

    brain = cv2.bitwise_and(img, img, mask=brain_mask)
    return brain


def segment_stroke(brain):
    """
    Detect stroke/lesion regions and return separate masks per type.

    Ischemic     → abnormally DARK patches  (blocked blood flow, less signal)
    Hemorrhagic  → abnormally BRIGHT patches (blood pooling, high signal)

    Returns:
        combined    — union of both masks (for general use / feature extraction)
        dark_mask   — ischemic candidate regions
        bright_mask — hemorrhagic candidate regions
    """
    empty = np.zeros_like(brain)

    if np.count_nonzero(brain) == 0:
        return empty, empty, empty

    brain_pixels = brain[brain > 0]
    mean_val = np.mean(brain_pixels)
    std_val  = np.std(brain_pixels)

    # 1.5 std below mean = ischemic candidate
    dark_thresh   = max(0,   int(mean_val - 1.5 * std_val))
    # 1.5 std above mean = hemorrhagic candidate
    bright_thresh = min(255, int(mean_val + 1.5 * std_val))

    dark_mask   = cv2.inRange(brain, 1, dark_thresh)
    bright_mask = cv2.inRange(brain, bright_thresh, 255)
    combined    = cv2.bitwise_or(dark_mask, bright_mask)

    # Morphological cleanup — remove noise, close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def clean(mask):
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    return clean(combined), clean(dark_mask), clean(bright_mask)


def extract_region_features(brain, mask):
    """
    Measure properties of suspicious regions for ML feature vector.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    region_areas         = []
    region_intensities   = []
    region_aspect_ratios = []
    brain_area           = np.count_nonzero(brain)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or (brain_area > 0 and area / brain_area > 0.4):
            continue

        region_mask = np.zeros_like(mask)
        cv2.drawContours(region_mask, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(brain, mask=region_mask)[0]

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio  = float(w) / h if h > 0 else 1.0

        region_areas.append(area)
        region_intensities.append(mean_intensity)
        region_aspect_ratios.append(aspect_ratio)

    if not region_areas:
        return {
            "num_regions": 0, "total_area": 0, "max_area": 0,
            "mean_intensity": 0, "min_intensity": 0, "mean_aspect_ratio": 1.0
        }

    return {
        "num_regions":       len(region_areas),
        "total_area":        sum(region_areas),
        "max_area":          max(region_areas),
        "mean_intensity":    np.mean(region_intensities),
        "min_intensity":     np.min(region_intensities),
        "mean_aspect_ratio": np.mean(region_aspect_ratios)
    }


def classify_stroke_type(dark_mask, bright_mask):
    """
    Determine stroke subtype based on which region type dominates.

    Logic:
      - Mostly dark  → ischemic   (blood flow blocked → dark on MRI/CT)
      - Mostly bright → hemorrhagic (bleeding → bright on MRI/CT)
      - Both present → mixed
      - Neither      → normal (no stroke regions found)

    Returns:
        stroke_type (str): "ischemic" | "hemorrhagic" | "mixed" | "normal"
        dark_area   (int): pixel count of dark (ischemic) regions
        bright_area (int): pixel count of bright (hemorrhagic) regions
    """
    dark_area   = int(np.sum(dark_mask   > 0))
    bright_area = int(np.sum(bright_mask > 0))
    total       = dark_area + bright_area

    if total < 50:
        return "normal", dark_area, bright_area

    bright_ratio = bright_area / total  # fraction that is hemorrhagic

    if bright_ratio > 0.65:
        stroke_type = "hemorrhagic"
    elif bright_ratio < 0.35:
        stroke_type = "ischemic"
    else:
        stroke_type = "mixed"

    return stroke_type, dark_area, bright_area