import cv2
import numpy as np

TARGET_SIZE = (128, 128)


def preprocess_image(img, target_size=TARGET_SIZE):
    """
    Convert to grayscale, resize, denoise, and enhance contrast.
    Works on both BGR (from cv2.imread) and already-grayscale images.
    """
    if img is None:
        return None

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.resize(gray, target_size)

    # Median blur to remove salt-and-pepper noise
    blurred = cv2.medianBlur(gray, 5)

    # CLAHE for better local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    return enhanced
