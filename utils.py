import cv2
import numpy as np


def show(title, img):
    """Display an image window (blocking)."""
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_image(path):
    """Load image from path, return None with error message if failed."""
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Could not load image: {path}")
    return img


def normalize(img):
    """Normalize image to 0-255 uint8."""
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)
