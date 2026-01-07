import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def load_image(path: str) -> np.ndarray:
    """
    Loads an image and returns a NumPy array.

    Important:
    - If the image is grayscale (mode 'L' or '1'), returns shape (H, W).
    - If the image is RGB, returns shape (H, W, 3).
    """
    img = Image.open(path)  # do NOT force RGB; keep original mode
    return np.array(img)


def edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Sobel edge detection.

    Accepts:
    - RGB image (H, W, 3) or grayscale image (H, W)

    Returns:
    - edge magnitude image in uint8 range [0, 255] with shape (H, W)
    """
    # Convert to grayscale float
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(float)
    else:
        gray = image.astype(float)

    kernelY = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=float)

    kernelX = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=float)

    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)

    mag = np.sqrt(edgeX ** 2 + edgeY ** 2)

    # Normalize to 0..255 safely
    mag_min, mag_max = mag.min(), mag.max()
    if mag_max == mag_min:
        return np.zeros_like(mag, dtype=np.uint8)

    mag_norm = np.interp(mag, (mag_min, mag_max), (0, 255)).astype(np.uint8)
    return mag_norm
