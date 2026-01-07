import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path: str) -> np.ndarray:
    """
    Load image from disk and return as numpy array.
    Keeps original channels (RGB stays (H,W,3); grayscale stays (H,W)).
    """
    img = Image.open(path)
    arr = np.array(img)

    # If image comes as (H,W,1) convert to (H,W)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    return arr

def edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Sobel edge detection.
    Accepts either (H,W) grayscale or (H,W,3) RGB.
    Returns (H,W) uint8 values in range 0..255.
    """
    # Ensure grayscale 2D
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    kernelY = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float64)

    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    # Normalize to 0..255
    mn = float(edgeMAG.min())
    mx = float(edgeMAG.max())
    if mx > mn:
        edgeMAG = (edgeMAG - mn) / (mx - mn) * 255.0
    else:
        edgeMAG = np.zeros_like(edgeMAG)

    return edgeMAG.astype(np.uint8)
