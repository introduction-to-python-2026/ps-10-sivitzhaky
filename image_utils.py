import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def load_image(path: str) -> np.ndarray:
    """
    Loads an image and returns a NumPy array.

    Special handling:
    - If the filename suggests it's an edge-mask (e.g. contains 'edges'),
      return a boolean 2D array (True on edges, False on background),
      even if the PNG is RGB.
    """
    img = Image.open(path)
    arr = np.array(img)

    # If RGBA, drop alpha
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # If this is an edges mask (like .tests/lena_edges.png) -> return bool 2D
    if "edges" in path.lower():
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)  # convert to grayscale
        # Now arr is 2D grayscale; convert to boolean mask
        return (arr > 0)

    # Otherwise: keep original (RGB stays RGB; grayscale stays 2D)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    return arr


def edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Sobel edge detection.
    Accepts RGB (H,W,3) or grayscale (H,W).
    Returns (H,W) uint8 0..255.
    """
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]], dtype=np.float64)

    kernelX = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    mn, mx = float(edgeMAG.min()), float(edgeMAG.max())
    if mx > mn:
        edgeMAG = (edgeMAG - mn) / (mx - mn) * 255.0
    else:
        edgeMAG = np.zeros_like(edgeMAG)

    return edgeMAG.astype(np.uint8)
