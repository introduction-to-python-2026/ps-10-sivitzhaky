import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def load_image(path: str) -> np.ndarray:
    """
    Load image from disk and return as NumPy array.

    - If the image is RGB -> returns shape (H, W, 3) uint8.
    - If the image is grayscale -> returns shape (H, W).
      If it looks binary (values like {0,255} or {0,1}) -> returns boolean array (H, W).
    """
    img = Image.open(path)
    arr = np.array(img)

    # Some grayscale PNGs can arrive as (H, W, 1) -> squeeze
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # If grayscale, detect if it is a binary mask and return bool
    if arr.ndim == 2:
        # Check if values look like binary image (0/1 or 0/255)
        uniq = np.unique(arr)
        if np.all(np.isin(uniq, [0, 1])) or np.all(np.isin(uniq, [0, 255])):
            return (arr > 0)
        return arr

    # If RGB/RGBA: keep only RGB channels if needed
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    return arr


def edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Sobel edge detection.

    Input:
      - image: (H,W,3) RGB or (H,W) grayscale

    Output:
      - edgeMAG: (H,W) uint8 in range 0..255
    """
    # Convert to grayscale float
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    kernelY = np.array([[1,  2,  1],
                        [0,  0,  0],
                        [-1, -2, -1]], dtype=np.float64)

    kernelX = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)

    edgeMAG = np.sqrt(edgeX ** 2 + edgeY ** 2)

    # Normalize to 0..255 safely
    mn = float(edgeMAG.min())
    mx = float(edgeMAG.max())
    if mx > mn:
        edgeMAG = (edgeMAG - mn) / (mx - mn) * 255.0
    else:
        edgeMAG = np.zeros_like(edgeMAG)

    return edgeMAG.astype(np.uint8)
