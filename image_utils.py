import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and return it as a NumPy array.

    - If the file is grayscale, returns shape (H, W)
    - If the file is RGB, returns shape (H, W, 3)

    Important: If this is a binary mask image saved as 0/255,
    convert it to 0/1 so comparisons with boolean arrays work.
    """
    img = Image.open(path)
    arr = np.array(img)

    # If grayscale came as (H, W, 1) -> squeeze to (H, W)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # If RGBA -> drop alpha (just in case)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Convert binary mask 0/255 -> 0/1
    if arr.ndim == 2:
        vals = np.unique(arr)
        if vals.size <= 2 and set(vals.tolist()).issubset({0, 255}):
            arr = (arr > 0).astype(np.uint8)

    return arr


def edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Perform Sobel edge detection.

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

    # Sobel kernels
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

    # Normalize to 0..255 safely
    mn = float(edgeMAG.min())
    mx = float(edgeMAG.max())
    if mx > mn:
        edgeMAG = (edgeMAG - mn) / (mx - mn) * 255.0
    else:
        edgeMAG = np.zeros_like(edgeMAG)

    return edgeMAG.astype(np.uint8)
