import numpy as np
from PIL import Image
from scipy import ndimage

def load_image(file_path: str) -> np.ndarray:
    """
    Load a color image and return as a NumPy array (H, W, 3).
    """
    img = Image.open(file_path).convert("RGB")
    return np.array(img)

def edge_detection(image_array: np.ndarray) -> np.ndarray:
    """
    Convert RGB image -> grayscale, apply Sobel-like filters (X,Y),
    return edge magnitude normalized to 0..255 (uint8).
    """
    # Grayscale by averaging channels
    gray = np.mean(image_array.astype(np.float32), axis=2)

    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]], dtype=np.float32)

    kernelX = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]], dtype=np.float32)

    edgeY = ndimage.convolve(gray, kernelY, mode="constant", cval=0.0)
    edgeX = ndimage.convolve(gray, kernelX, mode="constant", cval=0.0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    # Normalize to 0..255 safely
    mn, mx = float(edgeMAG.min()), float(edgeMAG.max())
    if mx - mn < 1e-12:
        norm = np.zeros_like(edgeMAG, dtype=np.uint8)
    else:
        norm = ((edgeMAG - mn) / (mx - mn) * 255.0).astype(np.uint8)

    return norm
