import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path: str) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr

def edge_detection(image: np.ndarray) -> np.ndarray:
    # image is GRAYSCALE here
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

    # normalize to 0..255
    mn, mx = edgeMAG.min(), edgeMAG.max()
    if mx > mn:
        edgeMAG = (edgeMAG - mn) / (mx - mn) * 255
    else:
        edgeMAG = np.zeros_like(edgeMAG)

    return edgeMAG.astype(np.uint8)
