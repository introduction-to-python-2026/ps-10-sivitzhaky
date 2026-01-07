import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(file_path):
    """
    Loads a color image from the given file path and converts it into a NumPy array.

    Args:
        file_path (str): The path to the image file.

    Returns:
        np.array: The image as a NumPy array.
    """
    img = Image.open(file_path)
    img_array = np.array(img)
    return img_array

def edge_detection(image_array):
    """
    Performs edge detection on a color image array.

    Args:
        image_array (np.array): The input color image as a NumPy array (height, width, 3).

    Returns:
        np.array: The magnitude of the edges (edgeMAG) as a NumPy array.
    """
    grayscale_image = np.mean(image_array, axis=2).astype(float)

    kernelY = np.array([
        [ 1,  2,  1 ],
        [ 0,  0,  0 ],
        [-1, -2, -1 ]
    ], dtype=float)

    kernelX = np.array([
        [-1, 0, 1 ],
        [-2, 0, 2 ],
        [-1, 0, 1 ]
    ], dtype=float)

    edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    edgeMAG = np.interp(edgeMAG, (edgeMAG.min(), edgeMAG.max()), (0, 255)).astype(np.uint8)

    return edgeMAG
