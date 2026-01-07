import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def load_image(file_path):
    img = Image.open(file_path)
    return np.array(img)


def edge_detection(image_array):
    # Convert to grayscale
    gray = np.mean(image_array, axis=2)

    # Sobel kernels
    kernel_y = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])

    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Convolution
    edge_y = convolve2d(gray, kernel_y, mode="same", boundary="fill", fillvalue=0)
    edge_x = convolve2d(gray, kernel_x, mode="same", boundary="fill", fillvalue=0)

    # Edge magnitude (NO normalization, NO threshold)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)

    return edge_mag
