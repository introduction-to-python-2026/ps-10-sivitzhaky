import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

from image_utils import load_image, edge_detection


def main():
    """
    Loads an image, suppresses noise, detects edges,
    converts the result to a binary image, and saves it.
    """

    # Step 1: load image
    image = load_image("lena.jpg")

    # Step 2: suppress noise (median filter)
    image = median(image, ball(3))

    # Step 3: edge detection
    edges = edge_detection(image)

    # Step 4: threshold to binary image (as required by the test)
    edge_binary = edges > 50

    # Step 5: save binary image as PNG (0 / 255)
    output = (edge_binary.astype(np.uint8)) * 255
    Image.fromarray(output).save("detected_edges.png")


if __name__ == "__main__":
    main()
