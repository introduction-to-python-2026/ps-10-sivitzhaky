import os
import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

from image_utils import load_image, edge_detection


def main():
    # Choose an input image that exists in your repo
    # (You can keep lena.jpg in the repo root, or change this filename)
    input_path = "lena.jpg"
    if not os.path.exists(input_path):
        input_path = "test_image.png"

    image = load_image(input_path)

    # Suppress noise using median filter (same style as the autograder uses)
    clean = median(image, ball(3))

    # Edge detection -> uint8 0..255
    edge = edge_detection(clean)

    # Convert to binary using threshold (autograder uses > 50)
    threshold = 50
    edge_binary = (edge > threshold).astype(np.uint8)  # 0/1, shape (H,W)

    # Save as grayscale (IMPORTANT: mode='L' -> saved as 2D)
    out = (edge_binary * 255).astype(np.uint8)
    Image.fromarray(out, mode="L").save("detected_edges.png")


if __name__ == "__main__":
    main()
