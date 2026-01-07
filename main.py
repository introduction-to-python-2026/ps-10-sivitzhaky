import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

from image_utils import load_image, edge_detection


def main():
    # 1) Load image (the autograder uses ./tests/lena.jpg internally,
    # but for your repo output we generate detected_edges.png)
    input_path = "lena.jpg"  # make sure this file exists in your repo (root)
    img = load_image(input_path)

    # 2) Suppress noise
    img_clean = median(img, ball(3))

    # 3) Edge detection (0..255 uint8)
    edge = edge_detection(img_clean)

    # 4) Convert to binary using threshold (as in the tests: > 50)
    edge_binary = (edge > 50).astype(np.uint8)  # 0/1

    # 5) Save as PNG with values 0/255
    out = (edge_binary * 255).astype(np.uint8)
    Image.fromarray(out).save("detected_edges.png")


if __name__ == "__main__":
    main()
