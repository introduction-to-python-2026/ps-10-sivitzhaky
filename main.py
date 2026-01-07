import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

from image_utils import load_image, edge_detection

def main():
    # 1. Load image
    img = load_image("lena.jpg")

    # 2. Convert to grayscale BEFORE median
    if img.ndim == 3:
        img = np.mean(img, axis=2)

    # 3. Median filter
    img = median(img, ball(3))

    # 4. Edge detection
    edge = edge_detection(img)

    # 5. Threshold EXACTLY like the test
    edge_binary = (edge > 50).astype(np.uint8)

    # 6. Save result
    out = edge_binary * 255
    Image.fromarray(out).save("detected_edges.png")

if __name__ == "__main__":
    main()
