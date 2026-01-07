import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

from image_utils import load_image, edge_detection

def main():
    # 1) Load image (make sure this file exists in your GitHub repo)
    input_image_path = "lena.jpg"          # <-- אם קראת לקובץ אחרת, תשני כאן
    output_edges_path = "detected_edges.png"

    image = load_image(input_image_path)

    # 2) Suppress noise using median filter
    clean_image = median(image, ball(3))

    # 3) Edge detection
    edgeMAG = edge_detection(clean_image)  # uint8 0..255

    # 4) Convert to binary using a threshold
    # (choose a threshold; you can tune it)
    threshold = 100
    edge_binary = (edgeMAG > threshold).astype(np.uint8)  # 0/1

    # 5) Save as PNG (0..255)
    edge_binary_img = (edge_binary * 255).astype(np.uint8)
    Image.fromarray(edge_binary_img).save(output_edges_path)

    # Optional: print info (usually safe)
    print(f"Saved edges image to: {output_edges_path}")

if __name__ == "__main__":
  main()
