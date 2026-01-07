import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

from image_utils import load_image, edge_detection


def main():
    # אם רוצים לעבוד על תמונה שלך בריפו:
    input_path = "lena.jpg"   # או כל תמונה אחרת ששמת בריפו
    output_path = "detected_edges.png"

    img = load_image(input_path)
    img_clean = median(img, ball(3))
    edge = edge_detection(img_clean)

    edge_binary = (edge > 50)              # bool
    out = (edge_binary.astype(np.uint8) * 255)  # 0/255

    Image.fromarray(out).save(output_path)


if __name__ == "__main__":
    main()
