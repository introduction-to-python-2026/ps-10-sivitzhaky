import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

def main():
    image = load_image("lena.jpg")
    image = median(image, ball(3))
    edge = edge_detection(image)
    edge_binary = (edge > 50).astype(np.uint8) * 255
    Image.fromarray(edge_binary).save("detected_edges.png")

if __name__ == "__main__":
    main()
