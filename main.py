import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball

from image_utils import load_image, edge_detection


# --- Step 1: Load image ---
image_path = "lena.jpg"
image_array = load_image(image_path)

# --- Step 2: Edge detection ---
edges = edge_detection(image_array)

# --- Step 3: Noise suppression on edge image ---
clean_edges = median(edges, ball(3))

# --- Step 4: Convert to binary image ---
# threshold chosen after inspecting histogram / experimentation
threshold = 40
edge_binary = (clean_edges > threshold).astype(np.uint8)

# --- Step 5: Display ---
plt.figure(figsize=(6, 6))
plt.imshow(edge_binary, cmap="gray")
plt.title("Binary Edge Image")
plt.axis("off")
plt.show()

# --- Step 6: Save result ---
binary_for_save = edge_binary * 255
Image.fromarray(binary_for_save).save("my_edges.png")
