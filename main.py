import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
# Import functions from image_utils.py
from image_utils import load_image, edge_detection

# --- Step 1: Ensure a test image exists and load it ---
image_path = 'test_image.png'
if not os.path.exists(image_path):
    print(f"Creating dummy image '{image_path}' for demonstration...")
    dummy_img = Image.new('RGB', (100, 100), color = 'red')
    dummy_img.save(image_path)
    print(f"Dummy image '{image_path}' created.")

try:
    image_array = load_image(image_path)
    print(f"Image '{image_path}' loaded successfully.")
    print(f"Shape: {image_array.shape}, Data type: {image_array.dtype}")
except FileNotFoundError:
    print(f"Error: Image file '{image_path}' not found. Please ensure it exists.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the image: {e}")
    exit()

# --- Step 2: Suppress noise using a median filter ---
print("Applying median filter to suppress noise...")
clean_image = median(image_array, ball(3))
print("Noise suppression completed.")
print(f"Shape of clean image: {clean_image.shape}, Data type: {clean_image.dtype}")

# --- Step 3: Run the noise-free image through edge_detection ---
print("Detecting edges on the clean image...")
edges = edge_detection(clean_image)
print("Edge detection completed.")
print(f"Shape of edge magnitude array: {edges.shape}, Data type: {edges.dtype}")

# --- Step 4: Convert to binary image with a threshold ---
threshold = 100 # You can experiment with different threshold values
print(f"Applying threshold of {threshold} to convert to binary image...")
edge_binary = edges > threshold
print("Conversion to binary image completed.")
print(f"Shape of binary edge array: {edge_binary.shape}, Data type: {edge_binary.dtype}")

# --- Step 5: Display and save the binary image ---
# Convert the boolean array to uint8 (0 or 255) for display and saving
edge_binary_display = (edge_binary * 255).astype(np.uint8)

# Display the image using matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(edge_binary_display, cmap='gray')
plt.title('Binary Edge Detected Image')
plt.axis('off') # Hide axes ticks
plt.show()

# Save the image as a .png file
output_filename = 'my_edges.png'
edge_image = Image.fromarray(edge_binary_display)
edge_image.save(output_filename)
print(f"Binary edge detected image saved as '{output_filename}'.")

