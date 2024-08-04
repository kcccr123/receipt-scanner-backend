import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
image_path = r"D:\photos\RCNN4\BBOXES\98.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply thresholding
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Define a kernel for dilation
kernel = np.ones((5, 100), np.uint8)

# Dilate the binary image to combine characters into lines
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by top-to-bottom
bounding_boxes = [cv2.boundingRect(c) for c in contours]
bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])

# Extract and save lines
lines = []
for i, (x, y, w, h) in enumerate(bounding_boxes):
    line_image = image[y:y+h, x:x+w]
    lines.append(line_image)
    line_image_path = f'/mnt/data/line_{i+1}.png'
    # cv2.imwrite(line_image_path, line_image)

# Display the original image with bounding boxes for reference
image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in bounding_boxes:
    cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Plot the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image_with_boxes, cmap='gray')
plt.title('Detected Lines')
plt.axis('off')
plt.show()


