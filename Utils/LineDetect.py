# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  # Apply thresholding
#     return image, thresh

# def detect_contours(thresh):
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
#     return contours

# def merge_contours(contours, image_shape):
#     mask = np.zeros(image_shape, dtype=np.uint8)
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
#     # Dilate the mask to merge nearby boxes
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
#     dilated = cv2.dilate(mask, kernel, iterations=1)
#     merged_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return merged_contours

# def draw_bounding_boxes(image, contours):
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     return image

# # Example usage
# image_path = r"D:\photos\RCNN4\BBOXES\20.jpg"
# image, thresh = preprocess_image(image_path)
# contours = detect_contours(thresh)
# merged_contours = merge_contours(contours, thresh.shape)
# image_with_boxes = draw_bounding_boxes(image, merged_contours)

# # Display the result
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()



import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Load image
data_dir = r"D:\photos\RCNN4\BBOXES"


largest_index = 492


for id in range(0, largest_index+1):
    image_path = os.path.join(data_dir, str(id) + ".jpg").replace("\\","/")
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Apply thresholding
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Define a kernel for dilation
        kernel = np.ones((5, 50), np.uint8)

        # Dilate the binary image to combine characters into lines
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to remove small areas that are unlikely to be full lines
        min_contour_area = 1000  # Adjust this threshold as necessary
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        # Sort contours by top-to-bottom
        bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])

        # Extract and save lines
        lines = []
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            if w > 20 and h>30:
                line_image = image[y:y+h, x:x+w]
                lines.append(line_image)
                line_image_path = os.path.join("D:/photos/RCNN4/Lines", str(id) + "-" + str(i) + ".jpg").replace("\\","/")
                print(line_image_path)
                cv2.imwrite(line_image_path, line_image)

        # Display the original image with bounding boxes for reference
        image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in bounding_boxes:
            if w > 20 and h>30:
                cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Plot the image with bounding boxes
        plt.figure(figsize=(10, 2))
        plt.imshow(image_with_boxes, cmap='gray')
        plt.title('Detected Lines')
        plt.axis('off')
        plt.show()

        # Return the paths to the extracted lines
        lines_paths = [f'/mnt/data/line_{i+1}.png' for i in range(len(bounding_boxes))]
        lines_paths
    else:
        print("invalid")
        continue
