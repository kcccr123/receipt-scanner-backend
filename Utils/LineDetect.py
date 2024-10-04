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
from skimage.filters import threshold_local

# Load image
data_dir = r"D:\photos\RCNN_new_data\bboxes\items"
save_dir = r"D:\photos\RCNN_new_data\words2"


largest_index = 2651
# minHeight = 40
# minWidth = 60
num_results = 1764

def preprocess_image(image):
    # Load the image using OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    return invert


for id in range(0, largest_index+1):
    image_path = os.path.join(data_dir, str(id) + ".jpg").replace("\\","/")
    if os.path.exists(image_path):
        image = cv2.imread(image_path)

        img_height, img_width = image.shape[:2]

        minWidth = int(img_width * 0.05)
        minHeight = int(img_height * 0.25)
        minContourArea = int(img_width * img_height * 0.0015)

        image = preprocess_image(image)

        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        kernel = np.ones((3, 15), np.uint8)


        dilated_image = cv2.dilate(binary_image, kernel, iterations=2)

        start_x = img_width * 3 // 4  # Start from 3/4th of the width to the end
        roi = dilated_image[:, start_x:]


        kernel = np.ones((3, 15), np.uint8)

        # Apply dilation on the ROI
        dilated_roi = cv2.dilate(roi, kernel, iterations=2)

        # Replace the right quarter of the original image with the dilated ROI
        dilated_image[:, start_x:] = dilated_roi

        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = [c for c in contours if cv2.contourArea(c) > minContourArea]
        bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])

        # Extract and save lines
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            if w > minWidth and h>minHeight:
                line_image = image[y:y+h, x:x+w]
                line_image_path = os.path.join(save_dir, str(num_results) + ".png").replace("\\","/")
                num_results += 1
                cv2.imwrite(line_image_path, line_image)

        # # Display the original image with bounding boxes for reference
        # image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # for (x, y, w, h) in bounding_boxes:
        #     if w > 20 and h>30:
        #         cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # # Plot the image with bounding boxes
        # plt.figure(figsize=(10, 2))
        # plt.imshow(image_with_boxes, cmap='gray')
        # plt.title('Detected Lines')
        # plt.axis('off')
        # plt.show()

        # # Return the paths to the extracted lines
        # lines_paths = [f'/mnt/data/line_{i+1}.png' for i in range(len(bounding_boxes))]
        # lines_paths
    else:
        print("invalid")
        continue
print(f"total: {num_results}")