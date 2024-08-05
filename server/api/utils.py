from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import os

model = YOLO('models/bestonnx.pt')

def runYOLO(image):
    # 1. Dockerify
    # 2. Set up kubernetes config
    # 3. upload to cloud platform
    
    img_byte_arr = io.BytesIO()
    Image.open(image).save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    np_img = np.frombuffer(img_byte_arr, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    results = model(img, conf=0.4, iou=0.4)

    # annotated_img = results[0].plot()
    # Save the annotated image
    # if os.path.exists('annotated_image.jpg'):
    #         os.remove('annotated_image.jpg')
    # cv2.imwrite('annotated_image.jpg', results[0].plot())

    # Load and display the saved image using matplotlib
    # plt.figure()
    # annotated_img = cv2.imread('annotated_image.jpg')
    # plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    # plt.title("Annotated Image with Detections")
    # plt.show()

    bounding_boxes = []
    # loop through results
    for result in results:
        boxes = result.boxes.cpu().numpy()
    
        # convert into cv2 rectangle
        for xyxy in boxes.xyxy:
            bounding_boxes.append(xyxy)
        
    return bounding_boxes