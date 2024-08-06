from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
import onnxruntime as ort
from itertools import groupby
# import matplotlib.pyplot as plt
# import os

def runYOLO(img, modelpath):
    # image is a numpy array, in BGR 
    model = YOLO(modelpath)
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
            bounding_boxes.append([int(coord) for coord in xyxy.tolist()])
    print(bounding_boxes, "bounding boxes")
    return bounding_boxes


class inferencemode:
    def __init__(self, model_path: str = ""):
        self.model_path = model_path.replace("\\", "/")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]

        self.model = ort.InferenceSession(model_path, providers = providers)

        self.metadata = {}
        for key, value in self.model.get_modelmeta().custom_metadata_map.items():
            new_value = value
            self.metadata[key] = new_value

        self.input_shapes = [meta.shape for meta in self.model.get_inputs()]
        self.input_names = [meta.name for meta in self.model._inputs_meta]
        self.output_names = [meta.name for meta in self.model._outputs_meta]

    def predict(self, image: np.ndarray):
        # image is a numpy array, in BGR 
        image = cv2.resize(image, (128, 32))

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        #highest prob
        argmax_preds = np.argmax(preds, axis=-1)
        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]
        texts = ["".join([self.metadata["vocab"][k] for k in group if k < len(self.metadata["vocab"])]) for group in grouped_preds]
        text = texts[0]

        return text
    
    def run(self, image: np.ndarray, bbox_coords: list): #list of format [[xmin, xmax, ymin, ymax], [xmin, xmax, ymin, ymax], [xmin, xmax, ymin, ymax],....]
        # image is a numpy array, in BGR 
        bboxes = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(bbox_coords):
            box = image[y_min:y_max, x_min:x_max,]
            bboxes.append(box)
        
        results = []
        for b in bboxes:

            grey_image = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(grey_image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            kernel = np.ones((10, 36), np.uint8)
            dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_contour_area = 1000
            filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
            bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
            bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])

            elements = []
            for i, (x, y, w, h) in enumerate(bounding_boxes):
                if w > 20 and h> 25:
                    line_image = b[y:y+h, x:x+w]
                    elements.append(line_image)

            text = []
            for capture in elements:
                prediction_text = self.predict(capture)
                text.append(prediction_text)
            results.append(text)

        return results #results is a [[item: str], [item: str], [item: str], [item: str]....]
    


def runRecieptPrediction(image, yoloPath, rcnnPath):
    img_byte_arr = io.BytesIO()
    Image.open(image).save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    np_img = np.frombuffer(img_byte_arr, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # run yolo model to get bounding boxes
    bounding_boxes = runYOLO(img, yoloPath)

    # run rcnn to decipher words
    rcnn = inferencemode(rcnnPath)
    results = rcnn.run(img, bounding_boxes)
    if isinstance(results, np.ndarray):
            print('check')
            results = results.tolist()
    return results

