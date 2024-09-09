import onnxruntime as ort
from itertools import groupby
import cv2
import numpy as np
import torch

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

    def preprocess(self, image: np.ndarray):
        # Load the image using OpenCV
        blur = cv2.GaussianBlur(image, (3,3), 0)

        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening

        return invert

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, (128, 32))

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        image_pred = np.expand_dims(image_pred, axis=1)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        #highest prob
        argmax_preds = np.argmax(preds, axis=-1)
        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]
        texts = ["".join([self.metadata["vocab"][k] for k in group if k < len(self.metadata["vocab"])]) for group in grouped_preds]
        text = texts[0]

        return text
    
    def run(self, image: np.ndarray, bbox_coords: list): #list of format [[xmin, xmax, ymin, ymax], [xmin, xmax, ymin, ymax], [xmin, xmax, ymin, ymax],....]
        bboxes = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(bbox_coords):
            box = image[y_min:y_max, x_min:x_max,]
            bboxes.append(box)
        
        results = []
        for b in bboxes:

            grey_image = self.preprocess(b)
            img_height, img_width = grey_image.shape[:2]

            minWidth = int(img_width * 0.05)
            minHeight = int(img_height * 0.25)
            minContourArea = int(img_width * img_height * 0.0015)

            # cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(grey_image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            kernel = np.ones((3, 17), np.uint8)
            dilated_image = cv2.dilate(binary_image, kernel, iterations=2)

            start_x = img_width * 3 // 4 
            roi = dilated_image[:, start_x:]
            kernel = np.ones((5, 36), np.uint8)
            dilated_roi = cv2.dilate(roi, kernel, iterations=1)
            dilated_image[:, start_x:] = dilated_roi

            contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            filtered_contours = [c for c in contours if cv2.contourArea(c) > minContourArea]
            bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]
            bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])

            elements = []
            for i, (x, y, w, h) in enumerate(bounding_boxes):
                if w > minWidth and h> minHeight:
                    line_image = grey_image[y:y+h, x:x+w]
                    elements.append(line_image)

            text = []
            for capture in elements:
                mean = 198.87491
                std = 105.648796
                img_tensor = torch.tensor(capture, dtype=torch.float32)
                norm_img = (img_tensor - mean) / std
                processed_capture = norm_img.numpy()
                prediction_text = self.predict(processed_capture)
                text.append(prediction_text)
            results.append(text)

            for (x, y, w, h) in bounding_boxes:
                if w > minWidth and h> minHeight:
                    cv2.rectangle(b, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Plot the image with bounding boxes
            print(text)
        return results #results is a [[item: str], [item: str], [item: str], [item: str]....]