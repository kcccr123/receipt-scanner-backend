import onnxruntime as ort
from itertools import groupby
import cv2
import numpy as np
from skimage.filters import threshold_local


class inferencemode:
    def __init__(self, model_path: str = ""):
        self.model_path = model_path.replace("\\", "/")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]

        self.model = ort.InferenceSession(self.model_path, providers = providers)

        self.metadata = {}
        for key, value in self.model.get_modelmeta().custom_metadata_map.items():
            new_value = value
            self.metadata[key] = new_value

        self.input_shapes = [meta.shape for meta in self.model.get_inputs()]
        self.input_names = [meta.name for meta in self.model._inputs_meta]
        self.output_names = [meta.name for meta in self.model._outputs_meta]

    def resize_maintaining_aspect_ratio(self, image: np.ndarray, width_target: int = 224, height_target: int = 36, padding_color: int=0) -> np.ndarray:

        height, width = image.shape[:2]
        ratio = min(width_target / width, height_target / height)
        new_w, new_h = int(width * ratio), int(height * ratio)

        resized_image = cv2.resize(image, (new_w, new_h))
        delta_w = width_target - new_w
        delta_h = height_target - new_h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

        return new_image
    def bw_scanner(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T = threshold_local(gray, 21, offset = 5, method = "gaussian")
        return (gray > T).astype("uint8") * 255

    def preprocess(self, image: np.ndarray):
        # Load the image using OpenCV
        im = self.bw_scanner(image)
        blur = cv2.GaussianBlur(im, (3,3), 0)

        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening

        return invert

    def predict(self, image: np.ndarray):
        resized_img = self.resize_maintaining_aspect_ratio(image)

        image_pred = np.expand_dims(resized_img, axis=0).astype(np.float32)
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

            boxing_image = self.preprocess(b)
            prediction_image = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

            img_height, img_width = boxing_image.shape[:2]

            minWidth = int(img_width * 0.05)
            minHeight = int(img_height * 0.25)
            minContourArea = int(img_width * img_height * 0.0015)

            # cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(boxing_image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
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
                    line_image = prediction_image[y:y+h, x:x+w]
                    elements.append(line_image)

            text = []
            for capture in elements:
                prediction_text = self.predict(capture)
                text.append(prediction_text)
            results.append(text)

            print(text)
        return results #results is a [[item: str], [item: str], [item: str], [item: str]....]