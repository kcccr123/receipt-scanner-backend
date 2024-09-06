from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
import onnxruntime as ort
from itertools import groupby
from transformers import BartTokenizer, BartForConditionalGeneration
from fix_angle import fix_angle
import os
import matplotlib.pyplot as plt
import torch

def runYOLO(img, modelpath):
    # Ensure the image is in BGR format (from grayscale)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Load the YOLO model
    model = YOLO(modelpath)
    
    # Perform inference
    result = model(img, conf=0.3, iou=0.5)[0]
    
    # Get annotated image with detections
    annotated_img = result.plot()

    # Save the annotated image if necessary
    if os.path.exists('annotated_image.jpg'):
        os.remove('annotated_image.jpg')
    
    # Save the annotated image
    cv2.imwrite('annotated_image.jpg', annotated_img)

    # Display the annotated image with bounding boxes using Matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title("Annotated Image with YOLO Detections")
    plt.axis('off')
    plt.show()

    bounding_boxes = []
    labels = []

    # Extract the bounding box coordinates
    class_names = model.names
    for c in result.boxes.cls:
        labels.append(class_names[int(c)])

    # Convert into cv2 rectangle format
    boxes = result.boxes.cpu().numpy()
    for xyxy in boxes.xyxy:
        bounding_boxes.append([int(coord) for coord in xyxy.tolist()])
    
    print(bounding_boxes, "bounding boxes")
    print(labels, "labels")
    return bounding_boxes, labels



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
        return results #results is a [[item: str], [item: str], [item: str], [item: str]....]

# BART Loading
device = torch.device("cpu")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
model.load_state_dict(torch.load("models/bart_model.pt", map_location=device))
model.to(device)
model.eval()
length = 60

def runBartPrediction(lst):
    print(lst)
    result = []
    for item in lst:
        input_text = " ".join(item)
        # Tokenize the input
        tokenized_input = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=length, truncation=True)

        # Generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(input_ids=tokenized_input["input_ids"], attention_mask=tokenized_input["attention_mask"], max_length=length)
        predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        result.append(predicted_text)
    print (result)
    return result

def processPredictionForResponse(predictions):
    objects = []

    for string in predictions:
        tag = ""
        obj = {}
        for i in range(len(string) - 1, 0, -1):
           if string[i] + string[i - 1] == "##":
                tag = string[i - 1:]
                hash_start = i - 1
                break
        
        if "TOTAL" in tag:
            if "total" in obj:
                obj["total"].append(tag[8:])
            else:
                obj["total"] = [tag[8:]]
        elif "PRICE" in tag:
            value = string[:hash_start]
            if value in obj:
                obj[value].append(tag[8:])
            else:
                obj[value] = [tag[8:]]
        elif "SUBTOTAL" in tag:
            if "subtotal" in obj:
                obj["subtotal"].append(tag[10:])
            else:
                obj["subtotal"] = [tag[10:]]
        else:
            continue
        if len(obj) > 0:
            objects.append(obj)
    
    # handle multiple objects of same type before return
    return objects

def runRecieptPrediction(image, yoloPath, rcnnPath):
    img_byte_arr = io.BytesIO()
    Image.open(image).save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    np_img = np.frombuffer(img_byte_arr, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    #some moderate issue here

    fixed_image, fixed_image_coloured = fix_angle(img)

    if len(fixed_image) == 0:
        return (400, {"error": "Receipt is badly aligned, please try again."})
    

    # run yolo model to get bounding boxes
    bounding_boxes, labels = runYOLO(fixed_image, yoloPath)
        
    
    # run rcnn to decipher words
    rcnn = inferencemode(rcnnPath)
    rcnn_results = rcnn.run(cv2.cvtColor(fixed_image_coloured, cv2.COLOR_BGR2GRAY), bounding_boxes)
    if isinstance(rcnn_results, np.ndarray):
        print('check')
        rcnn_results = rcnn_results.tolist()

    conversion = {'item': "##PRICE:", 'subtotal': '##SUBTOTAL:', 'total': '##TOTAL:'}

    for i in range(len(rcnn_results)):
        rcnn_results[i].append(conversion[labels[i]])
    
    bart_results = runBartPrediction(rcnn_results)
    results = processPredictionForResponse(bart_results)
    return (500, results)
