from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
from transformers import BartTokenizer, BartForConditionalGeneration
from fix_angle import fix_angle
import os
import matplotlib.pyplot as plt
import torch
import re
from inference_mode import inferencemode

def runYOLO(img, modelpath):
    # Ensure the image is in BGR format (from grayscale)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Load the YOLO model
    model = YOLO(modelpath)
    
    # Perform inference
    result = model(img, conf=0.3, iou=0.5)[0]
    
    """
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
    plt.show()"""
    

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

def findPrice(rcnn_results):
    maxs = []
    removed = []
    decimal_pattern = re.compile(r'^\d+\.\d{2}$')
    for result in rcnn_results:
        temp = []
        non_temp = []
        for i in result:
            if decimal_pattern.match(i):
                temp.append(i)
            else:
                non_temp.append(i)
        if len(temp) > 0:
            maxs.append(max(temp))
        else:
            maxs.append(0)
        removed.append(non_temp)
    print(removed, maxs)
    return removed, maxs

def temporaryProcess(bartResults, labels, conversion, maxs):


    for i in range(len(bartResults)):
        bartResults[i] += " " + conversion[labels[i]] + str(maxs[i])

    return bartResults



def processPredictionForResponse(predictions):
    objects = {}

    for string in predictions:
        print(string)
        try:
            tag = string[string.index("##"):]
        except:
            continue
        if "TOTAL" in tag:
            total_object = {"name": "##TOTAL", "price": tag[tag.index(':') + 1:]}
            objects[len(objects)] = total_object

        elif "PRICE" in tag:
            
            item_object = {"name": string[:string.index(tag)], "price": tag[tag.index(':') + 1:]}
            objects[len(objects)] = item_object
        elif "SUBTOTAL" in tag:
            sub_object = {"name": "##SUBTOTAL", "price": tag[tag.index(':') + 1:]}
            objects[len(objects)] = sub_object
        else:
            continue
    
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
    rcnn_results = rcnn.run(fixed_image_coloured, bounding_boxes)
    if isinstance(rcnn_results, np.ndarray):
        print('check')
        rcnn_results = rcnn_results.tolist()

    conversion = {'item': "##PRICE:", 'subtotal': '##SUBTOTAL:', 'total': '##TOTAL:'}

    # append labels to end of rcnn results
    for i in range(len(rcnn_results)):
        rcnn_results[i].append(conversion[labels[i]])


    # (temporary)
    # remove prices from rcnn results and find maxs
    removed, maxs = findPrice(rcnn_results)
    
    bart_results = runBartPrediction(removed)

    # (temporary)
    # Bart is not producing tags, so add tags with max prices found earlier.
    temp_staging = temporaryProcess(bart_results, labels, conversion, maxs)
    print(temp_staging)

    # process results for response
    results = processPredictionForResponse(temp_staging)
    return (500, results)
