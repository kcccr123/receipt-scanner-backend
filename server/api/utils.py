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
from openai import OpenAI
from dotenv import load_dotenv
import json

def runYOLO(img, modelpath):
    # Ensure the image is in BGR format (from grayscale)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Load the YOLO model
    model = YOLO(modelpath)
    
    # Perform inference
    result = model(img, conf=0.3, iou=0.5)[0]
    
    
    # Get annotated image with detections
    #annotated_img = result.plot()

    # Save the annotated image if necessary
    #if os.path.exists('annotated_image.jpg'):
    #    os.remove('annotated_image.jpg')
    

    """# Save the annotated image
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
    decimal_pattern = re.compile(r'^\$?\d+\.\d{2}$')
    for result in rcnn_results:
        temp = []
        non_temp = []
        for i in result:
            if decimal_pattern.match(i):
                formatted = i.replace("$", "")
                temp.append(formatted)
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
        if "SUBTOTAL" in tag:
            sub_object = {"name": "##SUBTOTAL", "price": tag[tag.index(':') + 1:]}
            objects[len(objects)] = sub_object
        elif "TOTAL" in tag:
            total_object = {"name": "##TOTAL", "price": tag[tag.index(':') + 1:]}
            objects[len(objects)] = total_object

        elif "Price" in tag:
            item_object = {"name": string[:string.index(tag)], "price": tag[tag.index(':') + 1:]}
            objects[len(objects)] = item_object
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
        return (401, {"error": "Receipt is badly aligned, please try again."})
    

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
    #removed, maxs = findPrice(rcnn_results)

    #joined_lst = []
    
    #for i in removed:
    #    joined_lst.append(" ".join(i))

    print(rcnn_results, "rcnn results")
    bart_results = runBartPrediction(rcnn_results)
    print(bart_results, "here")

    # (temporary)
    # Bart is not producing tags, so add tags with max prices found earlier.
    #temp_staging = temporaryProcess(bart_results, labels, conversion, maxs)
    #print(temp_staging)

    # process results for response
    results = processPredictionForResponse(bart_results)
    return (200, results)


openai_api_key = os.getenv('OPENAI_API_KEY')

def runGptPrediction(values):

    client = OpenAI(
        api_key=openai_api_key,
    )

    # Prepare the prompt for GPT
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that processes a list of items. "
                "Each item is a list containing three elements:\n"
                "1. Item name or text (which may be misspelled)\n"
                "2. Price\n"
                "3. Tag (e.g., ##PRICE:, ##SUBTOTAL:, ##TOTAL:)\n\n"
                "For each item, please:\n"
                "- Correct any misspellings in the item name or text.\n"
                "- Swap the order if necessary so that the item name comes first, "
                "followed by the tag and price.\n"
                "- If the tag corresponds to a priced line item, output it as a JSON object with:\n"
                "    {\n"
                "      \"type\": \"receipt item\",\n"
                "      \"name\": \"<Corrected Item Name>\",\n"
                "      \"price\": <Price>\n"
                "    }\n"
                "- If the tag corresponds to a subtotal, output it as:\n"
                "    {\n"
                "      \"type\": \"subtotal\",\n"
                "      \"price\": <Price>\n"
                "    }\n"
                "- If the tag corresponds to a total, output it as:\n"
                "    {\n"
                "      \"type\": \"total\",\n"
                "      \"price\": <Price>\n"
                "    }\n"
                "Collect all these objects into a JSON array—nothing else. "
                "Return only that valid JSON array, for example:\n\n"
                "[\n"
                "  {\"type\": \"receipt item\", \"name\": \"Apple\", \"price\": 1.5},\n"
                "  {\"type\": \"subtotal\", \"price\": 10},\n"
                "  {\"type\": \"total\", \"price\": 10}\n"
                "]\n\n"
                "Do not include any additional text, explanation, or code fences."
            )
        },
        {
            "role": "user",
            "content": f"Here is the list of items:\n{values}"
        }
    ]

    # Call the GPT API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    # Extract the assistant's reply
    assistant_reply = response.choices[0].message.content
    print("GPT raw output:", assistant_reply)

    # Convert the JSON string to a Python list (or array of objects)
    try:
        corrected_list = json.loads(assistant_reply)
    except json.JSONDecodeError:
        raise ValueError("GPT did not return valid JSON.")

    return corrected_list
    


def runRecieptPredictionGpt(image):
    # takes base64 encoded receipt and makes api call to chatGPT to decipher contents.
    client = OpenAI(
        api_key=openai_api_key,
    )
    

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ],
        }
    ],
    )

    print(response.choices[0].message.content)




