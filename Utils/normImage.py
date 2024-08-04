import os
import torch
import pandas as pd
from PIL import Image as im 
import json as js
import cv2
import numpy as np

#load data
#Specify path to main database
data_dir = r"D:\photos\RCNN4\BBOXES"
model_path = r"D:\Projects\reciept-scanner\RCNN\models"
save_dir = r"D:\photos\dataset_norm"
database, vocab, max_len = [], set(), 0

largest_index = 492

for id in range(0, largest_index+1):

    img_path = os.path.join(data_dir, str(id) + ".jpg").replace("\\","/")
    
    if os.path.exists(img_path):
        
        with open(os.path.join(data_dir, str(id) + ".txt").replace("\\","/"), 'r') as file:
            ground_truths = [line.strip() for line in file.readlines()]
        
        for line in ground_truths:
            if not line.strip() == '':
                label = line.rstrip("\n")
                database.append([img_path, label])
                vocab.update(list(label))
                max_len = max(max_len, len(label))
    else:
        print("image with index " + str(id) + " do not exist")

print("dataset1 done")

#load second dataset
data_path = r"D:\photos\SORIE"

path = os.path.join(data_path, "train").replace("\\","/")
i = 1
while i <= 2:
    with open(os.path.join(path, "metadata.jsonl").replace("\\","/"), 'r') as file:
        for line in file:
            
            row = js.loads(line)
            img_path = os.path.join(path, row.get("file_name")).replace("\\","/")
            if os.path.exists(img_path):
                label = row.get("text").rstrip("\n")
                vocab.update(list(label))
                max_len = max(max_len, len(label))
                database.append([img_path, label])
            else:
                print("image with path " + str(img_path) + " do not exist")
    if i == 1:
        print("dataset2 done")
    
    i += 1
    path = os.path.join(data_path, "test").replace("\\","/")

print("dataset3 done")
print(len(database))

#normalize images and load them as an np array

#get the mean and std of dataset
num_pixels = 0
channel_sum = torch.tensor([0.0, 0.0, 0.0])
channel_sum_squared = torch.tensor([0.0, 0.0, 0.0])

for pair in database:
    image_path = pair[0]
    img = cv2.imread(image_path)

    if img is None:
        print(f"Warning: Could not read image at {image_path}. Skipping.")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pair[0] = img
    height, width, num_channels = img.shape

    num_pixels += height * width

    channel_sum += torch.tensor(img.sum(axis=(0, 1)), dtype=torch.float64)
    channel_sum_squared += torch.tensor((img.astype(np.float64) ** 2).sum(axis=(0, 1)), dtype=torch.float64)

mean = channel_sum/num_pixels
variance = (channel_sum_squared / num_pixels) - (mean**2)
std = torch.sqrt(variance)
print("Mean:", mean.numpy())
print("Standard Deviation:", std.numpy())

index = []
labels = []

i = 0
for pair in database:
    img_tensor = torch.tensor(pair[0], dtype=torch.float64)
    norm_img = (img_tensor - mean) / std
    image_save = norm_img.numpy()
        
    save_path = os.path.join(save_dir, str(i) + ".png" ).replace("\\","/")
    cv2.imwrite(save_path, image_save)
    
    index.append(str(i)+".png")
    labels.append(pair[1])
    i+=1

dict = {'name': index, 'label': labels}

df = pd.DataFrame(dict)
df.to_csv(os.path.join(save_dir, "annotations.csv").replace("\\","/"))


