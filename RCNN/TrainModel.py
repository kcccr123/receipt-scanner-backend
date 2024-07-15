import os
import torch
import pandas as pd
import callbacks
import data
import dataUtils as du
import image
import metric
import modelArc
import trainer
import tqdm
from config import ConfigFile
from datetime import datetime
import json as js

#load data
#Specify path to main database
data_dir = r"D:\photos\RCNN4\BBOXES"
model_path = r"D:\Projects\reciept-scanner\RCNN\models"
database, vocab, max_len = [], set(), 0

largest_index = 492


print("dataset of " + str(largest_index) + " images")

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

print("database, vocab, max_len, complete")


#create data loaders
model_config = ConfigFile(name = "CRNN1", path = model_path, lr=0.008, bs=32)

model_config.vocab = "".join(vocab)
model_config.max_txt_len = max_len
model_config.save()

dataset_loader = data.DataLoader(dataset = database, batch_size = model_config.batch_size, 
                                 data_preprocessors = [image.ImageReader(image.CVImage)], 
                                 transformers = [du.ImageResizer(model_config.width, model_config.height), du.LabelIndexer(model_config.vocab), 
                                                 du.LabelPadding(padding_value = len(model_config.vocab), max_word_len = max_len)])#, du.ImageShowCV2()


train_set, val_set = dataset_loader.split(split = 0.7)

train_set.augmentors = [
    du.RandomBrightness(),
    du.RandomErodeDilate(),
    du.RandomSharpen(),
    du.RandomRotate(angle=10),
    ]

#initialize model, optimizer, and loss
model = modelArc.CRNN(len(model_config.vocab))
loss = trainer.CTCLoss(blank = len(model_config.vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=model_config.lr)

if torch.cuda.is_available():
    model = model.cuda()
    print("CUDA Enabled...Training On GPU")

#initialze callbacks and trainer
earlystop = callbacks.EarlyStopping(monitor = "val_CER", patience = 50, verbose = True)
ckpt = callbacks.ModelCheckpoint((model_config.model_path + "/model.pt").replace("\\","/"), monitor = "val_CER", verbose = True)
tracker = callbacks.TensorBoard((model_config.model_path + "/logs").replace("\\","/"))
auto_lr = callbacks.ReduceLROnPlateau(monitor = "val_CER", factor=0.9, patience = 10, verbose = True)
save_model = callbacks.Model2onnx(saved_model_path = (os.path.join(model_path, datetime.strftime(datetime.now(), "%Y%m%d%H%M"),"model.pt").replace("\\","/")), input_shape = (1, model_config.height, model_config.width, 3), verbose = True, metadata = {"vocab": model_config.vocab})


train_struct = trainer.Trainer(model, optimizer, loss, metrics = [metric.CERMetric(model_config.vocab), metric.WERMetric(model_config.vocab)])

#train
train_struct.run(train_set, val_set, epochs=1000, callbacks = [ckpt, tracker, auto_lr, save_model, earlystop])

train_set.to_csv(os.path.join(model_config.model_path, "train.csv").replace("\\","/"))
val_set.to_csv(os.path.join(model_config.model_path, "val.csv").replace("\\","/"))