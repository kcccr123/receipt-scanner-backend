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

#Specify path to main database
data_dir = ""
model_path = ""

database, vocab, max_len = [], set(), 0

num_data = len(os.path.join(data_dir).replace("\\","/"))

print("dataset of" + num_data + "images")

for id in range(0, num_data/2):

    img_path = os.path.join(data_dir, str(id), ".jpg").replace("\\","/")

    labels = ground_truths.split()
    
    if os.path.exists(img_path):
        ground_truths = open(os.path.join(data_dir, str(id), ".txt").replace("\\","/")).readlines()
        for line in tqdm(ground_truths):
            if not line.isspace():
                label = line.rstrip("\n")
                database.append([img_path, label])
                vocab.update(list(label))
                max_len = max(max_len, len(label))
    else:
        print("image with index" + id + "do not exist")

print("database, vocab, max_len, complete")
    
config = modelArc.ConfigFile(name = "CRNN1", path = model_path)

config.vocab = "".join(vocab)
config.max_txt_len = max_len
config.save()

dataset_loader = data.DataLoader(dataset = database, batch_size = config.batch_size, 
                                 data_preprocessors = [image.ImageReader(image.ImageReader)], 
                                 transformers = [du.ImageResizer(config.width, config.height), du.LabelIndexer(config.vocab), 
                                                 du.LabelPadding(padding_value = len(config.vocab), max_word_len = max_len)])

train_set, val_set = dataset_loader.split()

train_set.augmentors = [
    du.RandomBrightness(),
    du.RandomErodeDilate(),
    du.RandomSharpen(),
    du.RandomRotate(angle=10),
    ]

model = modelArc.CRNN1(len(config.vocab))
loss = trainer.CTCLoss(blank = len(config.vocab))
optimizer = torch.optim.Adam(model.parameters, lr=config.lr)

if torch.cuda.is_available():
    model = model.cuda()
    print("CUDA Enabled...Training On GPU")

earlystop = callbacks.EarlyStopping(monitor = "val_cer", patience = 20, verbose = True)
ckpt = callbacks.ModelCheckpoint(config.model_path + "/model.pt", monitor = "val_cer", verbose = True)
tracker = callbacks.TensorBoard(config.model_path + "/logs")
auto_lr = callbacks.ReduceLROnPlateau(monitor = "val_cer", factor=0.9, patience = 10, verbose = True)
save_model = callbacks.Model2onnx(saved_model_path = model_path + "/model.pt", input_shape = (1, config.height, config.width, 3), verbose = True, metadata = {"vocab": config.vocab})

train_struct = trainer.Trainer(model, optimizer, loss, metrics = [metric.CERMetric(config.vocab), metric.WERMetric(config.vocab)])

train_struct.run(train_set, val_set, epochs=5, callbacks = [earlystop, ckpt, tracker, auto_lr, save_model])

train_set.to_csv(os.path.join(config.model_path, "train.csv").replace("\\","/"))
val_set.to_csv(os.path.join(config.model_path, "val.csv").replace("\\","/"))
