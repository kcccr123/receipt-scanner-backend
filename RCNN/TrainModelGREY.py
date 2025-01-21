import os
import torch
import callbacks
import data
import dataUtils as du
import metric
import modelArc
import trainer
from config import ConfigFile
from datetime import datetime
import json as js
import numpy as np
import cv2
import typing
from skimage.filters import threshold_local
import image

def preprocess_image(image):
    # Load the image using OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (3,3), 0)
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    # # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # # Morph open to remove noise and invert image
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # # invert = 255 - opening

    return gray

#Specify path to main database
model_path = r"D:\Projects\reciept-scanner\RCNN\models"
database, vocab, max_len = [], set(), 0

#load second dataset
data_path = r"D:\photos\SORIE"

path = os.path.join(data_path, "train").replace("\\","/")
i = 1
while i <= 2:
    print("loading dataset")
    with open(os.path.join(path, "metadata.jsonl").replace("\\","/"), 'r') as file:
        for line in file:
            
            row = js.loads(line)
            img_path = os.path.join(path, row.get("file_name")).replace("\\","/")
            if os.path.exists(img_path):
                label = row.get("text").rstrip("\n")
                if len(label.split(' ')) <= 3 and label != "***":
                    vocab.update(list(label))
                    max_len = max(max_len, len(label))
                    database.append([img_path, label])
            else:
                print("image with path " + str(img_path) + " do not exist")
    if i == 1:
        print("dataset1 done")
    
    i += 1
    path = os.path.join(data_path, "test").replace("\\","/")

print("dataset1 done")

print("loading dataset")
with open(os.path.join(path, "testing.jsonl").replace("\\","/"), 'r') as file:
    for line in file:
        
        row = js.loads(line)
        img_path = os.path.join(path, row.get("file_name")).replace("\\","/")
        if os.path.exists(img_path):
            label = row.get("text").rstrip("\n")
            if len(label.split(' ')) <= 3 and label != "***":
                vocab.update(list(label))
                max_len = max(max_len, len(label))
                database.append([img_path, label])
        else:
            print("image with path " + str(img_path) + " do not exist")
print("dataset2 done")
print(len(database))

print("database, vocab, max_len, complete")

#normalize images and load them as an np array

#get the mean and std of dataset
# num_pixels = 0
# channel_sum = torch.tensor([0.0])
# channel_sum_squared = torch.tensor([0.0])

# for pair in database:
#     image_path = pair[0]
#     image = cv2.imread(image_path)
#     img = preprocess_image(image)

#     if img is None:
#         print(f"Warning: Could not read image at {image_path}. Skipping.")
#         continue
    
#     height, width = img.shape

#     num_pixels += height * width

#     channel_sum += torch.tensor(img.sum(), dtype=torch.float64)
#     channel_sum_squared += torch.tensor((img.astype(np.float64) ** 2).sum(), dtype=torch.float64)

# mean = channel_sum/num_pixels
# variance = (channel_sum_squared / num_pixels) - (mean**2)
# std = torch.sqrt(variance)
# print("Mean:", mean.numpy())
# print("Standard Deviation:", std.numpy())


class CVImage(image.Image):
    #Image class for storing image data and metadata (opencv based)

    init_successful = False

    def __init__(self, image: typing.Union[str, np.ndarray], method: int = cv2.IMREAD_COLOR,
                path: str = "", color: str = "RGB") -> None:
        super().__init__()
        
        if isinstance(image, str):#image path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image {image} not found.")

            self._image = cv2.imread(image)
            self.path = image
            self.color = "GREY"
            self._image = preprocess_image(self._image)

        elif isinstance(image, np.ndarray):
            self._image = image
            self.path = path
            self.color = color

        else:
            raise TypeError(f"Image must be either path to image or numpy.ndarray, not {type(image)}")

        self.method = method

        if self._image is None:
            return None
        
        self.init_successful = True

        # save width, height and channels
        self.width = self._image.shape[1]
        self.height = self._image.shape[0]
        self.channels = 1 if len(self._image.shape) == 2 else self._image.shape[2]

    @property
    def image(self) -> np.ndarray:
        return self._image

    @image.setter
    def image(self, value: np.ndarray):
        self._image = value

    @property
    def shape(self) -> tuple:
        return self._image.shape

    @property
    def center(self) -> tuple:
        return self.width // 2, self.height // 2

    def update(self, image: np.ndarray):
        if isinstance(image, np.ndarray):
            self._image = image

            # save width, height and channels
            self.width = self._image.shape[1]
            self.height = self._image.shape[0]
            self.channels = 1 if len(self._image.shape) == 2 else self._image.shape[2]

            return self

        else:
            raise TypeError(f"image must be numpy.ndarray, not {type(image)}")
        
    def RGB(self) -> np.ndarray:
        if self.color == "RGB":
            return self._image
        elif self.color == "BGR":
            return cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unknown color format {self.color}")
        
    def HSV(self) -> np.ndarray:
        if self.color == "BGR":
            return cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
        elif self.color == "RGB":
            return cv2.cvtColor(self._image, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def flip(self, axis: int = 0):
        #flip image along x or y axis
        if axis not in [0, 1]:
            raise ValueError(f"axis must be either 0 or 1, not {axis}")

        self._image = self._image[:, ::-1] if axis == 0 else self._image[::-1]

        return self

    def numpy(self) -> np.ndarray:
        return self._image
    
    def __call__(self) -> np.ndarray:
        return self._image
    
#create data loaders
model_config = ConfigFile(name = "CRNNFinal", path = model_path, lr=0.0005, bs=16, h=36, w=224)

model_config.vocab = "".join(vocab)
model_config.max_txt_len = max_len
model_config.save()

dataset_loader = data.DataLoader(dataset = database, batch_size = model_config.batch_size, 
                                 data_preprocessors = [image.ImageReader(CVImage)], 
                                 transformers = [du.ImageResizer(model_config.width, model_config.height), du.LabelIndexer(model_config.vocab), 
                                                 du.LabelPadding(padding_value = len(model_config.vocab), max_word_len = max_len)])# ,du.ImageShowCV2()


train_set, val_set = dataset_loader.split(split = 0.9)

train_set.augmentors = [
    # du.RandomBrightness(),
    du.RandomErodeDilate(),
    du.RandomSharpen(),
    # du.RandomRotate(angle=5),
    ]

#initialize model, optimizer, and loss
model = modelArc.CRNNGBIGGER(len(model_config.vocab))
loss = trainer.CTCLoss(blank = len(model_config.vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=model_config.lr)

if torch.cuda.is_available():
    model = model.cuda()
    print("CUDA Enabled...Training On GPU")

#initialze callbacks and trainer
earlystop = callbacks.EarlyStopping(monitor = "val_CER", patience = 38, verbose = True)
ckpt = callbacks.ModelCheckpoint((model_config.model_path + "/model.pt").replace("\\","/"), monitor = "val_CER", verbose = True)
tracker = callbacks.TensorBoard((model_config.model_path + "/logs").replace("\\","/"))
auto_lr = callbacks.ReduceLROnPlateau(monitor = "val_CER", factor=0.9, patience = 5, verbose = True)
save_model = callbacks.Model2onnx(saved_model_path = (os.path.join(model_path, datetime.strftime(datetime.now(), "%Y%m%d%H%M"),"model.pt").replace("\\","/")), input_shape = (1, 1, model_config.height, model_config.width), verbose = True, metadata = {"vocab": model_config.vocab})

train_struct = trainer.Trainer(model, optimizer, loss, metrics = [metric.CERMetric(model_config.vocab), metric.WERMetric(model_config.vocab)])

#train
train_struct.run(train_set, val_set, epochs=1000, callbacks = [ckpt, tracker, auto_lr, save_model, earlystop])

train_set.to_csv(os.path.join(model_config.model_path, "train.csv").replace("\\","/"))
val_set.to_csv(os.path.join(model_config.model_path, "val.csv").replace("\\","/"))