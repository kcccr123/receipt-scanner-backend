import os
import numpy
import cv2 as cv
import pandas as pd

def bbox2img(index: int, img_path: str, x_min: int, y_min: int, x_max: float, y_max: float, save_path: str) -> None:

    img = cv.imread(img_path)
    # cv.imshow('image',img)
    bbox = img[y_min:y_max, x_min:x_max,]
    s_path = os.path.join(save_dir, str(str(i) + ".jpg")).replace("\\","/")
    cv.imwrite(s_path, bbox)

    return None

csv_path = r"D:\photos\RCNN_new_data\Original\annotations.csv"
img_dir = r"D:\photos\RCNN_new_data\Original"

csv = pd.read_csv(csv_path)
for i in range(0, csv.shape[0]):
    if csv.loc[i].at["class"] == "item":
        save_dir = r"D:\photos\RCNN_new_data\bboxes\items"
    else:
        save_dir = r"D:\photos\RCNN_new_data\bboxes\other"
    bbox2img(i, os.path.join(img_dir, csv.loc[i].at["filename"]).replace("\\","/"), csv.loc[i].at["xmin"], csv.loc[i].at["ymin"], csv.loc[i].at["xmax"], csv.loc[i].at["ymax"], save_dir)




