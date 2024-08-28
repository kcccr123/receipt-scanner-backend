import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from PIL import Image

def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def plot_rgb(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def plot_gray(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(image, cmap='Greys_r')
# approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)

def get_receipt_contour(contours):    
    for c in contours:
        approx = approximate_contour(c)
        # four points = rectangle
        if len(approx) == 4:
            return approx

def wrap_perspective(img, rect):
    # top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255

"""CALL THIS FUCTION BELOW, WHERE image = cv2.imread(path) AND WILL RETURN AN ARRAY THAT YOU CAN USE Image.fromarray TO TURN TO IMAGE AND SAVE"""
def fix_angle(image):
    original = image.copy()
    resize_ratio = 500 / image.shape[0]

    def contour_to_rect(contour):
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect / resize_ratio

    image = opencv_resize(image, resize_ratio)
    # Convert to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plot_gray(gray)
    # Get rid of noise 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect white regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)

    edged = cv2.Canny(dilated, 100, 200, apertureSize=3)
    plot_gray(edged)
    # Detect all contours 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Get 10 largest contours
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    receipt_contour = get_receipt_contour(largest_contours)
    image_with_receipt_contour = cv2.drawContours(image.copy(), [receipt_contour], -1, (0, 255, 0), 2)
    plot_rgb(image_with_receipt_contour)
    result = wrap_perspective(original.copy(), contour_to_rect(receipt_contour))
    greyscale_output = bw_scanner(result)
    return greyscale_output