import cv2
import os

directory = os.fsencode('/home/oliver/segmentation/resize_data')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    W = 1480.
    oriimg = cv2.imread(filename)
    height, width, depth = oriimg.shape
    imgScale = W / width
    newX, newY = oriimg.shape[1] * imgScale, oriimg.shape[0] * imgScale
    newimg = cv2.resize(oriimg, (int(newX), int(newY)))
    cv2.imwrite('/home/oliver/segmentation/after_resize/' + filename, newimg)
