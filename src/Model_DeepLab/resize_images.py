import cv2
import os

source = '/home/oliver/segmentation/resize_data'

for root, dirs, filenames in os.walk(source):
    for f in filenames:
        fullpath = os.path.join(source, f)
        W = 1664
        H = 825
        oriimg = cv2.imread(fullpath)
        # height, width, depth = oriimg.shape
        # imgScale = W / width
        # newX, newY = oriimg.shape[1] * imgScale, oriimg.shape[0] * imgScale
        newimg = cv2.resize(oriimg, (W, H))
        cv2.imwrite('/home/oliver/segmentation/after_resize/' + f, newimg)
        # cv2.imwrite('../Model_HOG_SVM/model/test_image/' + f + 'New.jpg', newimg)
