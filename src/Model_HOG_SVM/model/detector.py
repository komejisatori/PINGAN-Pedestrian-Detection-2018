from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
import cv2
from skimage import color
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os
from model.config import *
import numpy as np


def train_svm():
    pos_feat_path = '../data/features/pos'
    neg_feat_path = '../data/features/neg'

    # Classifiers supported
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print(np.array(fds).shape, len(labels))
    clf = None
    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print("Training a Linear SVM Classifier")
        clf.fit(fds, labels)
        print("Classifier saved to {}".format(model_path))
    return clf


def sliding_window(image, window_size, step_size):
    """
    This function returns a patch of the input 'image' of size
    equal to 'window_size'. The first image returned top-left
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.
    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window
    step_size - incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    """
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])


def detector(filename, clf):
    im = cv2.imread(filename)
    im = imutils.resize(im, width=min(400, im.shape[1]))
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 1.25

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0

    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # The list contains detections at the current scale
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window = color.rgb2gray(im_window)
            fd = hog(im_window, orientations, pixels_per_cell, cells_per_block)

            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)

            if pred == 1:

                if clf.decision_function(fd) > 0.5:
                    detections.append(
                        (int(x * (downscale ** scale)), int(y * (downscale ** scale)), clf.decision_function(fd),
                         int(min_wdw_sz[0] * (downscale ** scale)),
                         int(min_wdw_sz[1] * (downscale ** scale))))

        scale += 1

    clone = im.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)

    res = filename.split("\\")[1]

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
    print("shape, ", pick.shape)

    for (xA, yA, xB, yB) in pick:
        x = xA
        y = yA
        w = xB - xA
        h = yB - yA
        res += " "
        res += str(1)
        res += " "
        res += str(x)
        res += " "
        res += str(y)
        res += " "
        res += str(w)
        res += " "
        res += str(h)

        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # plt.axis("off")
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.title("Raw Detection before NMS")
    # plt.show()

    # plt.axis("off")
    # plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    # plt.title("Final Detections after applying NMS")
    # plt.show()
    return res


def test_folder(foldername):
    filenames = glob.iglob(os.path.join(foldername, '*'))
    clf = train_svm()
    for filename in filenames:
        detector(filename, clf)


def test_all(foldername):
    filenames = glob.iglob(os.path.join(foldername, '*'))
    clf = train_svm()

    with open('image_train.txt', 'a') as the_file:
        for filename in filenames:
            the_file.write(detector(filename, clf))


if __name__ == '__main__':
    #foldername = 'test_image'
    # test_folder(foldername)
    foldername = '/media/workspace/bgong/data/WIDER_Pedestrian_Challenge/data/train'
    test_all(foldername)
