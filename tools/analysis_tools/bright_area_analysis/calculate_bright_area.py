#!/usr/bin/env python3
"""
Calculate_bright_area.py: output pkl of bright and area information, given extracted fp/tp/fn result pkl.
"""
__author__ = "Zhuoran Wu"
__email__ = "zhuoran.wu@pactera.com"
__credits__ = ["Jiaqi Cai", "Zhuoran Wu", "Kun Li"]

import os
import pickle
import numpy as np
import copy
from PIL import Image

submission_label = dict()
submission_bright = list()
submission_area = list()
total_labels = dict()


def dump_pkl(sub, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(sub, fp)
    fp.close()


def bbox_area(single_box):
    """

    Return the bounding box Area, w * h.
    :param single_box: (c, x, y, w, h)
    :return: Area
    """
    return single_box[3] * single_box[4]


def image_brightness(image):
    """

    Given an image or a region of image: ndarray, return brightness [0,1] of an image.

    :param image: Original Image or Box Area as ndArray with (W * H * 3) Dimensions
    :return: float brightness
    """
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


def bbox_brightness(image, box):
    """

    Given an image and a bounding box. Return the brightness of the box region in the image.

    :param image: Original Image as ndArray with (W * H * 3) Dimensions
    :param box: one box on the imagepocky as list (c, x, y, w, h)
    :return: float brightness
    """
    area = (box[1], box[2], box[1] + box[3], box[2] + box[4])
    box_area = image.crop(area)
    return image_brightness(box_area)


def read_image_submission(submission):
    """
    Read Submission File with format:

    img00001.jpg c x y w h
    img00001.jpg c x y w h

    :param submission: Submission File
    :return:
    """
    count = 0
    image = None
    with open(submission, "r") as fr:
        for idx, line in enumerate(fr):
            count += 1
            print(count)
            parts = line.rstrip("\r\n").split(" ")
            single_box = list()
            single_box.append(float(parts[1]))
            single_box.append(float(parts[2]))
            single_box.append(float(parts[3]))
            single_box.append(float(parts[4]))
            single_box.append(float(parts[5]))

            im_file = os.path.join(IMG_DIR, parts[0])

            if parts[0] in submission_label.keys():
                submission_label[parts[0]].append(single_box)

                bright_box = bbox_brightness(image, single_box)
                single_box.append(bright_box)
                submission_bright.append(bright_box)

                area_box = bbox_area(single_box)
                single_box.append(area_box)
                submission_area.append(area_box)
            else:
                image = Image.open(im_file)

                bright_box = bbox_brightness(image, single_box)
                single_box.append(bright_box)
                submission_bright.append(bright_box)

                area_box = bbox_area(single_box)
                single_box.append(area_box)
                submission_area.append(area_box)
                submission_label[parts[0]] = list()
                submission_label[parts[0]].append(single_box)

    fr.close()


def read_image_annotation(annotation, img_dir):
    """
    Read Annotation File with format:

    img00001.jpg c x y w h c x y w h
    img00002.jpg

    :param annotation:
    :return:
    """
    bbox_count = 0
    img_count = 0
    with open(annotation, "r") as fw:
        for idx, line in enumerate(fw):
            img_count += 1
            print("Deal with Image: " + str(img_count))

            parts = line.rstrip("\r\n").split(" ")
            if len(parts) <= 1:
                continue
            else:
                im_file = os.path.join(img_dir, parts[0])
                image = Image.open(im_file)

                count = int((len(parts) - 1) / 5)
                bbox_count += count
                total_labels[parts[0]] = list()
                for i in range(count):
                    # i * 5 + 1
                    single_box = list()
                    for j in range(5):
                        # j: 0 1 2 3 4
                        single_box.append(float(parts[i * 5 + 1 + j]))

                    bright_box = bbox_brightness(image, single_box)
                    single_box.append(bright_box)

                    area_box = bbox_area(single_box)
                    single_box.append(area_box)

                    total_labels[parts[0]].append(single_box)
    fw.close()
    return bbox_count


def load_submission_pkl(submission_pkl_file):
    with open(submission_pkl_file, 'rb') as fp:
        sub_label = pickle.load(fp)
    fp.close()
    return sub_label


def get_bright_area_from_pkl(sub_label, img_dir):
    sub_label_bright_area = copy.deepcopy(sub_label)
    count = 0
    for key, value in sub_label.items():
        im_file = os.path.join(img_dir, key)
        image = Image.open(im_file)

        count += 1
        if count % 1000 == 0:
            print('Deal with image: ' + str(count))

        for idx, single_box in enumerate(value):
            bright_box = bbox_brightness(image, single_box)
            submission_bright.append(bright_box)
            area_box = bbox_area(single_box)
            submission_area.append(area_box)
            sub_label_bright_area[key][idx] = np.append(sub_label_bright_area[key][idx], bright_box)
            sub_label_bright_area[key][idx] = np.append(sub_label_bright_area[key][idx], area_box)

    return sub_label_bright_area


if __name__ == '__main__':

    ious = [0.5]
    confidences = [0.4]
    types = ['fn']
    BASE_DIR = '../img/'
    DATASET_FLAG = 'val'
    IMG_DIR = os.path.join(BASE_DIR, 'val')

    ANNOTATION_FILE = 'val_annotations.txt'

    for typ in types:
        for iou in ious:
            for confidence in confidences:
                SUBMISSION_PKL = '0.59_' + typ + '_' + str(iou) + '_' + str(confidence) + '_val_result_ignore.pkl'

                submission_label_bright_area = get_bright_area_from_pkl(load_submission_pkl(SUBMISSION_PKL), IMG_DIR)

                dump_pkl(submission_label_bright_area, '0.59_' + typ + '_' + str(iou)
                         + '_' + str(confidence) + '_val_result_bright_area_ignore.pkl')

                submission_label.clear()
                submission_bright.clear()
                total_labels.clear()
                submission_area.clear()

    # read_image_annotation(os.path.join(BASE_DIR, ANNOTATION_FILE), IMG_DIR)

    # dump_pkl(total_labels, 'val_result_bright_area_2.pkl')
