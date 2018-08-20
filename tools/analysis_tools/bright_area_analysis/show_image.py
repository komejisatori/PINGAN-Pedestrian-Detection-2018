#!/usr/bin/env python3
"""
show_image.py: output image ids from fp/tp/fn analysis, given ranges of brightness & area.
"""
__author__ = "Zhuoran Wu"
__email__ = "zhuoran.wu@pactera.com"
__credits__ = ["Jiaqi Cai", "Zhuoran Wu", "Kun Li"]

import pickle
import numpy as np


def extract_one_column(submission_label, column_num):
    result_list = list()

    for key, value in submission_label.items():
        if value:
            l = np.array(value)
            result_list.extend(l[:, column_num])

    return result_list


def load_submission_bright_area_pkl(submission_bright_area_pkl_file):
    with open(submission_bright_area_pkl_file, 'rb') as fp:
        sub_label = pickle.load(fp)
    fp.close()
    return sub_label


def get_image_ids(pkl_filename, bright_range, area_range, set, score, iou, confidence):
    submission_bright_area_label = load_submission_bright_area_pkl(pkl_filename)

    for key, value in submission_bright_area_label.items():
        for idx, item in enumerate(value):
            print(item)
            if bright_range[0] <= item[5] <= bright_range[1] and area_range[0] <= item[6] <= area_range[1]:
                print(key)


if __name__ == '__main__':

    scores = [0.59]
    ious = [0.9]
    # ious = [0.6, 0.7, 0.8, 0.9]
    confidences = [0.4]
    # confidences = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # types = ['tp', 'fp', 'fn']
    types = ['fp']

    bright_range = [0.06, 0.1]
    area_range = [0.16, 0.18]

    BASE_DIR = './'

    for score in scores:
        for iou in ious:
            for confidence in confidences:
                for typ in types:
                    filename = str(score) + '_' + typ + '_' + str(iou) + '_' \
                                         + str(confidence) + '_val_result_bright_area_ignore.pkl'
                    get_image_ids(filename, bright_range, area_range, typ.upper(), score, iou, confidence)

