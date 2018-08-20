#!/usr/bin/env python3
"""
draw_graph.py: draw brightness & area count 2D histogram, given bright & area pkl.
"""
__author__ = "Zhuoran Wu"
__email__ = "zhuoran.wu@pactera.com"
__credits__ = ["Jiaqi Cai", "Zhuoran Wu", "Kun Li"]

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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


def draw_graph(base_dir, submission_file, score=0.59, set='Val', confidence=0.4, iou=0.5):
    submission_bright_area_label = load_submission_bright_area_pkl(os.path.join(base_dir,
                                                                                submission_file))

    bright_result = extract_one_column(submission_bright_area_label, 5)
    area_result = extract_one_column(submission_bright_area_label, 6)

    print(set + ' ' + str(confidence) + ' ' + str(len(bright_result)))
    print(set + ' ' + str(confidence) + ' ' + str(len(area_result)))

    area_result = np.asarray(area_result)
    area_result = (area_result - np.min(area_result)) / np.ptp(area_result)

    """
    plt.hist(bright_result, bins=50)
    plt.ylabel('Count')
    plt.xlabel('Brightness')
    plt.show()

    plt.hist(area_result, bins=50)
    # plt.xlim([0, 0.03])
    plt.ylabel('Count')
    plt.xlabel('Area')
    plt.show()
    """

    color_bar_min = 0
    color_bar_max = 50
    bins = 50

    fig, ax = plt.subplots()
    norm = matplotlib.colors.Normalize(vmin=color_bar_min, vmax=color_bar_max)
    counts, xedges, yedges, im = ax.hist2d(bright_result,
                                           area_result,
                                           bins=bins,
                                           norm=norm,
                                           cmin=0)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks(np.linspace(color_bar_min, color_bar_max, 10, dtype=int))
    cbar.set_label("Count")
    # cbar.set_ticklabels(('160', '180', '200', '220', '240', '260', '280', '300'))

    title = "Brightness & Area with Count - Score " + str(score) \
            + " - Confidence " + str(confidence) + " - IOU " + str(iou) + " - " + set

    plt.title(title)
    # plt.hist2d(bright_result, area_result, bins=50, cmin=0)
    plt.ylim([0.0, 0.5])
    # plt.xlim([0.0, 1.0])
    plt.ylabel('Area')
    plt.xlabel('Brightness')
    plt.savefig(os.path.join('/home/oliver/img/FFN', title + '.png'))
    plt.show()


if __name__ == '__main__':

    scores = [0.59]
    # ious = [0.9]
    ious = [0.5]
    confidences = [0.4]
    # confidences = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # types = ['tp', 'fp', 'fn']
    types = ['fn']

    BASE_DIR = './'

    for score in scores:
        for iou in ious:
            for confidence in confidences:
                for typ in types:
                    filename = str(score) + '_' + typ + '_' + str(iou) + '_' \
                                         + str(confidence) + '_val_result_bright_area_ignore.pkl'
                    draw_graph(BASE_DIR, filename, score=score, set=typ.upper(),
                               confidence=confidence, iou=iou)

    """
    SUBMISSION_TP_BRIGHT_AREA_FILENAME = str(score) + '_tp_' + str(iou) + '_' \
                                         + str(confidence) + '_val_result_bright_area_ignore.pkl'
    SUBMISSION_FP_BRIGHT_AREA_FILENAME = str(score) + '_fp_' + str(iou) + '_' \
                                         + str(confidence) + '_val_result_bright_area_ignore.pkl'
    SUBMISSION_FN_BRIGHT_AREA_FILENAME = str(score) + '_fn_' + str(iou) + '_' \
                                         + str(confidence) + '_val_result_bright_area_ignore.pkl'
    VAL_BRIGHT_AREA_FILENAME = 'val_result_bright_area_2.pkl'

    # draw_graph(BASE_DIR, VAL_BRIGHT_AREA_FILENAME, 1, 'Val', 1, 1)
    
    draw_graph(BASE_DIR, SUBMISSION_TP_BRIGHT_AREA_FILENAME, score=score, set='TP',
               confidence=confidence, iou=iou)
    draw_graph(BASE_DIR, SUBMISSION_FP_BRIGHT_AREA_FILENAME, score=score, set='FP',
               confidence=confidence, iou=iou)
    
    draw_graph(BASE_DIR, SUBMISSION_FN_BRIGHT_AREA_FILENAME, score=score, set='FN',
               confidence=confidence, iou=iou)
    """
