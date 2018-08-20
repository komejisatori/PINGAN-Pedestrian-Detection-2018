#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract TP & FP & FN in a certain result file.

 extract_result.py:

 Given a submission file and a validation file.
 Output True Positive and False Positive Results.
"""
__author__ = "Zhuoran Wu"
__email__ = "zhuoran.wu@pactera.com"
__credits__ = ["Jiaqi Cai", "Zhuoran Wu", "Kun Li"]

import numpy as np
import os
import pickle
from operator import itemgetter

# Global Variables to store the results of tp and fp
tp_result_dict = dict()
fp_result_dict = dict()
fn_result_dict = dict()


def judge_overlap(pbox, ignore_box):
    """
    Return the overlap area for the predict box and ignore box.

    :param pbox:
    :param ignore_box:
    :return:
    """
    overlap = []
    delete = []
    for p in pbox:
        pl = min(p[0], p[2])
        pr = max(p[0], p[2])
        pb = min(p[1], p[3])
        pt = max(p[1], p[3])
        s_p = (pr - pl) * (pt - pb)
        s_lap = -0.01
        for c in ignore_box:
            cl = min(c[0], c[2])
            cr = max(c[0], c[2])
            cb = min(c[1], c[3])
            ct = max(c[1], c[3])
            if not (cr < pl or cl > pr or ct < pb or cb > pt):
                s_lap += (min(cr, pr) - max(cl, pl)) * (min(ct, pt) - max(cb, pb))
        if s_lap > 0:
            overlap.append([p, s_lap / s_p])
    for o in overlap:
        if o[1] > 0.5:
            delete.append(o[0])
    remain_id = [p for p in pbox if p not in delete]
    return remain_id


def parse_ignore_file(ignore_file):
    """
    Return Ignore Dict
    :param ignore_file:
    :return:
    """
    with open(ignore_file, 'r') as f:
        lines = f.readlines()
    ig = [x.strip().split() for x in lines]
    ignore = {}
    for item in ig:
        key = item[0]
        ignore_num = (len(item) - 1) / 4
        bbox = []
        for i in range(int(ignore_num)):
            b = []
            b.append(int(item[1 + 4 * i]))
            b.append(int(item[2 + 4 * i]))
            b.append(int(item[1 + 4 * i]) + int(item[3 + 4 * i]))
            b.append(int(item[2 + 4 * i]) + int(item[4 + 4 * i]))
            bbox.append(b)
        ignore[key] = bbox
    return ignore


def parse_submission(submission_file, ignore_file):
    """
    Read submission file and ignore file.

    :param submission_file:
    :param ignore_file:
    :return:
    """
    ignore_zone = parse_ignore_file(ignore_file)
    ignore_keys = ignore_zone.keys()
    with open(submission_file, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split() for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = []
    for x in splitlines:
        bb = []
        bb.append(float(x[2]))
        bb.append(float(x[3]))
        bb.append(float(x[2]) + float(x[4]))
        bb.append(float(x[3]) + float(x[5]))
        BB.append(bb)

    sub_key = []
    for x in image_ids:
        if x not in sub_key:
            sub_key.append(x)

    final_confidence = []
    final_ids = []
    final_BB = []

    for key in sub_key:
        find = [i for i, v in enumerate(image_ids) if v == key]
        BB_sub = [BB[i] for i in find]
        confid_sub = [confidence[i] for i in find]
        if key in ignore_keys:
            ignore_bbox = ignore_zone[key]
            bbox_remain = judge_overlap(BB_sub, ignore_bbox)
            find_remain = []
            for i, v in enumerate(BB_sub):
                if v in bbox_remain:
                    find_remain.append(i)
            confid_remain = [confid_sub[i] for i in find_remain]
            BB_sub = bbox_remain
            confid_sub = confid_remain
        ids_sub = [key] * len(BB_sub)
        final_ids.extend(ids_sub)
        final_confidence.extend(confid_sub)
        final_BB.extend(BB_sub)

    final_BB = np.array(final_BB)
    final_confidence = np.array(final_confidence)
    sorted_ind = np.argsort(-final_confidence)
    final_BB = final_BB[sorted_ind, :]
    final_ids = [final_ids[x] for x in sorted_ind]
    return final_ids, final_BB


def parse_gt_annotation(gt_file, ignore_file):
    """
    Read ground truth and ignore files.

    :param gt_file:
    :param ignore_file:
    :return:
    """
    ignore_zone = parse_ignore_file(ignore_file)
    ignore_keys = ignore_zone.keys()
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    info = [x.strip().split() for x in lines]
    gt = {}
    for item in info:
        bbox = []
        bbox_num = (len(item) - 1) / 5
        for i in range(int(bbox_num)):
            b = []
            b.append(int(item[2 + 5 * i]))
            b.append(int(item[3 + 5 * i]))
            b.append(int(item[2 + 5 * i]) + int(item[4 + 5 * i]))
            b.append(int(item[3 + 5 * i]) + int(item[5 + 5 * i]))
            bbox.append(b)
        if item[0] in ignore_keys:
            ignore_bbox = ignore_zone[item[0]]
            bbox_remain = judge_overlap(bbox, ignore_bbox)
        else:
            bbox_remain = bbox
        gt[item[0]] = np.array(bbox_remain)
    return gt


def compute_ap(rec, prec):
    """
    Compute Average Precision according to recall & precision.

    :param rec:
    :param prec:
    :return:
    """
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def write_file(sub, file_name):
    """
    Write TP or FP dict to a file.

    :param sub:
    :param file_name:
    :return:
    """
    with open(file_name, "w") as fw:
        for key, value in sub.items():
            # print(value)
            for box in value:
                line = key + " " + str(box[0]) + " " + str(box[1]) + " " \
                       + str(box[2]) + " " + str(box[3]) + " " + str(box[4]) + "\n"
                fw.write(line)
    fw.close()


def dump_pkl(sub, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(sub, fp)
    fp.close()


def pedestrian_eval(input, gt_file, ignore_file, ove, score, confidence):
    """
    Main evalution function.

    :param input:
    :param gt_file:
    :param ignore_file:
    :return:
    """
    global tp_result_dict
    global fn_result_dict
    global fp_result_dict

    gt = parse_gt_annotation(gt_file, ignore_file)
    image_ids, BB = parse_submission(input, ignore_file)
    aap = []
    tp_count = 0

    npos = 0
    recs = {}
    for key in gt.keys():
        det = [False] * len(gt[key])
        recs[key] = {'bbox': gt[key], 'det': det}
        npos += len(gt[key])
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] not in recs.keys():
            raise KeyError(
                "Can not find image {} in the groundtruth file, "
                "did you submit the result file for the right dataset?".format(
                    image_ids[d]))

    for d in range(nd):
        R = recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ove:
            if not R['det'][jmax]:
                if image_ids[d] in tp_result_dict.keys():
                    tp_result_dict[image_ids[d]].append(np.insert(bb, 0, 1.0, axis=0))
                else:
                    tp_result_dict[image_ids[d]] = list()
                    tp_result_dict[image_ids[d]].append(np.insert(bb, 0, 1.0, axis=0))
                tp[d] = 1.
                tp_count += 1
                R['det'][jmax] = 1
            else:
                if image_ids[d] in fp_result_dict.keys():
                    fp_result_dict[image_ids[d]].append(np.insert(bb, 0, 1.0, axis=0))
                else:
                    fp_result_dict[image_ids[d]] = list()
                    fp_result_dict[image_ids[d]].append(np.insert(bb, 0, 1.0, axis=0))
                fp[d] = 1.
                R['det'][jmax] = 2
        else:
            if image_ids[d] in fp_result_dict.keys():
                fp_result_dict[image_ids[d]].append(np.insert(bb, 0, 1.0, axis=0))
            else:
                fp_result_dict[image_ids[d]] = list()
                fp_result_dict[image_ids[d]].append(np.insert(bb, 0, 1.0, axis=0))
            fp[d] = 1.

        # Get False Negative in Ground Truth

    for key, value in recs.items():

        idx = [i for i, v in enumerate(value['det']) if v == False]
        fn = itemgetter(idx)(value['bbox'])

        if key in fn_result_dict.keys():
            for i, v in enumerate(fn):
                fn_result_dict[key].append(np.insert(v, 0, 1.0, axis=0))
        else:
            fn_result_dict[key] = list()
            for i, v in enumerate(fn):
                fn_result_dict[key].append(np.insert(v, 0, 1.0, axis=0))

    # File name Format:
    dump_pkl(tp_result_dict, str(score) + "_tp_" + str(ove) + "_" + str(confidence) + "_val_result_ignore.pkl")
    dump_pkl(fp_result_dict, str(score) + "_fp_" + str(ove) + "_" + str(confidence) + "_val_result_ignore.pkl")
    dump_pkl(fn_result_dict, str(score) + "_fn_" + str(ove) + "_" + str(confidence) + "_val_result_ignore.pkl")

    print(str(score) + "_tp|fp|fn_" + str(ove) + "_" + str(confidence) + "_val_result_ignore Done")

    tp_result_dict.clear()
    fn_result_dict.clear()
    fp_result_dict.clear()

    return aap


def wider_ped_eval(input, gt, ignore_file, ove, score, confidence):
    aap = pedestrian_eval(input, gt, ignore_file, ove, score, confidence)
    mAP = np.average(aap)
    return mAP


if __name__ == '__main__':

    BASE_DIR = '../img'

    GT_FILENAME = 'val_annotations.txt'
    IGNORE_FILENAME = 'pedestrian_ignore_part_val.txt'

    gt_file = os.path.join(BASE_DIR, GT_FILENAME)
    ignore_file = os.path.join(BASE_DIR, IGNORE_FILENAME)

    oves = [0.5]
    scores = [0.59]
    confidences = [0.4]

    for ove in oves:
        for score in scores:
            for confidence in confidences:
                SUBMIT_FILENAME = '0.59571483026248_' + str(confidence) + '.txt'
                submit_file = os.path.join(BASE_DIR, SUBMIT_FILENAME)
                mAP = wider_ped_eval(submit_file, gt_file, ignore_file, ove, score, confidence)
