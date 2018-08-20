#!/usr/bin/env python3
"""
Ensembling methods for object detection.
"""
__author__ = "Zhuoran Wu"
__email__ = "zhuoran.wu@pactera.com"
__credits__ = ["Jiaqi Cai", "Zhuoran Wu", "Kun Li"]

import argparse
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description='BBOX Ensemble')
    parser.add_argument(
        '--result',
        dest='result_files',
        help='All Results Files to be ensembles (/path/to/)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output',
        dest='output_files',
        help='Output Results Files after ensemble (/path/to/)',
        default="ensemble.txt",
        type=str
    )
    parser.add_argument(
        '--img-id-list',
        dest='id_list_file',
        help='All the image name file (/path/to/)',
        default="val.txt",
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def general_ensemble(dets, iou_thresh=0.5, weights=None):
    """
    General Ensemble - find overlapping boxes of the same class and average their positions
    while adding their confidences. Can weigh different detectors with different weights.
    No real learning here, although the weights and iou_thresh can be optimized.
    Input:
     - dets : List of detections. Each detection is all the output from one detector, and
              should be a list of boxes, where each box should be on the format
              [box_x, box_y, box_w, box_h, class, confidence] where box_x and box_y
              are the left-up corner coordinates, box_w and box_h are width and height resp.
              The values should be floats, except the class which should be an integer.
     - iou_thresh: Threshold in terms of IOU where two boxes are considered the same,
                   if they also belong to the same class.

     - weights: A list of weights, describing how much more some detectors should
                be trusted compared to others. The list should be as long as the
                number of detections. If this is set to None, then all detectors
                will be considered equally reliable. The sum of weights does not
                necessarily have to be 1.
    Output:
        A list of boxes, on the same format as the input. Confidences are in range 0-1.

    :param dets:
    :param iou_thresh:
    :param weights:
    :return:
    """
    assert (type(iou_thresh) == float)

    ndets = len(dets)

    if weights is None:
        w = 1 / float(ndets)
        weights = [w] * ndets
    else:
        assert (len(weights) == ndets)

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()

    for idet in range(0, ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue

            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = compute_iou(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox

                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox, w))
                    used.append(bestbox)

            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w * b[0]
                    yc += w * b[1]
                    bw += w * b[2]
                    bh += w * b[3]
                    conf += w * b[5]

                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                new_box = [xc, yc, bw, bh, box[4], conf]
                out.append(new_box)
    return out


def get_coords(box):
    """
    get (x1, x2, y1, y2) from (x, y, w, h)
    :param box:
    :return:
    """
    x1 = float(box[0])
    y1 = float(box[1])
    x2 = float(box[0] + box[2])
    y2 = float(box[1] + box[3])
    return x1, x2, y1, y2


def compute_iou(box1, box2):
    x11, x12, y11, y12 = get_coords(box1)
    x21, x22, y21, y22 = get_coords(box2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou


def read_files(path):
    idx = 0
    imgs = [{}, {}]
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as fp:
            for cnt, line in enumerate(fp):
                bboxs = str(line.rstrip("\r\n")).split(" ")
                if bboxs[0] in imgs[idx]:
                    single_detetion = list()
                    single_detetion.append(float(bboxs[2]))
                    single_detetion.append(float(bboxs[3]))
                    single_detetion.append(float(bboxs[4]))
                    single_detetion.append(float(bboxs[5]))
                    single_detetion.append(float(0))  # Class does not metter
                    single_detetion.append(float(bboxs[1]))

                    imgs[idx][str(bboxs[0])].append(single_detetion)
                else:
                    imgs[idx][str(bboxs[0])] = []

                    single_detetion = list()
                    single_detetion.append(float(bboxs[2]))
                    single_detetion.append(float(bboxs[3]))
                    single_detetion.append(float(bboxs[4]))
                    single_detetion.append(float(bboxs[5]))
                    single_detetion.append(int(0))  # Class does not metter
                    single_detetion.append(float(bboxs[1]))

                    imgs[idx][str(bboxs[0])].append(single_detetion)

        idx += 1
        fp.close()

    return imgs


if __name__ == "__main__":
    args = parse_args()

    # num_results = len(os.listdir(args.result_files)) # We only need 2

    images = read_files(args.result_files)
    results = dict()
    vals = dict()

    with open(args.id_list_file, 'r') as fp:
        for cnt, line in enumerate(fp):
            vals[line.rstrip("\n\r")] = list()
            results[line.rstrip("\n\r")] = list()

    for key, value in vals.items():
        if key in images[0].keys() and key in images[1].keys():
            det = [[], []]
            det[0] = images[0][key]
            det[1] = images[1][key]
            ens = general_ensemble(det)

        elif key in images[0].keys() and key not in images[1].keys():
            # Keep image[0]
            ens = images[0][key]

        elif key not in images[0].keys() and key in images[1].keys():
            ens = images[1][key]

        else:
            # Not in any file
            pass

        results[key] = ens

    # Write results to file
    with open(args.output_files, "w") as fw:
        for k, v in results.items():
            for item in v:
                result_string = str(k) + " " + \
                                str(item[5]) + " " + \
                                str(item[0]) + " " + \
                                str(item[1]) + " " + \
                                str(item[2]) + " " + \
                                str(item[3]) + "\n"
                fw.write(result_string)

        fw.close()
