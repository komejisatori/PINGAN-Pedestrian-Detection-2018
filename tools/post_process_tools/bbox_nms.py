import numpy as np
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='BBOX NMS')
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
        help='Output Results Files after ensemble (/path/to/) without txt',
        default="ensemble",
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='threshes',
        nargs='*',
        default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help='Thresh NMS, a list. [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def non_max_suppression_fast(boxes, probs=None, overlapThresh=0.3):
    """

    :param boxes:
    :param probs:
    :param overlapThresh:
    :return:
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick]


def get_coordinate(x, y, w, h):
    return x, y, x + w, y + h


def read_bbox(submission_file):
    all_bbox = dict()
    with open(submission_file, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            parts = line.split(" ")
            x1, y1, x2, y2 = get_coordinate(float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
            if parts[0] in all_bbox.keys():
                all_bbox[parts[0]].append((x1, y1, x2, y2, float(parts[1])))
            else:
                all_bbox[parts[0]] = list()
                all_bbox[parts[0]].append((x1, y1, x2, y2, float(parts[1])))

    fr.close()
    return all_bbox


def bbox_to_images(bbox):
    images = list()
    for key, value in bbox.items():
        one_image = (key, np.asarray(value))
        images.append(one_image)

    return images


if __name__ == "__main__":
    args = parse_args()

    THRESH = [float(i) for i in args.threshes]
    result_file = args.result
    output_file = args.output

    # loop over the images
    all_box = read_bbox(result_file)
    img = bbox_to_images(all_box)

    for thresh in THRESH:
        OUTPUT_FILE = output_file + "_nms_" + str(thresh) + ".txt"

        with open(OUTPUT_FILE, "w") as fw:
            count = 0
            for (imagePath, boundingBoxes) in img:
                # load the image and clone it
                print("[x] %d initial bounding boxes" % (len(boundingBoxes)))
                np_bounding_box = np.asarray(boundingBoxes)
                # print(np_bounding_box[:, -1])
                # perform non-maximum suppression on the bounding boxes
                pick = non_max_suppression_fast(boundingBoxes, np_bounding_box[:, -1], thresh)
                print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
                # print(imagePath)
                # print(pick)
                for bbox in pick:
                    result_str = imagePath + " " + str(bbox[4]) + " " + str(bbox[0]) + " " \
                                 + str(bbox[1]) + " " + str(bbox[2] - bbox[0]) + " " + str(bbox[3] - bbox[1]) + "\n"
                    # print(result_str)
                    count += 1
                    fw.write(result_str)

            print("Total BBOX Number: " + str(count))
        fw.close()
