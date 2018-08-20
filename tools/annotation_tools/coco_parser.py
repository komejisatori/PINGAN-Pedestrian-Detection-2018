#!/usr/bin/env python3
"""
coco_parser.py: parse WIDER format annotation to COCO format

'iscrowd' is set to 0
--input : input WIDER format annotation
--image : input image folder to get image size for coco format
--output : output coco format json file
--help : argument information 

# TODO
--ignore : ignore file, not implemented yet
# TODO
interface to parse any kind of data to coco format
"""
__author__ = "Jiaqi Cai"
__email__ = "jiaqi.cai22@pactera.com"

import json
import os
import cv2
import re
import argparse
from pprint import pprint


class CocoParser:
    def __init__(self, file_in, file_ignore, img_in):
        if file_in is None:
            raise ValueError('no input annotation file')
        if img_in is None:
            raise ValueError('no image folder')
        self.annotations = {}
        self.init_img = []
        self.json = {
            'images': [],
            'type':
                'instances',
            'annotations': [],
            'catagories': [{
                'supercategory': 'none',
                'id': 1,
                'name': 'pedestrian'
            }, {
                'supercategory': 'none',
                'id': 2,
                'name': 'cyclist'
            }]
        }
        self.image_path = img_in
        # with open(file_in, 'r') as anno_file:
        #     init_lines = anno_file.readlines()
        #     self.annotations = self.convert_lines_to_lists(init_lines)

        if not os.path.isdir(img_in):
            raise OSError('wrong input image directory')
        self.__parse()

        if file_ignore is not None and os.path.isfile(file_ignore):
            with open(file_ignore, 'r') as ignore_file:
                ignore_lines = ignore_file.readlines()
                self.ignore = self.convert_ignores_to_lists(ignore_lines)

    @staticmethod
    def convert_lines_to_lists(lines):
        lists = {}
        for line in lines:
            line_list = line.split()
            line_list_num = list(map(int, line_list[1:]))
            if line_list[0] not in lists:
                lists[line_list[0]] = []
            idx = 0
            while idx < len(line_list_num):
                lists[line_list[0]].append({
                    line_list_num[idx]:
                        line_list_num[idx + 1:idx + 5]
                })
                # goto next box
                idx = idx + 5
        return lists

    @staticmethod
    def convert_ignores_to_lists(lines):
        lists = {}
        for line in lines:
            line_list = line.split()
            line_list_num = list(map(int, line_list[1:]))
            if line_list[0] not in lists:
                lists[line_list[0]] = []
            idx = 0
            while idx < len(line_list_num):
                lists[line_list[0]].append(line_list_num[idx:idx + 4])
                # goto next box
                idx = idx + 4
            # TODO 
            # delete break
            break
        return lists

    def __parse(self):
        seg_id = 1
        for file_name in self.annotations:
            img_id = int(re.match(r'img([\d]*).jpg', file_name).group(1))
            file_path = os.path.join(self.image_path, file_name)
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            img_json = {
                'file_name': file_name,
                'height': height,
                'width': width,
                'id': img_id
            }
            self.json['images'].append(img_json)
            bounding = []
            for bbox in self.annotations[file_name]:
                for key in bbox:  # only one key in each bbox
                    bounding = bbox[key]
                area = bounding[2] * bounding[3]
                segmentation = [[
                    bounding[0], bounding[1], bounding[0],
                    bounding[3] + bounding[1], bounding[2] + bounding[0],
                    bounding[3] + bounding[1], bounding[2] + bounding[0],
                    bounding[1]
                ]]
                anno_json = {
                    'segmentation': segmentation,
                    'area': area,
                    'iscrowd': 0,
                    'image_id': img_id,
                    'bbox': bounding,
                    'catagory_id': key,
                    'id': seg_id,
                    'ignore': 0
                }
                self.json['annotations'].append(anno_json)
                seg_id = seg_id + 1

    # def __iscrowd(ignore_box, gt_box):

    def save(self, dir):
        with open(dir, 'w') as outfile:
            json.dump(self.json, outfile)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Parse WIDER Format')
    parser.add_argument('--input', dest='input', help='Origin File')
    parser.add_argument('--image', dest='image', help='Image Folder')
    parser.add_argument(
        '--output', dest='output', help='Output Coco File', default='out.json')
    parser.add_argument(
        '--ignore', dest='ignore', help='Ignore File', default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    input_file = args.input
    input_image_dir = args.image
    output_file = args.output
    ignore_file = args.ignore
    cp = CocoParser(input_file, ignore_file, input_image_dir)
    cp.save(output_file)
