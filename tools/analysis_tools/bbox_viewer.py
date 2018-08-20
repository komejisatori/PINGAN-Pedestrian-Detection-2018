#!/usr/bin/env python3
"""
bbox_viewer.py: draw bounding box on original image

This is a tool for drawing WIDER format bounding box on origin images.
--ignore : ignore area in image, image1 x1 y1 w1 h1 x2 y2 w2 h2...
--annotation : ground truth in image, image1 class1 x1 y1 w1 h1 class2 x2 y2 w2 h2
--result : detection result in image, image1 confidence1 x1 y1 w1 h1
                                      image1 confidence2 x2 y2 w2 h2
                                      .
                                      .
                                      .
--compare : secondary result file, same as result
--img : image folder
--output : output folder
--help/-h: argument information
"""
__author__ = "Jiaqi Cai"
__email__ = "jiaqi.cai22@pactera.com"
__credits__ = ["Jiaqi Cai", "Zhuoran Wu", "Kun Li"]

import os
import cv2
import argparse


# TODO
# change split to regualr expression, may speed up a little.
class Bbox:
    def __init__(self, bounding_file, ignore_file, result_file, compare_file,
                 img_dir, out_dir):
        self.annotation_list = self.__get_box_list(bounding_file)
        self.ignore_list = self.__get_box_list(ignore_file)
        self.result_file = self.__get_box_list_multi_line(result_file)
        self.compare_file = self.__get_box_list_multi_line(compare_file)
        self.img_dir = img_dir
        self.out_dir = out_dir

    def draw_box(self):
        for im_name in os.listdir(self.img_dir):
            im_file = os.path.join(self.img_dir, im_name)
            im = cv2.imread(im_file)

            if len(self.result_file) < 4:
                continue
            if self.annotation_list == []:
                pass
            elif im_name not in self.annotation_list:
                print('image %s not found in annotation list' % im_name)
            elif len(self.annotation_list[im_name]) == 0:
                print('image %s has no annotation box' % im_name)
            else:
                self.__draw_single_box(im, self.annotation_list[im_name],
                                       'annotation')

            if self.ignore_list == [] or im_name not in self.ignore_list:
                print('image %s not found in ignore list' % im_name)
            else:
                self.__draw_single_box(im, self.ignore_list[im_name], 'ignore')

            if self.result_file == [] or im_name not in self.result_file:
                print('image %s not found in result list' % im_name)
                continue
            else:
                self.__draw_single_box(im, self.result_file[im_name], 'result')

            if self.compare_file == [] or im_name not in self.compare_file:
                print('image %s not found in compare list' % im_name)
            else:
                self.__draw_single_box(im, self.compare_file[im_name],
                                       'compare')

            out_fig = os.path.join(self.out_dir,
                                   '%s_out.png' % im_name.split('.')[0])
            cv2.imwrite(out_fig, im)

    @staticmethod
    def __get_box_list_multi_line(file):
        if file == 'none':
            return []
        file_lines = open(file, 'r').readlines()
        box_list = {}
        for line in file_lines:
            line_splitted = line.split()
            if line_splitted[0] in box_list:
                box_list[line_splitted[0]].extend(line_splitted[1:])
            else:
                box_list[line_splitted[0]] = line_splitted[1:]
        return box_list

    @staticmethod
    def __get_box_list(file):
        if file == 'none':
            return []
        file_lines = open(file, 'r').readlines()
        box_list = {}
        for line in file_lines:
            line_splitted = line.split()
            box_list[line_splitted[0]] = line_splitted[1:]
        # print(box_list)
        return box_list

    @staticmethod
    def __draw_single_box(im, box_list, box_type='annotation'):
        index = 0
        while index < len(box_list):
            if box_type == 'annotation':
                box_color = (0, 255, 0) if box_list[index] == '1' else (255, 0,
                                                                        255)
                index = index + 1
            elif box_type == 'ignore':
                box_color = (255, 255, 0)
            elif box_type == 'result':
                if box_list[index] == '1':
                    box_color = (255, 0, 0)
                else:
                    box_color = (0, 0, 255)
                index = index + 2
            elif box_type == 'compare':
                box_color = (0, 255, 0)
                index = index + 1
            else:
                raise RuntimeError('no box type %s' % box_type)
            box = list(map(int, map(
                float, box_list[index:index + 4])))  # get present box
            # print(box)
            cv2.rectangle(im, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), box_color, 1)
            index = index + 4


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Draw Bounding Box')
    parser.add_argument(
        '--ignore', dest='ignore', help='Ignore file', default='none')
    parser.add_argument(
        '--annotation',
        dest='annotation',
        help='annotation file',
        default='none')
    parser.add_argument('--img', dest='img', help='Image folder')
    parser.add_argument(
        '--output', dest='output', help='Image output folder', default='out')
    parser.add_argument(
        '--result', dest='result', help='Result file', default='none')
    parser.add_argument(
        '--compare',
        dest='compare',
        help='Another result file to compare',
        default='none')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    anno_file = args.annotation
    ignore_file = args.ignore
    result_file = args.result
    compare_file = args.compare
    img_dir = args.img
    out_dir = args.output

    for path in [anno_file, ignore_file, result_file, compare_file]:
        if (not os.path.isfile(path)) and path != 'none':
            raise IOError('wrong input file %s' % path)

    for path in [img_dir, out_dir]:
        if not os.path.isdir(path):
            raise IOError('wrong input directory %s' % path)
    show_bbox = Bbox(anno_file, ignore_file, result_file, compare_file,
                     img_dir, out_dir)
    show_bbox.draw_box()
