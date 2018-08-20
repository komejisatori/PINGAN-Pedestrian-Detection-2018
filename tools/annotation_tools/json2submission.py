#!/usr/bin/env python3
##############################################################################
#
# Purpose: Transfer COCO JSON output file to submission file format.
# Author: Zhuoran Wu <zhuoran.wu@pactera.com>
# Created: May 22, 2018
#
##############################################################################

import sys
import argparse
import json

THRESHES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def parse_args():
    parser = argparse.ArgumentParser(description='JSON Output file to submission format.')
    parser.add_argument(
        '--input',
        dest='input_files',
        help='Input json file.',
        default=None,
        type=str
    )
    parser.add_argument(
        '--input',
        dest='output_file',
        help='Output file name without txt.',
        default="Submission_result",
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='threshes',
        nargs='*',
        default=THRESHES,
        help='Thresh Confidence, a list. [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    threshs = [float(i) for i in args.threshes]
    input_file = args.input_files
    output_filename = args.output_file

    with open(input_file) as f:
        data = json.load(f)

    for thresh in threshs:
        OUTPUT_FILE_NAME = output_filename + '_' + str(thresh) + "_box_number_"

        data = filter(lambda x: float(x['score']) >= thresh, data)

        output_dict = [x for x in data if float(x['score']) >= thresh]

        OUTPUT_FILE_NAME += str(len(output_dict))
        OUTPUT_FILE_NAME += '.txt'

        with open(OUTPUT_FILE_NAME, 'w') as fw:
            for item in output_dict:
                result_string = ""
                image_id = str(item['image_id']).zfill(5)
                image_id = "img" + image_id + ".jpg"
                result_string += image_id
                result_string += " "
                result_string += str(item['score'])
                for bbox in item['bbox']:
                    result_string += " "
                    result_string += str(bbox)

                fw.write(result_string + "\n")
