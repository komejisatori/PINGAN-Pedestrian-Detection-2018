#!/usr/bin/env python3
##############################################################################
#
# Purpose: Transfer WIDER Annotations file to submission file format.
# Author: Zhuoran Wu <zhuoran.wu@pactera.com>
# Created: May 22, 2018
#
# Annotations Format
#
# ImageName [(c x y w h) (c x y w h)]
# ```
# img18169.jpg 1 1110 275 187 299 1 1047 312 123 252 1 936 302 42 88 1 896 311 24 62 1 810 318 22 63 1 756 281 90 203 1 703 322 19 44 1 647 324 16 43 1 546 321 27 54
# img06638.jpg 1 863 71 18 39 1 279 92 31 69 1 261 79 24 61 1 338 100 15 65 1 319 90 18 71 1 21 54 24 75 1 42 65 19 64 1 117 33 21 62 1 144 40 15 56 1 5 77 11 52 1 166 15 17 39 1 794 46 22 30 2 1023 85 36 56 2 888 89 29 56 2 929 97 37 75 2 860 69 22 45 2 1202 181 49 89
# img15001.jpg
# img08548.jpg 1 64 177 62 136 1 1387 442 207 368 1 1439 317 82 230 1 1567 289 81 225 1 1543 280 70 198 1 1521 196 52 127 1 1524 223 63 159 1 1388 165 41 105 1 1416 166 37 97 2 114 548 160 274 2 1445 161 80 116
# img15784.jpg
# ```
#
# Output Result Format
# ```
# img06638.jpg 1 863 71 18 39
# img06638.jpg 1 279 92 31 69
# img06638.jpg 1 261 79 24 61
# img06638.jpg 1 338 100 15 65
# img06638.jpg 1 319 90 18 71
# img06638.jpg 1 21 54 24 75
# img06638.jpg 1 42 65 19 64
# img06638.jpg 1 117 33 21 62
# img06638.jpg 1 144 40 15 56
# img06638.jpg 1 5 77 11 52
# img06638.jpg 1 166 15 17 39
# img06638.jpg 1 794 46 22 30
# img06638.jpg 2 1023 85 36 56
# img06638.jpg 2 888 89 29 56
# img06638.jpg 2 929 97 37 75
# img06638.jpg 2 860 69 22 45
# img06638.jpg 2 1202 181 49 89
# ```
##############################################################################

import argparse
import sys

ANNOTATIONS_FILE = 'val_annotations.txt'
PARSER_RESULT_FILE = 'val_result_gt.txt'


def parse_args():
    parser = argparse.ArgumentParser(description='Transfer Val Annotations file to submission file format.')
    parser.add_argument(
        '--annotation',
        dest='annotation',
        help='Annotations File (/path/to/val_annotations.txt)',
        default=ANNOTATIONS_FILE,
        type=str
    )
    parser.add_argument(
        '--result',
        dest='result',
        help='Output Dest Annotations File (/path/to/val_result_gt.txt)',
        default=PARSER_RESULT_FILE,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    with open(args.annotation, 'w') as result, open(args.result, 'r') as the_file:
        lines = the_file.readlines()
        for x, line in enumerate(lines):
            parts = line.split(" ")
            if len(parts) <= 1:
                print(parts)
            else:
                image = parts[0]
                count = int((len(parts) - 1) / 5)
                for i in range(count):
                    result_line = parts[0]
                    result_line += " "
                    result_line += "1"
                    for j in range(4):
                        result_line += " "
                        result_line += parts[5 * i + j + 2]
                    result_line += "\n"

                    result.write(result_line)

                    result_line = parts[0]
