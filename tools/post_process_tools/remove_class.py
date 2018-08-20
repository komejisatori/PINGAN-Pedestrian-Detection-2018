import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Remove class to submission')
    parser.add_argument(
        '--result',
        dest='result_files',
        help='Results Files with class(/path/to/)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output',
        dest='output_files',
        help='Output Results Files without class (/path/to/)',
        default="ensemble",
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result_file = args.result_files
    output_file = args.output_files

    with open(result_file, "r") as fr, open(output_file, "w") as fw:
        for idx, line in enumerate(fr):
            parts = line.split(" ")
            single = parts[0]
            for i in range(2, 7):
                single += " "
                single += parts[i]
            fw.write(single)

    fr.close()
    fw.close()
