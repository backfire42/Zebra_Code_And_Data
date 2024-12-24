from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Data Directory')
parser.add_argument('-r', '--rotate', default=90, type=int, help='Number of degrees to rotate image clockwise (default: 90)')
parser.add_argument('--ignore_ratio', action='store_true', help='Rotate image even if width is less than height')
args = parser.parse_args()
args.data_dir = os.path.abspath(args.data_dir)

for root, dirs, files in os.walk(args.data_dir, topdown=False):
    for file in files:
        file_name = os.fsdecode(os.path.join(root, file))
        if file_name.endswith(".jpg"):
            im = Image.open(file_name)
            if args.ignore_ratio or im.width < im.height:
                im = im.rotate(args.rotate, expand=True)
                im.save(file_name)
