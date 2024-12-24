import os
import numpy as np
from PIL import Image, ImageFilter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Data Directory')
parser.add_argument('-t', '--tolerance', default=15, type=int, help='Tolerance for chroma keying (default: 15)')
parser.add_argument('-c', '--crop', nargs=4, default=[560, 660, 230, 1230], type=int, help='Number of pixels to crop border for left, bottom, right, and top sides respectively (default for 2018-8-21 dataset: 560 660 230 1230)')
parser.add_argument('--blur1', default=50, type=int, help='First Gaussian blur radius (default: 50)')
parser.add_argument('--median', default=9, type=int, help='Median filter size (default: 9)')
parser.add_argument('--blur2', default=50, type=int, help='Second Gaussian blur radius (default: 30)')
args = parser.parse_args()
args.data_dir = os.path.abspath(args.data_dir)
print(args.data_dir)

for root, dirs, files in os.walk(args.data_dir, topdown=False):
    for file in files:
        file_name = os.fsdecode(os.path.join(root, file))
        if file_name.endswith(".jpg"):
            print(file_name)
            img = Image.open(file_name)

            w, h = img.size
            img = img.crop(
                (args.crop[0], args.crop[1], w - args.crop[2], h - args.crop[3])
            )
            old_data = np.array(img, dtype="uint8")
            img = img.convert("RGBA")
            white_data = np.array(img, dtype="uint8")
            img = img.filter(ImageFilter.GaussianBlur(radius=args.blur1))
            img = img.filter(ImageFilter.MedianFilter(size=args.median))
            data = np.array(img, dtype="uint8")

            red, green, blue, alpha = data.T
            white_areas = (
                ((red + args.tolerance > green) & (red - args.tolerance < green))
                & ((green + args.tolerance > blue) & (green - args.tolerance < blue))
                & ((blue + args.tolerance > red) & (blue - args.tolerance < red))
            )
            data[..., :-1][white_areas.T] = (0, 0, 0)

            img = Image.fromarray(data)
            img = img.filter(ImageFilter.GaussianBlur(radius=args.blur2))
            data = np.array(img, dtype="uint8")

            red, green, blue, alpha = data.T

            black_areas = (red == 0) & (green == 0) & (blue == 0)
            data[..., :-1][black_areas.T] = (255, 255, 255)
            white_data[..., :-1][black_areas.T] = (255, 255, 255)

            mask = white_data < 255
            coords = np.argwhere(mask)
            if coords.shape[0] == 0:
                continue
            x0, y0, i = coords.min(axis=0)
            x1, y1, i = coords.max(axis=0) + 1
            old_data = old_data[x0:x1, y0:y1]

            old_data = np.array(old_data, dtype="uint8")
            img = Image.fromarray(old_data)
            img.save(file_name)
