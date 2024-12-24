from PIL import Image
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Data Directory')
parser.add_argument('-r', '--ratio', nargs=2, default=[16.0, 9.0], type=int, help='Aspect ratio to compare image to (default: 16.0 9.0)')
args = parser.parse_args()
args.data_dir = os.path.abspath(args.data_dir)

aspect_ratio = args.ratio[0] / args.ratio[1]

for root, dirs, files in os.walk(args.data_dir, topdown=False):
    for file in files:
        file_name = os.fsdecode(os.path.join(root, file))
        if file_name.endswith(".jpg"):
            print(file_name)
            im = Image.open(file_name)

            new_width = im.width
            new_height = im.height
            width = im.width
            height = im.height
            if im.width / im.height > aspect_ratio:
                new_height = int(round((1 / aspect_ratio) * im.width))
                im = np.array(im)
                im = np.pad(im, ((int((new_height-height)/2), int((new_height-height)/2)), (0,0), (0,0)), mode='edge')
            else:
                new_width = int(round(aspect_ratio * im.height))
                im = np.array(im)
                im = np.pad(im, ((0,0), (int((new_width-width)/2),int((new_width-width)/2)), (0,0)), mode='edge')

            im = Image.fromarray(im)
            im.save(file_name)
