from tensorflow import keras

import numpy as np
import math
import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('scratch_dir', type=str, help='Scratch Directory')
parser.add_argument('data_dir', type=str, help='Dataset Directory')
parser.add_argument('-b', '--batch_size', default=50, type=int, help='Batch size during data generation (default: 50)')
parser.add_argument('-n', '--num_train', default=10000, type=int, help='Number of training images (default: 10000)')
parser.add_argument('-i', '--image_shape', nargs=2, default=[180, 320], type=int, help='Desired data image dimensions (default: 180 320)')
parser.add_argument('--rotation_range', default=360, type=int, help='Image Augmentation - Degree range for random rotations (default: 360)')
parser.add_argument('--width_shift_range', default=0.1, type=float, help='Image Augmentation - Range for fraction of total image width for random horizontal shifts (default: 0.1)')
parser.add_argument('--height_shift_range', default=0.1, type=float, help='Image Augmentation - Range for fraction of total image height for random vertical shifts (default: 0.1)')
parser.add_argument('--brightness_range', nargs=2, default=[0.3, 1.3], type=int, help='Image Augmentation - Range for random brightness shift values (default: 0.3 1.3)')
parser.add_argument('--shear_range', default=0.05, type=float, help='Image Augmentation - Range for random shear intesity values (Shear angle in counter-clockwise direction in degrees/360) (default: 0.05)')
parser.add_argument('--zoom_range', default=0.1, type=float, help='Image Augmentation - Range for random zoom (default: 0.1)')
parser.add_argument('--channel_shift_range', default=5, type=int, help='Image Augmentation - Range for random channel shifts (default: 0.1)')
parser.add_argument('--no_vertical_flip', action='store_false', help='Image Augmentation - Disable random vertical flips')
parser.add_argument('--no_horizontal_flip', action='store_false', help='Image Augmentation - Disable random horizontal flips')
args = parser.parse_args()

args.scratch_dir = os.path.abspath(args.scratch_dir)
args.data_dir = os.path.abspath(args.data_dir)
args.image_shape = tuple(args.image_shape)
os.environ["HOME"] = args.scratch_dir
print(args.data_dir)

ARRAY_PATH = args.data_dir + "/arrays/"


def make_arrays(datagen, name, is_train=False):
    generator = datagen.flow_from_directory(
        directory=args.data_dir + "/" + name + "/",
        target_size=args.image_shape,
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=args.batch_size,
        shuffle=True,
        seed=0,  # seed=None,
        follow_links=False,
        subset=None,
        interpolation="nearest",
    )

    x, y = [], []
    print(name + "_generator")
    num_images = len(
        glob.glob(args.data_dir + "/" + name + "/**/" + "*.jpg", recursive=True)
    )
    if is_train:
        num = int(math.ceil(args.num_train / num_images) * len(generator))
    else:
        num = len(generator)
    print(len(generator))
    print(num)
    for i in range(num):
        a, b = generator.next()
        print(i)
        print(a.shape)
        print(b.shape)
        for j in range(a.shape[0]):
            x.append(a[j])
            y.append(b[j])
    x, y = np.array(x), np.array(y)

    np.savez_compressed(ARRAY_PATH + "x_" + name, x)
    np.savez_compressed(ARRAY_PATH + "y_" + name, y)

    return x, y


if not os.path.exists(ARRAY_PATH[:-1]):
    os.mkdir(ARRAY_PATH[:-1])

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    channel_shift_range=args.channel_shift_range,
    shear_range=args.shear_range,
    zoom_range=args.zoom_range,
    vertical_flip=args.no_vertical_flip,
    horizontal_flip=args.no_horizontal_flip,
    rotation_range=args.rotation_range,
    width_shift_range=args.width_shift_range,
    height_shift_range=args.height_shift_range,
    brightness_range=args.brightness_range,
)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

make_arrays(train_datagen, "train", True)
make_arrays(val_datagen, "validation")
make_arrays(test_datagen, "test")
