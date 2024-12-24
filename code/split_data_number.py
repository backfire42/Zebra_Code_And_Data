import os
import shutil
import argparse
import glob
import random
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Target Dataset Directory')
parser.add_argument('source_dir', type=str, help='Source Directory')
parser.add_argument('-n', '--num_class', default=5, type=int, help='Number of classes (default: 5)')
parser.add_argument('--name_class', default='fish', type=str, help='Prefix for class (default: fish)')
parser.add_argument('--train_num', default='10', type=float, help='Number of files to include in train split  (default: 10)')
parser.add_argument('--validation_size', default='0.15', type=float, help='Fraction of files to include in validation split (default: 0.15)')
parser.add_argument('--test_size', default='0.15', type=float, help='Fraction of files to include in test split (default: 0.15)')
args = parser.parse_args()

args.data_dir = os.path.abspath(args.data_dir)
ARRAY_PATH = args.data_dir + "/arrays/"

if not os.path.exists(args.data_dir + "/train"):
    os.makedirs(args.data_dir + "/train")
if not os.path.exists(args.data_dir + "/validation"):
    os.makedirs(args.data_dir + "/validation")
if not os.path.exists(args.data_dir + "/test"):
    os.makedirs(args.data_dir + "/test")

copy_tree(args.source_dir, args.data_dir + "/")

for class_num in range(1, args.num_class + 1):
    args.data_dir = args.data_dir
    source = args.data_dir + "/" + args.name_class + str(class_num) + "/"
    train = args.data_dir + "/train/" + args.name_class + str(class_num) + "/"
    validation = args.data_dir + "/validation/" + args.name_class + str(class_num) + "/"
    test = args.data_dir + "/test/" + args.name_class + str(class_num) + "/"

    if not os.path.exists(train[:-1]):
        os.makedirs(train[:-1])
    if not os.path.exists(validation[:-1]):
        os.makedirs(validation[:-1])
    if not os.path.exists(test[:-1]):
        os.makedirs(test[:-1])

    num_images = int(len(glob.glob(source + "*.jpg")))
    print(int(args.train_num/args.num_class + num_images*(args.validation_size+args.test_size)))
    print(len(os.listdir(source)))
    files = random.sample(os.listdir(source), min(num_images, int(args.train_num/args.num_class + num_images*(args.validation_size+args.test_size))))
    for i,f in enumerate(files):
        if i < args.train_num/args.num_class:
            shutil.copy2(source + f, train + f)
        elif i >= args.train_num / args.num_class and i < min(
            int(
                args.train_num / args.num_class
                + (num_images - args.train_num / args.num_class)
                * (args.validation_size / (args.validation_size + args.test_size))
            ),
            int(args.train_num / args.num_class + num_images * args.validation_size),
        ):
            shutil.copy2(source + f, validation + f)
        else:
            shutil.copy2(source + f, test + f)
