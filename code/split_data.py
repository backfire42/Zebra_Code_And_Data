import os
import shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Dataset Directory')
parser.add_argument('-n', '--num_class', default=5, type=int, help='Number of classes (default: 5)')
parser.add_argument('--name_class', default='fish', type=str, help='Prefix for class (default: fish)')
parser.add_argument('--train_size', default='0.7', type=float, help='Fraction of files to include in train split  (default: 0.7)')
parser.add_argument('--validation_size', default='0.15', type=float, help='Fraction of files to include in validation split (default: 0.15)')
parser.add_argument('--test_size', default='0.15', type=float, help='Fraction of files to include in test split (default: 0.15)')
args = parser.parse_args()

args.data_dir = os.path.abspath(args.data_dir)
ARRAY_PATH = args.data_dir + "/arrays/"

if not os.path.exists(args.data_dir + "/train"):
    os.mkdir(args.data_dir + "/train")
if not os.path.exists(args.data_dir + "/validation"):
    os.mkdir(args.data_dir + "/validation")
if not os.path.exists(args.data_dir + "/test"):
    os.mkdir(args.data_dir + "/test")

for class_num in range(1, args.num_class + 1):
    args.data_dir = args.data_dir
    source = args.data_dir + "/" + args.name_class + str(class_num) + "/"
    train = args.data_dir + "/train/" + args.name_class + str(class_num) + "/"
    validation = args.data_dir + "/validation/" + args.name_class + str(class_num) + "/"
    test = args.data_dir + "/test/" + args.name_class + str(class_num) + "/"

    if not os.path.exists(train[:-1]):
        os.mkdir(train[:-1])
    if not os.path.exists(validation[:-1]):
        os.mkdir(validation[:-1])
    if not os.path.exists(test[:-1]):
        os.mkdir(test[:-1])

    files = os.listdir(source)
    for f in files:
        num = np.random.rand(1)
        if num < args.train_size:
            shutil.copy2(source + f, train + f)
        elif num >= args.train_size and num <= (args.train_size + args.validation_size):
            shutil.copy2(source + f, validation + f)
        elif num > (args.train_size + args.validation_size) and num <= (
            args.train_size + args.validation_size + args.test_size
        ):
            shutil.copy2(source + f, test + f)
