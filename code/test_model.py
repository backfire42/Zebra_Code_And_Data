from tensorflow import keras
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import model_from_json

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('scratch_dir', type=str, help='Scratch Directory')
parser.add_argument('data_dir', type=str, help='Data Directory')
parser.add_argument('--num_gpu', default=4, type=int, help='Number of GPUs. Must match trained model parameters (default: 4)')
parser.add_argument('-b', '--batch_size', default=50, type=int, help='Batch size during training (default: 50)')
parser.add_argument('-i', '--image_size', nargs=2, default=[180, 320], type=int, help='Data image dimensions. Must match model input image dimension (default: 180 320)')
parser.add_argument('--checkpoint_name', default='weights.hdf5', type=str, help='Name of checkpoint file where the best trained weights are stored (default: weights.hdf5')
parser.add_argument('--model_name', default='model.json', type=str, help='Name of json file where the model is stored (default: model.json')
args = parser.parse_args()

args.scratch_dir = os.path.abspath(args.scratch_dir)
args.data_dir = os.path.abspath(args.data_dir)
args.image_size = tuple(args.image_size)
print(args.data_dir)
print(args.checkpoint_name)
print(args.model_name)

os.environ["HOME"] = args.scratch_dir


if os.path.isfile(args.scratch_dir + "/checkpoints/" + args.model_name):
    json_file = open(args.scratch_dir + "/checkpoints/" + args.model_name, "rb")
    loaded_model_json = json_file.read()
    json_file.close()
else:
    print('Model json file not found at "' + args.scratch_dir+'/checkpoints/'+args.model_name + '". Check the name of the json file or run train_model.py')

if os.path.isfile(args.scratch_dir + "/checkpoints/" + args.checkpoint_name):
    model = model_from_json(loaded_model_json)
    model.load_weights(args.scratch_dir + "/checkpoints/" + args.checkpoint_name)
    print(model.summary())
else:
    print('Model checkpoint not found at "' + args.scratch_dir+'/checkpoints/'+args.checkpoint_name + '". Check the name of the checkpoint file or run train_model.py')

model.compile(
    loss=categorical_crossentropy,
    optimizer=SGD(lr=0.001, momentum=0.9),
    metrics=["accuracy"],
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    directory=args.data_dir,
    target_size=args.image_size,
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=args.batch_size,
    shuffle=True,
    seed=None,
    follow_links=False,
    subset=None,
    interpolation="nearest",
)

score = model.evaluate(test_generator, workers=16, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
