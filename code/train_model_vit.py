from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.compat.v1 import Summary
from tensorflow.compat.v1.summary import FileWriter
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

import tensorflow_hub as hub

from matplotlib import cm
from random import randint
import numpy as np
import cv2
from PIL import Image, ImageDraw
import io
import time
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('scratch_dir', type=str, help='Scratch Directory')
parser.add_argument('data_dir', type=str, help='Dataset Directory')
parser.add_argument('--num_class', default=5, type=int, help='Number of classes in dataset (default: 5')
parser.add_argument('--no_weights', action='store_false', help='Load Inception model without weights')
parser.add_argument('--train_grayscale', action='store_true', help='Make train and validation data grayscale')
parser.add_argument('--test_grayscale', action='store_true', help='Make test data grayscale')
parser.add_argument('-l', '--labels', action='store_true', help='Display class labels on Tensorboard images')
parser.add_argument('--layer_cuttoff', default='', type=str, help='InceptionV3 layer to end model at (example: mixed7)')
parser.add_argument('-e', '--epochs', default=200, type=int, help='Maximum number of epochs (default: 200)')
parser.add_argument('-b', '--batch_size', default=50, type=int, help='Batch size during training (default: 50)')
parser.add_argument('--checkpoint_name', default='weights.hdf5', type=str, help='Name of checkpoint file where the best trained weights are stored (default: weights.hdf5')
parser.add_argument('--patience', default=10, type=int, help='Early stopping patience in number of epochs (default: 10)')
parser.add_argument('--model_name', default='model.json', type=str, help='Name of json file where the model is stored (default: model.json')
args = parser.parse_args()

args.scratch_dir = os.path.expanduser(args.scratch_dir)
args.data_dir = os.path.expanduser(args.data_dir)

os.environ["HOME"] = args.scratch_dir
os.environ["KERAS_HOME"] = args.scratch_dir + "/.keras"
os.environ["TFHUB_CACHE_DIR"] = args.scratch_dir + "/tfhub_modules"
print(args.data_dir)


ARRAY_PATH = args.data_dir + "/arrays/"
if (
    os.path.isfile(ARRAY_PATH + "x_train.npz")
    & os.path.isfile(ARRAY_PATH + "y_train.npz")
    & os.path.isfile(ARRAY_PATH + "x_validation.npz")
    & os.path.isfile(ARRAY_PATH + "y_validation.npz")
    & os.path.isfile(ARRAY_PATH + "x_test.npz")
    & os.path.isfile(ARRAY_PATH + "y_test.npz")
):
    x_train, y_train = (
        np.load(ARRAY_PATH + "x_train.npz")["arr_0"],
        np.load(ARRAY_PATH + "y_train.npz")["arr_0"],
    )
    x_validation, y_validation = (
        np.load(ARRAY_PATH + "x_validation.npz")["arr_0"],
        np.load(ARRAY_PATH + "y_validation.npz")["arr_0"],
    )
    x_test, y_test = (
        np.load(ARRAY_PATH + "x_test.npz")["arr_0"],
        np.load(ARRAY_PATH + "y_test.npz")["arr_0"],
    )
else:
    print('Arrays not found in "' + ARRAY_PATH + '". Use make_arrays.py to create arrays.')
    sys.exit()

if args.train_grayscale:
    x_train = np.mean(x_train, axis=3)
    x_train = np.stack((x_train, x_train, x_train), axis=3)
    x_validation = np.mean(x_validation, axis=3)
    x_validation = np.stack((x_validation, x_validation, x_validation), axis=3)
if args.test_grayscale:
    x_test = np.mean(x_test, axis=3)
    x_test = np.stack((x_test, x_test, x_test), axis=3)

print(x_train.shape)

IMAGE_SHAPE = x_train.shape[1:3]

image_input = Input(shape=IMAGE_SHAPE + (3,))
if not args.no_weights:
    print("Using random Inception weights")

# tf.config.set_soft_device_placement(True)


def get_model(
    handle="https://tfhub.dev/sayakpaul/vit_b8_fe/1",
    num_classes=args.num_class,
):
    hub_layer = hub.KerasLayer(handle, trainable=True)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(IMAGE_SHAPE + (3,), name="vit/input"),
            tf.keras.layers.Resizing(224, 224, name="resizing"),
            hub_layer,
            tf.keras.layers.Dense(num_classes, activation="softmax", name="vit/dense"),
        ]
    )

    return model


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = 0.5 * self.learning_rate_base * (1 + tf.cos(self.pi * (tf.cast(step, tf.float32) - self.warmup_steps) / float(self.total_steps - self.warmup_steps)))

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

    def get_config(self):
        config = {
            "learning_rate_base": self.learning_rate_base,
            "total_steps": self.total_steps,
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }
        return config


EPOCHS = 8
TOTAL_STEPS = int((x_train.shape[0] / args.batch_size) * EPOCHS)
WARMUP_STEPS = 10
INIT_LR = 0.03
WAMRUP_LR = 0.006

scheduled_lrs = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=TOTAL_STEPS,
    warmup_learning_rate=WAMRUP_LR,
    warmup_steps=WARMUP_STEPS,
)

optimizer = tf.keras.optimizers.SGD(scheduled_lrs, clipnorm=1.0)
loss = tf.keras.losses.CategoricalCrossentropy()

model = get_model()

for i in range(len(model.weights)):
    model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)
for i in range(len(model.layers)):
    model.layers[i]._handle_name = model.layers[i]._name + "_" + str(i)

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
print(model.summary())

print("Loaded model")

model_json = model.to_json()
if not os.path.exists(args.scratch_dir + "/checkpoints"):
    os.mkdir(args.scratch_dir + "/checkpoints")
with open(args.scratch_dir + "/checkpoints/" + args.model_name, "w+") as json_file:
    json_file.write(model_json)

starttime = time.strftime("%c")

def make_image(array, text=''):
    height, width, channel = array.shape
    image = Image.fromarray(array)
    d = ImageDraw.Draw(image)
    d.text((20,20), text, fill=(255,255,255))
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string) 

def unmodified(image):
    summary = Summary(value=[Summary.Value(tag="Unmodified", image=image)])
    writer = FileWriter(args.scratch_dir+'/tensorboard/'+starttime)
    writer.add_summary(summary)
    writer.close() 

def cam(model, layer, array, class_weights, actual, prediction):
    output_layer = model.layers[layer].output

    activation_model = Model(inputs=model.input, outputs=output_layer)
    activation = activation_model.predict(array.reshape((1,)+IMAGE_SHAPE+(3,)))

    writer = FileWriter(args.scratch_dir+'/tensorboard/'+starttime)

    cam = np.zeros((activation.shape[1:3]))
    for i, w in enumerate(class_weights[:, prediction]):
        cam += w * activation[0, :, :, i]
    cam = np.interp(cam, (cam.min(), cam.max()), (0, 1))
    heatmap = cm.autumn_r(cam) * 255
    cam = cam * 150
    cam[np.where(cam < 30)] = 0
    heatmap[:, :, 3] = cam
    heatmap = cv2.resize(heatmap, IMAGE_SHAPE[::-1], interpolation=cv2.INTER_NEAREST)
    if args.labels:
        heatmap = make_image(
            np.uint8(heatmap),
            "Actual:" + str(actual + 1) + " Prediction:" + str(prediction + 1),
        )
    else:
        heatmap = make_image(np.uint8(heatmap))
    summary = Summary(value=[Summary.Value(tag="CAM", image=heatmap)])
    writer.add_summary(summary)


class TensorboardSimple(Callback):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        num = randint(0, x_test.shape[0] - 1)
        x, y = x_test[num], y_test[num]

        actual = int(y.argmax())
        prediction = int(
            model.predict(x.reshape((1,) + IMAGE_SHAPE + (3,)))[0].argmax()
        )

        if args.labels:
            image = make_image(
                np.uint8(np.array(x) * 255),
                "Actual:" + str(actual + 1) + " Prediction:" + str(prediction + 1),
            )
        else:
            image = make_image(np.uint8(np.array(x) * 255))

        unmodified(image)
        cam(
            model,
            layer=len(model.layers) - 3,
            array=x,
            class_weights=model.layers[-1].get_weights()[0],
            actual=actual,
            prediction=prediction,
        )
        return


if not os.path.exists(args.scratch_dir + "/checkpoints"):
    os.mkdir(args.scratch_dir + "/checkpoints")
if not os.path.exists(args.scratch_dir + "/tensorboard"):
    os.mkdir(args.scratch_dir + "/tensorboard")


tbi_simple_callback = TensorboardSimple("Basic Visualizations")

filepath = args.scratch_dir + "/checkpoints/" + args.checkpoint_name
checkpoint = ModelCheckpoint(
    filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)

tensorboard = TensorBoard(
    log_dir=args.scratch_dir + "/tensorboard/" + starttime,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)
stopping = EarlyStopping(patience=args.patience, verbose=1)

callbacks_list = [tensorboard, checkpoint, stopping]
print("Created callbacks")

model.fit(
    x_train,
    y_train,
    epochs=args.epochs,
    verbose=1,
    callbacks=callbacks_list,
    validation_data=(x_validation, y_validation),
    shuffle=True,
    batch_size=args.batch_size,
)


# Load best checkpoint
model.load_weights(args.scratch_dir + "/checkpoints/" + args.checkpoint_name)
score = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
