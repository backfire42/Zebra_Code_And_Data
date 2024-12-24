from tensorflow.keras.models import model_from_json
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('scratch_dir', type=str, help='Scratch Directory')
parser.add_argument('--checkpoint_name', default='weights.hdf5', type=str, help='Name of checkpoint file where the best trained weights are stored (default: weights.hdf5')
parser.add_argument('-i', '--image_size', nargs=2, default=[180, 320], type=int, help='Data image dimensions. Must match model input image dimension (default: 180 320)')
parser.add_argument('--model_name', default='model.json', type=str, help='Name of json file where the model is stored (default: model.json)')
args = parser.parse_args()

args.scratch_dir = os.path.abspath(args.scratch_dir)
args.image_size = tuple(args.image_size)
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
    model = model_from_json(
        loaded_model_json, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    model.load_weights(args.scratch_dir + "/checkpoints/" + args.checkpoint_name)
    print(model.summary())
else:
    print('Model checkpoint not found at "' + args.scratch_dir+'/checkpoints/'+args.checkpoint_name + '". Check the name of the checkpoint file or run train_model.py')


def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs]
    )
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=run_meta, cmd="op", options=opts
        )
        return flops.total_float_ops


print(str(get_flops(model) / 1e9) + " GFLOPS")
print(tf.config.experimental.get_memory_info("GPU:0"))
