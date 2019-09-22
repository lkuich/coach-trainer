'''
import tensorflow as tf
import tensorflow_hub as hub

classifier = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2"
#module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2")

# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
  resized_input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3], name="images")
  m = hub.Module(classifier)
  bottleneck_tensor = m(resized_input_tensor)
  #init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(bottleneck_tensor)
'''

import tensorflow as tf
import tensorflow_hub as hub
import os
from keras import backend as K

from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import optimize_for_inference_lib

tf.keras.backend.set_learning_phase(0)

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

print(model.input)
print(model.output)

def _parse_placeholder_types(values):
  """Extracts placeholder types from a comma separate list."""
  values = [int(value) for value in values.split(",")]
  return values if len(values) > 1 else values[0]

def optimize_for_inference(input_graph_path, output_path, input_names, output_names):
  PLACEHOLDER_TYPE_ENUM = str(dtypes.float32.as_datatype_enum)

  if not gfile.Exists(input_graph_path):
    print("Input graph file '" + input_graph_path + "' does not exist!")
    return -1

  input_graph_def = graph_pb2.GraphDef()
  with gfile.Open(input_graph_path, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

  output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      input_names,
      output_names,
      _parse_placeholder_types(PLACEHOLDER_TYPE_ENUM),
      False)

  f = gfile.GFile(output_path, "w")
  f.write(output_graph_def.SerializeToString())

def convert_to_barricuda(source_file, target_file):
  from mlagents.trainers.tensorflow_to_barracuda import convert
  convert(source_file, target_file)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph
        
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

# Write frozen
output_path = "model"
frozen_name = "tf_model.pb"
tf.io.write_graph(frozen_graph, output_path, frozen_name, as_text=False)

# Optimize
optimize_for_inference(os.path.join(output_path, frozen_name), os.path.join(output_path, "optimized.pb"), ["input_1"], ["out_relu/Relu6"])

# Unity
convert_to_barricuda(os.path.join(output_path, "optimized.pb"), os.path.join(output_path, "unity.bytes"))