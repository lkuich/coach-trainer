from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib


def optimize(input, output, input_names, output_names):
  placeholder_type_enum = str(dtypes.float32.as_datatype_enum)
  frozen_graph = True
    
  if not gfile.Exists(input):
    print("Input graph file '" + input + "' does not exist!")
    return False

  input_graph_def = graph_pb2.GraphDef()
  with gfile.Open(input, "rb") as f:
    data = f.read()
    if frozen_graph:
      input_graph_def.ParseFromString(data)
    else:
      text_format.Merge(data.decode("utf-8"), input_graph_def)

  output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      input_names.split(","),
      output_names.split(","),
      _parse_placeholder_types(placeholder_type_enum),
      False)

  if frozen_graph:
    f = gfile.GFile(FLAGS.output, "w")
    f.write(output_graph_def.SerializeToString())
  else:
    graph_io.write_graph(output_graph_def,
                         os.path.dirname(FLAGS.output),
                         os.path.basename(FLAGS.output))
  return True


def _parse_placeholder_types(values):
  """Extracts placeholder types from a comma separate list."""
  values = [int(value) for value in values.split(",")]
  return values if len(values) > 1 else values[0]

