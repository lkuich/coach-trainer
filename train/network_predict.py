#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import csv
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, Response, request, redirect, url_for

prefix = '/opt/ml/model/'

UPLOAD_FOLDER = '/opt/ml/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    #text_format.Merge(f.read(), graph_def)
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def read_tensor_from_image_file(file_name, input_height=224, input_width=224, input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def get_input_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  return parser.parse_args()

def predict(image):
  graph = load_graph(os.path.join(prefix, 'model.pb'))
  labels = load_labels(os.path.join(prefix, 'labels.csv'))  

  output_name = "import/softmax_input/Softmax"
  input_name = "import/lambda_input_input"

  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  t = read_tensor_from_image_file(image)
  
  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  r = []
  for i in top_k:
    r.append(str(labels[i] + ' ' + results[i]))
    
  return r

# The flask app for serving predictions
app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return Response(response='\n', status=200, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
  file = request.files['file']
  if file:
      filename = secure_filename(file.filename)
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      file.save(filepath)

  result = predict(filepath);

  return Response(response=result, status=200, mimetype='application/json')