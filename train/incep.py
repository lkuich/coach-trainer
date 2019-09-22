import tensorflow as tf
import tensorflow_hub as hub

# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
  resized_input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3], name="images")
  m = hub.Module(module_spec)
  bottleneck_tensor = m(resized_input_tensor)
  #init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(bottleneck_tensor)