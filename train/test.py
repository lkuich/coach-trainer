import tensorflowjs as tfjs

tfjs.converters.convert_tf_saved_model('/opt/ml/model/saved_model/', 'output', '/opt/ml/model/tfjs')