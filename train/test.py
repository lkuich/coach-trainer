def test_js():
    import tensorflowjs as tfjs
    tfjs.converters.convert_tf_saved_model('/opt/ml/model/saved_model/', 'output', '/opt/ml/model/tfjs')

def test_unity(source_file, target_file):
    from mlagents.trainers.tensorflow_to_barracuda import convert_barricuda
    convert_barricuda(source_file, target_file)

test_unity('/opt/ml/model/frozen.pb', '/opt/ml/model/unity.bytes')