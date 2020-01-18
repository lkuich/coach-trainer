import os

prefix = '/opt/ml/'
output_dir = os.path.join(prefix, 'model')

output_graph=os.path.join(output_dir, 'frozen.pb')
unity=os.path.join(output_dir, 'unity.bytes')

def convert_to_barricuda(source_file, target_file):
  from mlagents.trainers.tensorflow_to_barracuda import convert
  convert(source_file, target_file)

convert_to_barricuda(output_graph, unity)