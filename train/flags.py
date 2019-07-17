class Flags:
  def __init__(
    self,
    image_dir,
    output_graph,
    intermediate_output_graphs_dir,
    intermediate_store_frequency,
    output_labels,
    summaries_dir,
    how_many_training_steps,
    learning_rate,
    testing_percentage,
    validation_percentage,
    eval_step_interval,
    train_batch_size,
    test_batch_size,
    validation_batch_size,
    print_misclassified_test_images,
    bottleneck_dir,
    final_tensor_name,
    input_tensor_name,
    flip_left_right,
    random_crop,
    random_scale,
    random_brightness,
    tfhub_module,
    saved_model_dir,
    logging_verbosity,
    checkpoint_path,
    tf_lite,
    tfjs,
    unity
  ):
    self.image_dir = image_dir
    self.output_graph=output_graph
    self.intermediate_output_graphs_dir=intermediate_output_graphs_dir
    self.intermediate_store_frequency=intermediate_store_frequency
    self.output_labels=output_labels
    self.summaries_dir=summaries_dir
    self.how_many_training_steps=how_many_training_steps
    self.learning_rate=learning_rate
    self.testing_percentage=testing_percentage
    self.validation_percentage=validation_percentage
    self.eval_step_interval=eval_step_interval
    self.train_batch_size=train_batch_size
    self.test_batch_size=test_batch_size
    self.validation_batch_size=validation_batch_size
    self.print_misclassified_test_images=print_misclassified_test_images
    self.bottleneck_dir=bottleneck_dir
    self.final_tensor_name=final_tensor_name
    self.input_tensor_name=input_tensor_name
    self.flip_left_right=flip_left_right
    self.random_crop=random_crop
    self.random_scale=random_scale
    self.random_brightness=random_brightness
    self.tfhub_module=tfhub_module
    self.saved_model_dir=saved_model_dir
    self.logging_verbosity=logging_verbosity
    self.checkpoint_path=checkpoint_path
    self.tf_lite=tf_lite
    self.tfjs=tfjs
    self.unity=unity