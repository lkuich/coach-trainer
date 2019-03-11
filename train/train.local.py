#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers
import tensorflow.keras.backend as K

import argparse
import os
import json

from environment import create_trainer_environment

prefix = '/opt/ml/'

if __name__ =='__main__':
    env = create_trainer_environment()
    
    batch_size = env.hyperparameters.get('batch_size', object_type=int)
    data_augmentation = env.hyperparameters.get('data_augmentation', default=False, object_type=bool)
    epochs = env.hyperparameters.get('epochs', default=1, object_type=int)
    feature_extractor_url = env.hyperparameters.get('module', default="models/mobilenet_v2_100_224-feature_vector-2", object_type=str)
    
    data_root = env.channel_dirs['train']
    
    files = os.listdir(data_root)
    for name in files:
        filename = name
        break

    data_root = data_root + '/' + filename
    labels = []
    for label in os.listdir(data_root):
        labels.append(label)
    with open('labels.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(labels)
    csvFile.close()


    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

    def feature_extractor(x):
        feature_extractor_module = hub.Module(feature_extractor_url)
        return feature_extractor_module(x)

    IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

    image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)
    for image_batch,label_batch in image_data:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3], name="top_input")
    features_extractor_layer.trainable = False

    model = tf.keras.Sequential([
        features_extractor_layer,
        layers.Dense(image_data.num_classes, activation='softmax')
    ])
    model.summary()

    init = tf.global_variables_initializer()
    sess = K.get_session()
    sess.run(init)

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    class CollectBatchStats(tf.keras.callbacks.Callback):
        def __init__(self):
            self.batch_losses = []
            self.batch_acc = []

        def on_batch_end(self, batch, logs=None):
            self.batch_losses.append(logs['loss'])
            self.batch_acc.append(logs['acc'])

    steps_per_epoch = image_data.samples//image_data.batch_size
    print('steps/epoch: ' + steps_per_epoch)
    print('steps: ' + steps_per_epoch * epochs)
    batch_stats = CollectBatchStats()
    model.fit((item for item in image_data), epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks = [batch_stats]
                        validation_split=int(image_data.samples*0.1)

    export_path = tf.contrib.saved_model.save_keras_model(model, os.path.join(prefix, 'model', serving_only=True))  #env.model_dir)
    export_path
