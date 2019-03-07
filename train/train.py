#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers
import tensorflow.keras.backend as K

import argparse
import os
import json


from environment import create_trainer_environment

if __name__ =='__main__':
    feature_extractor_url = "models/mobilenet_v2_100_224-feature_vector-2"
    
    train = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
    #test = env.channel_dirs['test']
    epochs = 1
    
    data_root = tf.keras.utils.get_file(
            'flower_photos', train,
        untar=True)

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

    features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])
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
    batch_stats = CollectBatchStats()
    model.fit((item for item in image_data), epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks = [batch_stats])

    export_path = tf.contrib.saved_model.save_keras_model(model, 'testing_saved', serving_only=True)
    export_path
