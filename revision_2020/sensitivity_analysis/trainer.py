import random
import time
import glob
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, Activation, Flatten, Dense
import dataset
import argparse
import json

class Trainer(object):
    def __init__(self, config):
        self.train_config = config['train']
        self.model_config = config['model']
        self.input_config = config['input']

        self.gene_type = self.model_config['gene_type']
        if self.gene_type == 'HIS':
            self.feature_dim = len(dataset.HIS_fea_names)
        else:
            self.feature_dim = len(dataset.HS_fea_names)

    def _build_model(self):

        activation = 'relu'
        kernel_size = self.model_config.get('kernel_size', 3)
        filters = self.model_config.get('filters', 32)
        block_num = self.model_config.get('block_num', 2)

        input_ = Input(shape=(self.feature_dim, 1))
        x = Conv1D(filters, kernel_size, padding='same',
                   activation=activation)(input_)
        print('block', block_num)
        for _ in range(block_num):
            y = Conv1D(filters,
                       kernel_size,
                       padding='same',
                       activation=activation)(x)
            y = Conv1D(filters, kernel_size, padding='same')(y)
            x = x + y
            x = Activation(activation)(x)

        x = Flatten()(x)
        x = Dense(512, activation=activation)(x)
        x = Dense(1, name='dense1')(x)
        act2 = Activation('sigmoid', name='act2')(x)

        model = keras.Model(input_, act2)

        return model

    def train(self):
        model = self._build_model()
        model.summary()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, mode='auto', restore_best_weights=True)


        train_data, val_data = dataset.get_train_valiate(
            self.input_config, self.gene_type)

        batch_size = self.train_config.get('batch_size', 64)

        model.fit(x=train_data[0],
                  y=train_data[1],
                  batch_size=32,
                  epochs=256,
                  validation_data=val_data,
                  callbacks=[early_stop_callback])

        model_path = f'{self.train_config["base_dir"]}/best_model_{self.gene_type}.h5'
        model.save(model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--gene_type', type=str, required=True) 
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config['model']['gene_type'] = args.gene_type
    config['train']['base_dir'] = args.base_dir

    seed=2020
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    trainer_ = Trainer(config)
    trainer_.train()


if __name__ == '__main__':
    main()
