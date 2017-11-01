#!/usr/bin/env python3
from keras.layers.convolutional import Conv2D
from layers.layer import Layer


class ConvolutionLayer(Layer):
    def create_layer(self):
        self.current_layer = Conv2D(
            filters=self.layer_params['ngf'],
            kernel_size=self.layer_params['filter_size'],
            strides=self.layer_params['strides'],
            padding=self.layer_params['padding'],
            kernel_initializer=self.layer_params['kernel_initializer'],
            input_shape=self.layer_params['input_shape']
        )
