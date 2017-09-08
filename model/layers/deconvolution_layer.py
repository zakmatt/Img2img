#!/usr/bin/env python3
from keras.layers.convolutional import Deconvolution2D
from layer import Layer

class DeconvolutionLayer(Layer):
    def createLayer(self):
        return Deconvolution2D(
                nb_filter = self.layerParams['ngf'],
                nb_row = self.layerParams['filter_height'],
                nb_col = self.layerParams['filter_width'],
                subsample = self.layerParams['subsample'],
                border_mode = self.layerParams['border_mode'],
                init = self.layerParams['init'],
                input_shape = self.layerParams['input_shape']
                )