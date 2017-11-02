#!/usr/bin/env python3
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, layer_input, layer_params):
        self.layer_input = layer_input
        self.layer_params = layer_params

    @abstractmethod
    def create_layer(self):
        self.current_layer=None

    def process(self, activation=None, batch_norm=None, dropout=None):
        self.create_layer()
        convolved_output = self.current_layer(self.layer_input)

        processed_output = None

        if batch_norm is not None:
            convolved_output = batch_norm(convolved_output)
        if dropout is not None:
            convolved_output = dropout(convolved_output)

        if activation is not None:
            processed_output = activation(convolved_output)

        return (convolved_output, processed_output) if processed_output is not None else convolved_output
