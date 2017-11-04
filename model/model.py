#!/usr/bin/env python3
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, x_train, params):
        self.x_train = x_train
        self.params = params

    @abstractmethod
    def create_model(self, batch_size=64):
        pass
