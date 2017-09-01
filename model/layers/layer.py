#!/usr/bin/env python3
from abc import ABC, abstractmethod

class Layer(ABC):
    
    def __init__(self, layerInput, layerParams):
        self.input = layerInput
        self.layerParams = layerParams
    
    @abstractmethod   
    def createLayer(self):
        pass
    
    def process(self, activation, batchNorm = None, dropout = None):
        currentLayer = self.createLayer()
        
        layerOutput = currentLayer(self.layerInput)
        if batchNorm is not None:
            layerOutput = batchNorm(layerOutput)
        if dropout is not None:
            layerOutput = dropout(layerOutput)
            
        return layerOutput