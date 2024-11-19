from abc import ABC, abstractmethod
import torch

from model.variable.layer import Layer


class ReLULayer(Layer):
    """
    Class used to implement a layer with a ReLU activation function

    Methods
    -------
    activate():
        Returns the value of the layer's state, clamped at 0 below
    """

    def activate(self):
        """Returns the value of the layer's state, clamped at 0 below"""
        return self._state.clamp(min=0.)