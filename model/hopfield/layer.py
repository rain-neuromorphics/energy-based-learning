from abc import ABC
import torch

from model.variable.layer import Layer


class HardSigmoidLayer(Layer):
    """
    Class used to implement a layer with a hard-sigmoid activation function

    Methods
    -------
    activate():
        Returns the value of the layer's state, clamped between 0 and 1
    """

    def activate(self):
        """Returns the value of the layer's state, clamped between 0 and 1"""
        return self._state.clamp(min=0., max=1.)


class SigmoidLayer(Layer):
    """
    Class used to implement a layer with a sigmoid activation function ('logistic function')

    Methods
    -------
    activate():
        Applies the logistic function to the layer's state and returns the result
    """

    def activate(self):
        """Returns the logistic function applied to the layer's state"""
        return torch.sigmoid(4 * self._state - 2)


class SoftMaxLayer(Layer):
    """
    Class used to implement a layer with a `softmax' activation function

    Methods
    -------
    activate():
        Applies the `softmax' to the layer's state and returns the result
    """

    def activate(self):
        """Returns the softmax function applied to the layer's state"""
        # FIXME: computing the cross-entropy loss requires computing the log first.
        # Directly computing the log_softmax would be faster and would have better numerical properties.
        return torch.nn.functional.softmax(self._state)


class dSiLULayer(Layer):
    """
    Class used to implement a layer with the derivative of the sigmoid-weighted linear activation function

    Methods
    -------
    activate():
        Applies the sigmoid-weighted linear function to the layer's state and returns the result
    """

    def activate(self):
        """Returns the sigmoid-weighted linear function applied to the layer's state"""
        return self._auxiliary(self._state) - self._auxiliary(self._state - 1.)

    def _auxiliary(self, state):
        return torch.sigmoid(3 * state) * state


class DropOutLayer(Layer):
    """
    Class used to implement a layer with dropout

    Attributes
    ----------
    _layer (Layer): the layer on which we apply dropout
    _sparsity (float): the level of sparsity of the layer

    Methods
    -------
    activate():
        Activate the dropout layer
    draw_mask():
        Draw a new mask for dropout
    """

    def __init__(self, layer, sparsity=0.5):
        """Initializes an instance of Layer

        Args:
            layer (Layer): the layer on which we want to apply dropout
            parsity (float, optional): the level of sparsity of the layer. Default: 0.5
        """

        self._layer = layer
        self._sparsity = sparsity

    def activate(self):
        """Activate the dropout layer"""
        return 1./(1.-self._sparsity) * self._mask * self._layer.activate()
    
    def draw_mask(self):
        """Draw a new mask for dropout"""
        self._mask = (torch.rand(*self._layer.state._shape) > self._sparsity).float()



'''class MyLayer(Layer):
    """
    Class used to implement a layer with a smooth sigmoidal activation function

    Methods
    -------
    activate():
        Applies a sigmoidal function to the layer's state and returns the result
    """

    def activate(self):
        """Returns a sigmoidal function applied to the layer's state"""
        state = self._state.clamp(min=0., max=1.)
        state_low = 2. * (state ** 2)
        state_high = 1. - 2. * ((1.-state) ** 2)
        return torch.where(state < 0.5, state_low, state_high)'''