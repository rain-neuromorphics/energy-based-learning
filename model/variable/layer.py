from abc import ABC, abstractmethod
import torch

from model.variable.variable import Variable



class Layer(Variable, ABC):
    """
    Class used to implement a layer of units

    Attributes
    ----------
    _counter (int): the number of Layers instanciated so far
    name (str): the layer's name (used e.g. to identify the layer in tensorboard)

    Methods
    -------
    activate():
        Applies a nonlinearity to the layer's state
    """

    _counter = 0  # the number of instantiated Layers

    def __init__(self, shape, batch_size=1, device=None):
        """Initializes an instance of Layer

        Args:
            shape (tuple of int): shape of the tensor used to represent the state of the layer
            batch_size (int, optional): the size of the current batch processed. Default: 1
            device (str, optional): the device on which to run the layer's tensor. Either `cuda' or `cpu'. Default: None
        """

        Variable.__init__(self, shape)

        self._name = 'Layer_{}'.format(Layer._counter)  # the name of the layer is numbered for ease of identification

        self.init_state(batch_size, device)
        
        Layer._counter += 1  # the number of instanciated layers is now increased by 1

    @property
    def name(self):
        """Get the name of the layer"""
        return self._name

    def init_state(self, batch_size, device):
        """Initializes the state of the layer to zero

        Args:
            batch_size (int): size of the mini-batch of examples
            device (str): Either 'cpu' or 'cuda'
        """

        shape = (batch_size,) + self._shape
        self._state = torch.zeros(shape, requires_grad=False, device=device)
    
    @abstractmethod
    def activate(self):
        """Applies a nonlinearity to the layer's state"""
        pass



class InputLayer(Layer, ABC):
    """
    Class used to implement an input layer (layer of units for input variables)
    """
    
    def activate(self):
        # FIXME: an input layer does not require an activation function
        pass
    
    def set_input(self, x):
        """Set the input values

        Args:
            x (Tensor): input values
        """
        self._state = x


class LinearLayer(Layer):
    """
    Class used to implement a layer with a linear activation funtion (the identity function)

    Methods
    -------
    activate():
        Returns the value of the layer's state
    """

    def activate(self):
        """Returns the value of the layer's state"""
        return self._state