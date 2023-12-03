from abc import ABC, abstractmethod
import torch

from network.variable.variable import Variable



class Layer(Variable, ABC):
    """
    Class used to implement a layer of a network.

    A layer interacts with other layers and parameters via objects of the abstract class Interaction.

    Attributes
    ----------
    _counter (int): the number of Layers instanciated so far
    name (str): the layer's name (used e.g. to identify the layer in tensorboard)
    is_free (bool): whether the layer is free to to stabilize to equilbrium (True) or set to a specific value (False)

    Methods
    -------
    pre_activate():
        Computes the pre-activation of the layer
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
        self._is_free = True

        self.init_state(batch_size, device)
        
        Layer._counter += 1  # the number of instanciated layers is now increased by 1

    @property
    def name(self):
        """Get the name of the layer"""
        return self._name

    @property
    def is_free(self):
        """Get and set the `is_free' status of the layer"""
        return self._is_free

    @is_free.setter
    def is_free(self, is_free):
        self._is_free = is_free

    def init_state(self, batch_size, device):
        """Initializes the state of the layer to zero

        Args:
            batch_size (int): size of the mini-batch of examples
            device (str): Either 'cpu' or 'cuda'
        """

        shape = (batch_size,) + self._shape
        self._state = torch.zeros(shape, requires_grad=False, device=device)

    @abstractmethod
    def pre_activate(self):
        """Computes the pre-activation of the layer

        Returns:
            Tensor of shape (batch_size, layer_shape). Type is float32
        """
        pass
    
    @abstractmethod
    def activate(self):
        """Applies a nonlinearity to the layer's state"""
        pass



class HopfieldLayer(Layer, ABC):
    """
    Class used to implement a Hopfield layer.

    A layer interacts with other layers and parameters via objects of the abstract class Interaction.
    Each interaction is defined by its `primitive function' Phi_i, which is a function of the layer's state z, as well as other variables.
    The layer's `total primitive function' is Phi = sum_i Phi_i.
    A central quantity determining the time evolution of the layer's state is the `gradient' dPhi/dz = sum_i dPhi_i/dz

    Attributes
    ----------
    _grad_fns (list of functions): list of functions computing the gradients of the layer's interactions

    Methods
    -------
    add_interaction(grad_fn):
        Adds an interaction the layer is involved in (adds a function that computes dPhi_i/dz, the gradient of the interaction's primitive function wrt the layer's state)
    pre_activate():
        Computes the gradient of the primitive function wrt the layer, i.e. dPhi/dz, where Phi = sum_i Phi_i is the primitive function and z is the layer
    """

    _counter = 0  # the number of instantiated Layers

    def __init__(self, shape, batch_size=1, device=None):
        """Initializes an instance of Layer

        Args:
            shape (tuple of int): shape of the tensor used to represent the state of the layer
            batch_size (int, optional): the size of the current batch processed. Default: 1
            device (str, optional): the device on which to run the layer's tensor. Either `cuda' or `cpu'. Default: None
        """

        Layer.__init__(self, shape, batch_size, device)

        self._grad_fns = []

    def add_interaction(self, grad_fn):
        """Adds an interaction to the layer.

        Args:
            grad_fn (fn): function that computes the gradient of the interaction added
        """

        # TODO: the method should check that the layer is involved in the interaction

        if grad_fn: self._grad_fns.append(grad_fn)

    def pre_activate(self):
        """Computes the gradient of the primitive function wrt the layer's state, i.e. dPhi/dz, where Phi is the primitive function and z is the layer's state

        This gradient is equal to dPhi/dz = sum_i dPhi_i/dz

        Returns:
            Tensor of shape (batch_size, layer_shape). Type is float32
        """

        return sum([fn() for fn in self._grad_fns])  # tensor of size (batch_size, layer_shape)


class LinearLayer(HopfieldLayer):
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


class HardSigmoidLayer(HopfieldLayer):
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