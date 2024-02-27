from abc import ABC, abstractmethod
import numpy as np
import torch

from model.variable.variable import Variable



class Parameter(Variable, ABC):
    """Abstract class for parameter variables

    Attributes
    ----------
    _non_negative (bool): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False)

    Methods
    -------
    get()):
        Returns the state of the parameter
    clamp_():
        Clamps the parameter's state in its range of permissible values (in place operation)
    """

    def __init__(self, shape, device, non_negative=False):
        """Initializes an instance of Parameter

        Args:
            shape (tuple of ints): Shape of the tensor used to represent the parameter. Type is float32.
            device (str): Either 'cpu' or 'cuda'.
            non_negative (bool, optional): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False). Default: False
        """

        Variable.__init__(self, shape)

        self._state = torch.empty(*shape, dtype=torch.float32, device=device)
        self._non_negative = non_negative

    def get(self):
        """Returns the state of the parameter"""
        return self._state

    def clamp_(self):
        """Clamps the parameter's state in its range of permissible values (in place operation)"""
        if self._non_negative: self._state.clamp_(min=0., max=None)


class Bias(Parameter):
    """Class for biases

    Methods
    -------
    init_state(gain):
        Initializes the bias tensor
    """

    _counter = 0

    def __init__(self, shape, gain, device):
        """Initializes an instance of Bias

        Args:
            shape (tuple of ints): Shape of the bias Tensor. Type is float32.
        """

        Parameter.__init__(self, shape, device=device)

        self.init_state(gain)

        self.name = 'Bias_{}'.format(Bias._counter)

        Bias._counter += 1

    def init_state(self, gain):
        """Initializes the bias tensor to zero, i.e. b=0."""

        # TODO: implement recommended initialization schemes for biases, instead of zero

        # torch.nn.init.constant_(self._state, 0.)
        torch.nn.init.uniform_(self._state, -gain, +gain)


class DenseWeight(Parameter):
    """Class for dense ('fully connected') weights

    Methods
    -------
    init_state(gain, mode):
        Initializes the weight tensor
    """

    _counter = 0

    def __init__(self, layer_pre_shape, layer_post_shape, gain, device, clamp=False):
        """Initializes an instance of DenseWeight

        Args:
            layer_pre_shape (tuple of ints): shape of the pre-synaptic layer
            layer_post_shape (tuple of ints): shape of the post-synaptic layer
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            clamp (bool, optional): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False). Default: False
        """

        shape = layer_pre_shape + layer_post_shape
        Parameter.__init__(self, shape, device=device, non_negative=clamp)

        self._layer_pre_shape = layer_pre_shape
        self._layer_post_shape = layer_post_shape

        self.init_state(gain)
        self.clamp_()

        self.name = 'DenseWeight_{}'.format(DenseWeight._counter)

        DenseWeight._counter += 1

    def init_state(self, gain, mode='kaiming_uniform'):
        """Initializes the weight tensor according to a uniform or normal distribution.
        Args:
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            mode (str, optional): method to initialize the weight tensor. Either 'xavier_uniform', 'xavier_normal', 'kaiming_uniform' or 'kaiming_normal'. Default: 'xavier_uniform'.
        """

        size_pre = 1
        for dim in self._layer_pre_shape: size_pre *= dim
        size_post = 1
        for dim in self._layer_post_shape: size_post *= dim
        
        if mode == 'xavier_uniform':
            # half xavier uniform
            scale = gain * 0.5 * np.sqrt(6. / (size_pre + size_post))
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif mode == 'xavier_normal':
            # half xavier normal
            scale = gain * 0.5 * np.sqrt(2. / (size_pre + size_post))
            torch.nn.init.normal_(self._state, std=scale)
        elif mode == 'kaiming_uniform':
            # half kaiming uniform
            # scale = gain * 0.5 * np.sqrt(3. / size_pre)
            scale = gain * np.sqrt(1. / size_pre)
            torch.nn.init.uniform_(self._state, -scale, +scale)
        else:  #  mode == 'kaiming_normal'
            # half kaiming normal
            scale = gain * 0.5 * np.sqrt(1. / size_pre)
            torch.nn.init.normal_(self._state, std=scale)


class ConvWeight(Parameter):
    """Class for convolutional weights

    Methods
    -------
    init_state(gain, mode):
        Initializes the convolutional weight tensor
    """

    _counter = 0

    def __init__(self, shape, gain, device, clamp=False):
        """Initializes an instance of ConvWeight

        Args:
            shape (tuple of ints): shape of the convolutional weight tensor. Shape is (out_channels, in_channels, height, width).
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            clamp (bool, optional): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False). Default: False
        """

        Parameter.__init__(self, shape, device=device, non_negative=clamp)

        self.init_state(gain)
        self.clamp_()

        self.name = 'ConvWeight_{}'.format(ConvWeight._counter)

        ConvWeight._counter += 1

    def init_state(self, gain, mode='kaiming_uniform'):
        """Initializes the weight tensor.

        Args:
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            mode (str, optional): method to initialize the weight tensor. Either 'xavier_uniform', 'xavier_normal', 'kaiming_uniform' or 'kaiming_normal'. Default: 'kaiming_normal'.
        """

        (channels_out, channels_in, width, height) = self._shape
        size_pre = channels_in * width * height
        size_post = channels_out
        
        if mode == 'xavier_uniform':
            # half xavier uniform
            scale = gain * 0.5 * np.sqrt(6. / (size_pre + size_post))
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif mode == 'xavier_normal':
            # half xavier normal
            scale = gain * 0.5 * np.sqrt(2. / (size_pre + size_post))
            torch.nn.init.normal_(self._state, std=scale)
        elif mode == 'kaiming_uniform':
            # half kaiming uniform
            # scale = gain * 0.5 * np.sqrt(3. / size_pre)
            scale = gain * np.sqrt(1. / size_pre)
            torch.nn.init.uniform_(self._state, -scale, +scale)
        else:  #  mode == 'kaiming_normal'
            # half kaiming normal
            scale = gain * 0.5 * np.sqrt(1. / size_pre)
            torch.nn.init.normal_(self._state, std=scale)