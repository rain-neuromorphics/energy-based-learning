import copy
import torch



class Network():
    """
    Class used to implement a network.

    Attributes
    ----------
    input_layer (Layer): input layer of the network

    Methods
    -------
    set_input(x, reset):
        Set the input layer to input values x, and reset the state of the network to zero if reset=True
    free_layers():
        Return the list of free layers (free to stabilize to equilibrium)
    """

    def __init__(self, function):
        """Creates an instance of Network.

        Args:
            function (Function): the function used as an energy function
            idx_input_layer (int, optional): the index of the layer that plays the role of input layer. Default: 0
        """

        self._function = function

        self._input_layer = function.layers()[0]

        self._free_layers = function.layers()[1:]

    def free_layers(self):
        """Return the list of free layers"""
        return self._free_layers

    def set_input(self, input_values, reset=False):
        """Set the input layer to input values

        Args:
            input_values: input image. Tensor of shape (batch_size, channels, width, height). Type is float32.
            reset (bool, optional): if True, resets the state of the network to zero.
        """

        old_batch_size = self._input_layer.state.size(0)
        batch_size = input_values.size(0)
        
        # we set the input tensor on the network's device
        self._input_layer.set_input(input_values.to(self._function._device))  # FIXME

        # we set the state of the network to zero if reset is True, or if the size of the batch of examples is different from the previous batch size.
        if reset or batch_size != old_batch_size:
            for layer in self._free_layers:
                layer.init_state(batch_size, self._function._device)  # FIXME