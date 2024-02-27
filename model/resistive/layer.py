import torch

from model.variable.layer import Layer, InputLayer



class ResistiveInputLayer(InputLayer):
    """
    Class used to implement an input layer of a resistive network

    Attributes
    ----------
    gain (float): the 'gain' (or 'amplification factor') by which input variables are multiplied
    """

    def __init__(self, shape, gain, batch_size=1, device=None):
        """Creates an instance of ResistiveInputLayer

        Args:
            shape (tuple of int): shape of the tensor used to represent the state of the layer
            gain (float32): amplification factor by which input variables are multiplied
            batch_size (int, optional): the size of the current batch processed. Default: 1
            device (str, optional): the device on which to run the layer's tensor. Either `cuda' or `cpu'. Default: None
        """

        InputLayer.__init__(self, shape, batch_size=batch_size, device=device)

        self._gain = gain

    
    def set_input(self, input_values):
        """Set the input values

        We duplicate the inputs and invert one set (method used to overcome the constraint of non-negative weights in resistive networks)
        We then multiply the inputs by the gain (amplification factor)

        Args:
            input_values (Tensor): input values
        """
        self._state = self._gain * torch.cat((input_values, -input_values), 1)



class NonlinearResistiveLayer(Layer):
    """
    Class used to implement a nonlinear resistive layer (a resistive layer with diodes to implement nonlinearities)

    Methods
    -------
    activate():
        Returns the value of the variable's state, clamped between 0 and +infinity for excitatory units, and between -infinity and 0 for inhibitory units
    """

    def activate(self):
        """Returns the value of the layer's state, clamped between 0 and +infinity for excitatory units, and clamped between -infinity and 0 for inhibitory units"""
        
        dimension = self._shape[0] // 2  # number of excitatory units = number of inhibitory units = number of units / 2
        excitatory = self._state[:,:dimension].clamp(min=0., max=None)  # the first half of the units are excitatory units
        inhibitory = self._state[:,dimension:].clamp(min=None, max=0.)  # the second half of the units are inhibitory units
        return torch.cat((excitatory, inhibitory), 1)  # we concatenate excitatory and inhibitory units