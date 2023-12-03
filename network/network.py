import torch

from network.variable.layer import LinearLayer
from network.interaction import NudgingInteraction



class Network:
    """
    Class used to implement a network.

    Attributes
    ----------
    input_layer (Layer): input layer of the network
    output_layer (Layer): output layer of the network
    layers (list of Layer): list of ``hidden layers'' of the network, including the output layer
    params (list of Parameter): the parameters (weights and biases) of the network
    interactions (list of Interaction): list of all interactions of the network
    batch_size (int): Size of the mini-batch, i.e. number of examples from the dataset processed simultaneously
    device (str): Either 'cpu' or 'cuda'

    Methods
    -------
    add_layer(layer)
        Adds a layer to the network
    add_param(param, interaction)
        Adds an interaction and the corresponding parameter
    pack(idx_output_layer, idx_input_layer)
        Tells the network which layers are the input layer and the output layer
    params()
        Returns the list of parameter variables of the network
    layers()
        Returns the list of layers of the network
    init_layers()
        Initializes the layers of the network (to zero)
    set_device(self, device)
        Set the tensors of the network on a given device
    set_input(x, reset):
        Set the input layer to input values x, and reset the state of the network to zero if reset=True
    set_output(output):
        Set the output layer to output values
    set_output_force(force)
        Set the force of output nodes to a desired value (e.g. output errors)
    reset_output_force()
        Set the force of output nodes to zero
    read_output():
        Read the prediction of the network (the state of the output layer)
    primitive_fn()
        Returns the primitive value of the current configuration
    energy_fn()
        Returns the energy value of the current configuration
    relaxation(num_iterations)
        Let the network relax to equilibrium
    save(model_path)
        Saves the model
    load(model_path)
        Loads the model parameters
    """

    def __init__(self, device=None):
        """Creates an instance of Network.

        The network is initially empty: there is no layer (and no interaction between the layers).

        Args:
            device (str, optional): The device where the tensors of the network are set. Either 'cpu' or 'cuda'. Default: None
        """

        self._batch_size = 1  # FIXME
        self._device = device

        self._layers = []
        self._params = []
        self._interactions = []
        
    def add_layer(self, layer):
        """Adds a layer to the network

        Args:
            layer (Layer): the layer that we add to the network
        """

        self._layers.append(layer)

    def add_param(self, param):
        """Adds a parameter (weight or bias) to the network

        Args:
            param (Parameter): the parameter that we add to the network
        """

        self._params.append(param)

    def add_interaction(self, interaction):
        """Adds an interaction to the network

        Args:
            interaction (Interaction)
        """

        self._interactions.append(interaction)

    def pack(self, idx_output_layer=-1, idx_input_layer=0):
        """Tells the network which layers are the input and output layers

        Args:
            idx_output_layer (int, optional): the index of the layer that plays the role of output layer. Default: -1
            idx_input_layer (int, optional): the index of the layer that plays the role of input layer. Default: 0
        """

        # TODO: Before calling this method, one should not be able to use the relaxation method
        # TODO: After calling this method, one should no longer be able to use the add_layer and add_interaction methods

        self._input_layer = self._layers[idx_input_layer]
        self._output_layer = self._layers[idx_output_layer]

        self._output_force = LinearLayer(self._output_layer.shape, device=self._device)
        output_interaction = NudgingInteraction(self._output_layer, self._output_force)
        self._interactions.append(output_interaction)

        self._input_layer.is_free = False  # the input's layer value is set (not free to stabilize to equilibrium)

    def params(self):
        """Returns the list of parameter variables of the network"""

        return self._params

    def layers(self, free_only=True):
        """Returns the list of layers of the network

        Args:
            free_only (bool, optional): whether we return the free (floating, i.e. not clamped) layers only, or all the layers of the network. Default: True
        """

        if free_only: return [layer for layer in self._layers if layer.is_free]
        else: return self._layers

    def init_layers(self):
        """Initialize the layers of the network to zero"""

        for layer in self.layers(): layer.init_state(self._batch_size, self._device)

    def set_device(self, device):
        """Set the tensors of the network on a given device

        Args:
            device (str): the name of the device (e.g. 'cpu' or 'cuda')
        """

        self._device = device

        for layer in self._layers: layer.set_device(device)
        for param in self._params: param.set_device(device)

    def set_input(self, input_values, reset=False):
        """Set the input layer to input values

        Args:
            input_values: input image. Tensor of shape (batch_size, channels, width, height). Type is float32.
            reset (bool, optional): if True, resets the state of the network to zero.
        """

        # we set the input tensor on the network's device
        self._input_layer.state = input_values.to(self._device)

        # we set the state of the network to zero if reset is True, or if the size of the batch of examples is different from the previous batch size.
        batch_size = input_values.size(0)
        if reset or self._batch_size != batch_size:
            self._batch_size = batch_size
            self.init_layers()

    def set_output(self, output):
        """Set the output layer to output values
        
        This method is used to set the output values in Contrastive Learning.

        Args:
            output (Tensor): tensor of shape (batch_size, output_size). Type is float32.
        """

        # TODO: check that the shape of the output tensor is correct

        self._output_layer.state = output.to(self._device)

    def set_output_force(self, force):
        """Set the force of output nodes (used to set the output error values in equilibrium propagation)

        Args:
            force (Tensor): tensor of shape (batch_size, output_size). Type is float32.
        """

        # TODO: check that the shape of the force tensor is the same as the output layer

        self._output_force.state = force.to(self._device)

    def reset_output_force(self):
        """Set the force of output nodes to zero"""
        self._output_force.state = torch.zeros_like(self._output_layer.state)

    def read_output(self):
        """Read (i.e. return) the state of the output layer

        Returns:
            Vector of size (batch_size, output_size) and of type int: each row is the output state for the corresponding example in the mini-batch
        """

        return self._output_layer.state

    def primitive_fn(self):
        """Returns the primitive values of the current configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of primitive values for each of the examples in the current mini-batch
        """

        return sum([interaction.primitive_fn() for interaction in self._interactions])

    def energy_fn(self):
        """Returns the energy values of the current configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of energy values for each of the examples in the current mini-batch
        """

        # FIXME: the energy function misses the energy terms of the layers

        return - self.primitive_fn()

    def relaxation(self, num_iterations):
        """Let the layers of the network relax to equilibrium

        Performs num_iterations iterations of the network's dynamics.

        Args:
            num_iterations (int): Number of iterations of the dynamics performed (to converge to equilibrium)
        """

        for _ in range(num_iterations): self._step_asynchronous()

    def step_synchronous(self):
        """Runs one step of the network's dynamics, wherein all the layers are updated simultaneously"""

        pre_activations = [layer.pre_activate() for layer in self.layers()]
        for layer, pre_activation in zip(self.layers(), pre_activations):
            layer.state = pre_activation
            layer.state = layer.activate()

    def _step_asynchronous(self, backward=False):
        """Runs one step of the network's dynamics, wherein the even layers are updated first, and the odd layers are updated next.

        The order of relaxation of the layers may be `forward' or `backward': e.g. with four layers, either [0, 2, 1, 3] (forward: even layers first, and odd layers next) or [1, 3, 0, 2] (backward: odd layers first, and even layers next)

        Args:
            backward (bool, optional): if False, relaxes the layers in the 'forward' direction ; if True, relaxes the layers in the 'backward' direction. Default: False.
        """

        # TODO: this implementation is customized for a layered network without skip-layer connection

        list_layers = self.layers()
        list_layers = list_layers[::2] + list_layers[1::2]  # asynchronous update procedure, with even indices first, and odd indices next
        list_layers = reversed(list_layers) if backward else list_layers
        
        for layer in list_layers:
            layer.state = layer.pre_activate()
            layer.state = layer.activate()

    def save(self, path):
        """Saves the model parameters

        Args:
            path (str): path where to save the network's parameters
        """

        params = [param.state for param in self._params]
        torch.save(params, path)

    def load(self, path):
        """Loads the model parameters

        Args:
            path (str): path where to load the network's parameters from
        """

        params = torch.load(path, map_location=torch.device(self._device))
        for param, state in zip(self._params, params): param.state = state