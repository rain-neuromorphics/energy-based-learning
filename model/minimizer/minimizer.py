from abc import ABC, abstractmethod
import torch



class LayerUpdater(ABC):
    """
    Abstract class to update the state of a layer
    
    The layer z is updated so as to decrease a function E
    A central quantity used to determine the layer's update is the gradient dE/dz

    Methods
    -------
    grad():
        Computes the gradient of the function to minimize wrt the layer's state
    pre_activate():
        Computes the pre-activation of the layer
    """

    def __init__(self, layer, fn):
        """Creates an instance of LayerUpdater

        Args:
            layer (Layer): the layer to update
            fn (Function): the function to minimize
        """

        self._layer = layer

        self.grad = fn.grad_layer_fn(layer)  # this is a method, not an attribute

    @abstractmethod
    def pre_activate(self):
        """Computes the pre-activation of the layer

        Returns:
            Tensor of shape (batch_size, layer_shape). Type is float32
        """
        pass


class GradientDescentUpdater(LayerUpdater):
    """
    Class to update a layer by gradient descent on the function to minimize

    Attributes
    ----------
    step_size (float): the size of the gradient descent steps

    Methods
    -------
    pre_activate():
        Return the state obtained after one step of gradient descent, starting from the layer's state
    """

    def __init__(self, layer, fn, step_size=0.5):
        """Initializes an instance of Layer

        Args:
            layer (Layer): the layer to update
            fn (Function): the function to minimize
            step_size (float, optional): the step size of the gradient steps. Default: 0.5
        """

        LayerUpdater.__init__(self, layer, fn)

        self._step_size = step_size

    @property
    def step_size(self):
        """Gets and sets the step size for the gradient descent step"""
        return self._step_size

    @step_size.setter
    def step_size(self, step_size):
        self._step_size = step_size

    def pre_activate(self):
        """Return the state obtained after one step of gradient descent, starting from the layer's state

        Returns:
            Tensor. The configuration of the layer after one step of gradient descent (before activation)
        """
        return self._layer.state - self._step_size * self.grad()  # tensor of size (batch_size, layer_shape)



class Minimizer:
    """
    Abstract class for minimizing a scalar function.

    A `minimizer' is an algorithm to minimize a scalar function

    Attributes
    ----------
    num_iterations (int): number of iterations to minimize the function
    mode (str): either 'forward', 'backward', 'synchronous' or 'asynchronous'. `backward' means e.g. with four layers, [3, 2, 1, 0]
    _params (list of Parameter): the parameters (weights and biases) the function depends on
    _layers (list of Layer): the layers the function depends on
    _updaters (list of LayerUpdater): the layer updaters to minimize the function

    Methods
    -------
    compute_equilibrium()
        Minimize the scalar function (let the layers relax to equilibrium)
    compute_trajectory()
        Compute the trajectory as the layers relax to equilibrium
    step()
        Runs one step of the minimization process
    """

    def __init__(self, fn, updaters, num_iterations=15, mode='asynchronous'):
        """Creates an instance of Minimizer

        Args:
            fn (Function): the function to minimize
            updaters (list of LayerUpdater): the layer updaters to minimize the function
            num_iterations (int, optional): number of iterations to converge to equilibrium (a minimum of the function). Default: 15
            mode (str, optional): either 'forward', 'backward', 'synchronous' or 'asynchronous'. Default: 'asynchronous'
        """

        self._layers = fn.layers()
        self._params = fn.params()

        self._updaters = updaters

        self._num_iterations = num_iterations
        self._mode = mode
        self._set_mode()

    @property
    def mode(self):
        """Get and sets the mode ('forward', 'backward', 'synchronous' or 'asynchronous')"""
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode
        self._set_mode()

    @property
    def num_iterations(self):
        """Gets and sets the number of iterations allowed to converge to equilibrium"""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, num_iterations):
        self._num_iterations = num_iterations
        self._set_mode()

    def compute_equilibrium(self):
        """Compute the minimum of the function wrt the free layers

        Performs num_iterations iterations of the minimization process.

        Returns:
            layers: dictionary of Tensors. The state of the layers at equilibrium
        """

        for _ in range(self._num_iterations): self.step()

        layers = {layer.name: layer.state for layer in self._layers}

        return layers

    def compute_trajectory(self):
        """Compute the trajectory of the layers during the minimization process

        Returns:
            trajectory: dictionary of list of Tensors. The layers' and parameters' states at each step of the trajectory
        """
        
        # During the minimization process, we keep all the tensors of the computational graph in a 'trajectories' dictionary
        trajectories = dict()
        for layer in self._layers: trajectories[layer.name] = [layer.state]
        for param in self._params: trajectories[param.name] = [param.state]
        
        for _ in range(self._num_iterations):
            for param in self._params:
                # at each time step of the minimization process, we create a new parameter tensor,
                # this is useful to compute the partial derivative of the loss wrt the parameter at this time step in the backward pass of the backpropagation algorithm
                param.state = param.state + torch.zeros_like(param.state)
                trajectories[param.name].append(param.state)
            # perform one step of the layers' dynamics and store the layers' activations
            self.step()
            for layer in self._layers:
                trajectories[layer.name].append(layer.state)
        
        return trajectories

    def step(self):
        """Runs one step of the equilibration process"""
        
        for layer_group in self._list_layers:
            pre_activations = [updater.pre_activate() for updater in layer_group]
            for updater, pre_activation in zip(layer_group, pre_activations):
                updater._layer.state = pre_activation
                updater._layer.state = updater._layer.activate()
    
    def _set_mode(self):
        """Set the order of the layers to update at each 'step', depending on the 'mode' attribute"""
        mode = self._mode
        if mode == 'forward':
            self._list_layers = [[updater] for updater in self._updaters]
        elif mode == 'backward':
            self._list_layers = [[updater] for updater in reversed(self._updaters)]
        elif mode == 'synchronous':
            self._list_layers = [self._updaters]
        elif mode == 'asynchronous':
            self._list_layers = [self._updaters[::2], self._updaters[1::2]]
        else:
            raise ValueError("expected 'forward', 'backward', 'synchronous' or 'asynchronous' but got {}".format(mode))



class GradientDescentMinimizer(Minimizer):
    """
    Class for minimizing a scalar function by gradient descent
    """

    def __init__(self, fn, free_layers, step_size=0.5, num_iterations=5, mode='asynchronous'):
        """Creates an instance of GradientDescentMinimizer

        Args:
            fn (Function): the function to minimize
            free_layers (list of Layer): the layers wrt which we minimize the function
            step_size (float, optional): the step size of the gradient steps. Default: 0.5
            num_iterations (int, optional): number of iterations of gradient descent performed to converge to equilibrium (a minimum of the scalar function). Default: 5
            mode (str, optional): either 'forward', 'backward', 'synchronous' or 'asynchronous'. Default: 'asynchronous'
        """

        updaters = [GradientDescentUpdater(layer, fn, step_size) for layer in free_layers]

        Minimizer.__init__(self, fn, updaters, num_iterations, mode)

    def __str__(self):
        return 'Gradient descent minimizer -- mode={}, step_size={}, num_iterations={}'.format(self._mode, self._updaters[0].step_size, self._num_iterations)
    

class ParamUpdater:
    """
    Abstract class to update the state of a parameter
    
    The param theta is updated in proprtion to the gradient of a function E
    A central quantity used to determine the param's update is the gradient dE/dtheta

    Methods
    -------
    grad():
        Computes the gradient of the function wrt the param's state
    """

    def __init__(self, param, fn):
        """Creates an instance of ParamUpdater

        Args:
            param (Parameter): the param
            fn (Function): the function
        """

        self._param = param

        self.grad = fn.grad_param_fn(param)  # this is a method, not an attribute
        self.second_fn = fn.second_fn(param)  # this is a method, not an attribute