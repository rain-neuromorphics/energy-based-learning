from model.minimizer.minimizer import LayerUpdater, Minimizer


class HopfieldLayerUpdater(LayerUpdater):
    """
    Class to update a layer according to a `Hopfield update'.
    
    A `Hopfield update' sets the layer's pre-activation to the negative of the function's gradient.

    Methods
    -------
    pre_activate():
        Computes the negative of the gradient of the function wrt the layer, i.e. - dE/dz, where E is the (Hopfield energy) function and z is the layer's state
    """

    def pre_activate(self):
        """Computes the negative of the gradient of the (Hopfield energy) function wrt the layer's state, i.e. dE/dz, where E is the Hopfield energy function and z is the layer's state

        Returns:
            Tensor of shape (batch_size, layer_shape). Type is float32
        """

        return - self.grad()


class FixedPointMinimizer(Minimizer):
    """
    Class for minimizing a function by a fixed point method
    """

    def __init__(self, fn, free_layers, num_iterations=15, mode='asynchronous'):
        """Creates an instance of Minimizer

        Args:
            fn (Function): the function to minimize
            free_layers (list of Layer): the layers wrt which we minimize the function
            num_iterations (int, optional): number of iterations to converge to equilibrium (a minimum of the function). Default: 15
            mode (str, optional): either 'forward', 'backward', 'synchronous' or 'asynchronous'. Default: 'asynchronous'
        """

        updaters = [HopfieldLayerUpdater(layer, fn) for layer in free_layers]

        Minimizer.__init__(self, fn, updaters, num_iterations, mode)

    def __str__(self):
        return 'Fixed point minimizer -- mode={}, num_iterations={}'.format(self._mode, self._num_iterations)