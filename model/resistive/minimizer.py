from model.minimizer.minimizer import LayerUpdater, Minimizer


class QuadraticUpdater(LayerUpdater):
    """
    Class to update a layer assuming to the function to minimize is quadratic.
    
    We assume the function E to minimize is a quadractic function of the layer z,
    E(z) = a z^2 + b z + c, for some coefficients a, b and c. Furthermore we assume that the coefficient a is positive.
    A `quadratic update' sets the layer's pre-activation to - b/2a, 

    Methods
    -------
    pre_activate():
        Computes the value of the layer that achieves the minimum of the function, given other variables fixed.
    """

    def __init__(self, layer, fn):
        """Creates an instance of LayerUpdater

        Args:
            layer (Layer): the layer to update
            fn (Function): the function to minimize
        """

        self._layer = layer

        self._a = fn.a_coef_fn(layer)  # this is a method, not an attribute
        self._b = fn.b_coef_fn(layer)  # this is a method, not an attribute

    def pre_activate(self):
        """Computes the value of the layer that achieves the minimum of the function, given other variables fixed.
        
        It is assumed that all interactions the layer is involved in are quadratic in the layer's state, i.e. the interaction's energy is of the form
        E_i(z) = a_i z^2 + b_i z + c_i, where z is the layer's state.
        Thus, the global energy of the network is also quadratic in the layer's state, i.e. of the form E(z) = a * z^2 + b * z + c,
        with coefficients a = sum_i a_i and b = sum_i b_i.
        The minimum of this quadractic function in R is obtained at the point z = - b / 2*a (pre-activation)

        Note that the minimum in [min_interval, max_interval] is obtained by clipping - b / 2*a between min_interval and max_interval (activation)

        Returns:
            Tensor of shape (batch_size, layer_shape). Type is float32
        """

        b = self._b()
        a = self._a()
        return - b / (2. * a)
    

class QuadraticMinimizer(Minimizer):
    """
    Class for minimizing a function by a coordinate descent method with quadratic updates
    """

    def __init__(self, fn, free_layers, num_iterations=15, mode='asynchronous'):
        """Creates an instance of Minimizer

        Args:
            fn (Function): the function to minimize
            free_layers (list of Layer): the layers wrt which we minimize the function
            num_iterations (int, optional): number of iterations to converge to equilibrium (a minimum of the function). Default: 15
            mode (str, optional): either 'forward', 'backward', 'synchronous' or 'asynchronous'. Default: 'asynchronous'
        """

        updaters = [QuadraticUpdater(layer, fn) for layer in free_layers]

        Minimizer.__init__(self, fn, updaters, num_iterations, mode)

    def __str__(self):
        return 'Quadratic minimizer -- mode={}, num_iterations={}'.format(self._mode, self._num_iterations)