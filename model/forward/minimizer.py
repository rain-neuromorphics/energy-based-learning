from model.minimizer.minimizer import LayerUpdater, Minimizer


class ForwardUpdater(LayerUpdater):
    """
    Class to update a layer by forward computation

    Methods
    -------
    pre_activate():
        Computes the forward state
    """

    def __init__(self, layer, fn):
        """Creates an instance of ForwardUpdater

        Args:
            layer (Layer): the layer to update
            fn (Function): the forward-compatible function to minimize
        """

        LayerUpdater.__init__(self, layer, fn)

        self._forward = fn.forward_fn(layer)

    def pre_activate(self):
        """Return the configuration (pre-activation) of the layer obtained by forward computation,
        i.e. the configuration that minimizes the forward-compatible function

        Returns:
            Tensor of shape (batch_size, layer_shape). Type is float32
        """
        return self._forward()  # tensor of size (batch_size, layer_shape)


class ForwardPass(Minimizer):
    """
    Class for minimizing a feedforward-compatible function

    Perform a forward pass to minimize a sum-separable feedforward-compatible function
    """

    def __init__(self, fn, free_layers):
        """Creates an instance of ForwardPass

        Args:
            fn (Function): the feedforward-compatible function to minimize
            free_layers (list of Layer): the layers wrt which we minimize the feedforward-compatible function
        """

        updaters = [ForwardUpdater(layer, fn) for layer in free_layers]

        Minimizer.__init__(self, fn, updaters, num_iterations=1, mode='forward')

    def __str__(self):
        return 'Forward Pass Minimizer'