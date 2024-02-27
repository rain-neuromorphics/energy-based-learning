from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from model.function.interaction import QFunction



class DenseResistive(QFunction):
    """Dense resistive interaction between two layers

    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer.
    _layer_post (Layer): post-synaptic layer.
    _weight (DenseWeight): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.
    """

    def __init__(self, layer_pre, layer_post, dense_weight):
        """Initializes an instance of DenseResistive

        Args:
            layer_pre (Layer): pre-synaptic layer
            layer_post (Layer): post-synaptic layer
            dense_weight (DenseWeight): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = dense_weight

        QFunction.__init__(self, [layer_pre, layer_post], [dense_weight])

    def eval(self):
        """Computes the energy term corresponding to this weight tensor.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state  # / self._layer_post.gain
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        for _ in range(dims_post): layer_pre = layer_pre.unsqueeze(-1)  # broadcast layer_pre to (batch_size, shape_pre, shape_post)
        for _ in range(dims_pre): layer_post = layer_post.unsqueeze(1)  # broadcast layer_post to (batch_size, shape_pre, shape_post)
        weight = self._weight.get().unsqueeze(0)  # broadcast weight to (batch_size, shape_pre, shape_post)
        return 0.5 * ((layer_pre - layer_post)**2).mul(weight).flatten(start_dim=1).sum(dim=1)

    def a_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
            }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
            }
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]

    def _b_coef_layer_pre(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)  # number of dimensions involved in the tensor product
        weight = self._weight.get()
        dim_weight = len(weight.shape)
        permutation = tuple(range(dims_pre, dim_weight)) + tuple(range(dims_pre))
        b_coef = - torch.tensordot(layer_post, weight.permute(permutation), dims=dims_post)  # / self._layer_post.gain
        return b_coef

    def _a_coef_layer_pre(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        dims_pre = len(self._layer_pre.shape)
        a_coef = 0.5 * self._weight.get().flatten(start_dim=dims_pre).sum(dim=-1).unsqueeze(0)

        return a_coef

    def _b_coef_layer_post(self):
        """Returns the interaction's linear influence on the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the linear contribution on layer_post
        """

        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.shape)  # number of dimensions involved in the tensor product
        b_coef = - torch.tensordot(layer_pre, self._weight.get(), dims=dims_pre)  # * self._layer_post.gain
        return b_coef

    def _a_coef_layer_post(self):
        """Returns the interaction's quadratic influence on the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the quadratic contribution on layer_post
        """

        dims = len(self._layer_pre.shape) - 1
        a_coef = 0.5 * self._weight.get().flatten(end_dim=dims).sum(dim=0).unsqueeze(0)

        return a_coef

    def _grad_weight(self):
        """Returns the interaction's gradient wrt the weight

        Returns:
            Tensor of shape weight_shape and type float32: the gradient wrt the weights
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state  # / self._layer_post.gain
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        for _ in range(dims_post): layer_pre = layer_pre.unsqueeze(-1)
        for _ in range(dims_pre): layer_post = layer_post.unsqueeze(1)
        grad_weight = 0.5 * ((layer_pre - layer_post)**2).mean(dim=0)

        return grad_weight