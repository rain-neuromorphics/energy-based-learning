from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from model.function.interaction import Function



class ForwardInteraction(Function, ABC):
    """Abstract class for forward interactions

    A forward interaction is defined by a layer and a forward function
    The interaction's energy function is the square of the norm between the layer's state and the output of the forward function
    
    Attributes
    ----------
    _layer (Layer): the layer. Tensor of shape (batch_size, layer_shape). Type is float32.

    Methods
    -------
    forward():
        Returns forward pre-activation
    """

    def __init__(self, layer):
        """Creates an instance of ForwardInteraction

        Args:
            layer (Layer): the layer
        """

        self._layer = layer

    def eval(self):
        """Returns the value of the energy function of the forward interaction.

        The energy function of a forward interaction is the square of the norm between the layer's state and the output of the forward function
        
        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer = self._layer.state  # store the layer's state

        self._layer.state = self.forward()
        layer_new = self._layer.activate()

        self._layer.state = layer  # reset the layer's state

        return ((layer - layer_new)**2).flatten(start_dim=1).sum(dim=1)  # (forward() - layer)^2
    
    def forward_fn(self, layer):
        """ """
        if layer == self._layer: return self.forward
        else: return None
    
    @abstractmethod
    def forward(self):
        """Returns the forward pre-activation.

        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32
        """
        pass


class DenseForward(ForwardInteraction):
    """Dense ('fully connected') forward interaction between two layers

    A dense forward interaction is defined between four variables:
    - layer_pre,
    - layer_post,
    - the weight between layer_pre and layer_post
    - the bias of layer_post
    
    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_pre_shape). Type is float32.
    _layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_post_shape). Type is float32.
    _weight (DenseWeight): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.
    _bias (Bias): bias tensor of layer_post. Tensor of shape layer_post_shape. Type is float32.
    """

    def __init__(self, layer_pre, layer_post, weight, bias):
        """Creates an instance of DenseForward

        Args:
            layer_pre (Layer): pre-synaptic layer
            layer_post (Layer): post-synaptic layer
            weight (DenseWeight): the dense weights between the pre- and post-synaptic layers
            bias (Bias): the bias of the post-synaptic layer
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = weight
        self._bias = bias

        ForwardInteraction.__init__(self, layer_post)
        Function.__init__(self, [layer_pre, layer_post], [weight, bias])
    
    def forward(self):
        """Returns the forward pre-activation.

        This is the usual activation(layer_pre * weight + bias) for a dense layer

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32
        """

        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.shape)  # number of dimensions involved in the tensor product layer_pre * weight
        layer = torch.tensordot(layer_pre, self._weight.state, dims=dims_pre) + self._bias.state
        return layer