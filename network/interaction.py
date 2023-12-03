from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F



class Interaction(ABC):
    """Abstract class for interactions

    An interaction is defined by its variables (layers z_j and parameters theta_k) and by its primitive function Phi_i({z_j},{theta_k})

    Methods
    -------
    primitive_fn():
        Returns the interaction's primitive function (Phi_i)
    grad_layers_fns():
        Returns the list of gradient functions wrt the layers involved in the interaction ({dPhi_i/dz_j}_j)
    grad_params_fns():
        Returns the list of gradient functions wrt the parameters involved in the interaction ({dPhi_i/dtheta_k}_k)
    second_fns():
        Returns the list of `second functions' wrt the parameters involved in the interaction (state -> {d^2Phi_i / dtheta_k ds}_k * state)
    """

    def __init__(self, layers, params):
        """Constructor of Interaction

        Args:
            layers (list of Layer): the layers involved in the interaction
            params (list of Parameter): the parameters involved in the interaction
        """

        self._layers = layers
        self._params = params

        for layer, grad_fn in zip(layers, self._grad_layers_fns()): layer.add_interaction(grad_fn)

        for param, grad_fn, second_fn in zip(params, self._grad_params_fns(), self._second_fns()):
            param.add_interaction(grad_fn, second_fn)

    @abstractmethod
    def primitive_fn(self):
        """Returns the interaction's primitive function

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the interaction's primitive value wrt an example in the current mini-batch
        """
        pass

    def _grad_layers_fns(self):
        """Returns the list of gradient functions wrt the layers involved in the interaction

        Default implementation, valid for any layer and any primitive function

        Returns:
            List of functions. Each function computes the gradient of the corresponding layer
        """
        return list(map(lambda layer: (lambda: self._grad_primitive(layer, mean=False)), self._layers))

    def _grad_params_fns(self):
        """Returns the list of gradient functions wrt the parameters involved in the interaction

        Default implementation, valid for any parameter and any primitive function

        Returns:
            List of functions. Each function returns a Tensor of shape param_shape and type float32: the gradient of the primitive function wrt the parameter
        """
        return list(map(lambda param: (lambda: self._grad_primitive(param, mean=True)), self._params))

    def _second_fns(self):
        """Returns the list of functions state -> d^2Phi / dtheta ds * state, wrt the parameters (theta) involved in the interaction

        Default implementation, valid for any parameter, any layer, and any primitive function

        Returns:
            List of functions. Each function takes in a dictionary of Tensors of shape (bacth_size, layer_shape), and returns a Tensor of shape param_shape
        """
        return list(map(lambda param: (lambda direction: self._second_derivative(param, direction)), self._params))

    def _grad_primitive(self, variable, mean=False):
        """Returns the variable's gradient wrt the primitive function

        Implementation valid for any variable (layer or parameter) and any primitive function
        Implemntation valid whether the variable's requires_grad attribute is set to True or False

        Args:
            variable (Variable): the variable whose gradient we want to compute
            mean (bool, optional): whether we compute the gradient of the mean of the primitive, or the sum of the primitive (over the mini-batch of examples). Default: False.

        Returns:
            Tensor: the gradient of the primitive function wrt the variable. Shape is (batch_size, layer_shape) if the varaible is a Layer, or param_shape if it is a Parameter. Type is float32
        """

        if variable.state.requires_grad == False:
            variable.state.requires_grad = True
            primitive = torch.mean( self.primitive_fn() ) if mean else torch.sum( self.primitive_fn() )
            grad = torch.autograd.grad(primitive, variable.state)[0]
            variable.state.requires_grad = False

        else:
            # if the variable already has its requires_grad attribute set to True, we create the graph of derivatives as we compute the gradient of the primitive function
            primitive = torch.mean( self.primitive_fn() ) if mean else torch.sum( self.primitive_fn() )
            grad = torch.autograd.grad(primitive, variable.state, create_graph=True)[0]
            
        return grad

    def _second_derivative(self, param, direction):
        """Returns the param's gradient wrt the directional derivative of the primitive function (in the direction `direction')

        Implementation valid for any parameter and any primitive function
        Implementation valid whether the parameter's requires_grad attribute is set to True or False

        This method is only used in the context of Eqprop. Therefore the requires_grad attributes of the layers and params are set to False when calling this method.
        The requires_grad attributes of direction must also be set to False.

        Args:
            param (Parameter): the parameter whose gradient we want to compute
            direction (dict of Tensor): the direction in the state space in which we take the directional derivative of the primitive function

        Returns:
            Tensor: the gradient wrt the parameter of the directional derivative of the primitive function (in the direction state). Shape is param_shape. Type is float32
        """

        for layer in self._layers: layer.state.requires_grad = True
        for param in self._params: param.state.requires_grad = True
        primitive = torch.mean( self.primitive_fn() )
        layer_grads = torch.autograd.grad(primitive, [layer.state for layer in self._layers], create_graph=True)
        direction = [direction[layer.name] for layer in self._layers]
        param_grad = torch.autograd.grad(layer_grads, param.state, grad_outputs = direction)[0]
        for layer in self._layers: layer.state.requires_grad = False
        for param in self._params: param.state.requires_grad = False

        return param_grad


class NudgingInteraction(Interaction):
    """Interaction of the Nudging Force

    A nudging interaction is defined between a layer and the force acting on it

    Attributes
    ----------
    _layer (Layer): the layer involved in the interaction
    _force: the force acting on the layer
    """

    def __init__(self, layer, force):
        """Creates an instance of NudgingInteraction

        Args:
            layer (Layer): the layer involved in the interaction
            force: the force acting on the layer
        """

        self._layer = layer
        self._force = force

        Interaction.__init__(self, [layer], [])

    def primitive_fn(self):
        """Primitive function of the nudging interaction

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the primitive term of an example in the current mini-batch
        """

        return self._layer.state.mul(self._force.state).flatten(start_dim=1).sum(dim=1)

    def _grad_layers_fns(self):
        """Overrides the default implementation of Interaction"""
        return [self._grad_layer]

    def _grad_layer(self):
        """Returns the interaction's gradient wrt the layer

        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32: the gradient
        """

        return self._force.state


class BiasInteraction(Interaction):
    """Interaction of the Bias

    A bias interaction is defined between a layer and its corresponding bias variable

    Attributes
    ----------
    _layer (Layer): the layer involved in the interaction
    _bias (Bias): the layer's bias
    """

    def __init__(self, layer, bias):
        """Creates an instance of BiasInteraction

        Args:
            layer (Layer): the layer involved in the interaction
            bias (Bias): the layer's bias
        """

        self._layer = layer
        self._bias = bias

        Interaction.__init__(self, [layer], [bias])

    def primitive_fn(self):
        """Primitive function of the bias interaction

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the primitive term of an example in the current mini-batch
        """

        # FIXME: we need to broadcast the bias tensor to the same shape as the layer tensor, to make sure that the correct dimensions are multiplied together.

        bias = self._bias.state.unsqueeze(0)
        while len(bias.shape) < len(self._layer.state.shape): bias = bias.unsqueeze(-1)

        return self._layer.state.mul(bias).flatten(start_dim=1).sum(dim=1)

    def _grad_layers_fns(self):
        """Overrides the default implementation of Interaction"""
        return [self._grad_layer]

    def _grad_params_fns(self):
        """Overrides the default implementation of Interaction"""
        return [self._grad_bias]

    def _grad_layer(self):
        """Returns the interaction's gradient wrt the layer

        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32: the gradient
        """

        # FIXME: we need to broadcast the bias tensor to the same shape as the layer tensor, to make sure that the correct dimensions are added together.
        grad = self._bias.state.unsqueeze(0)
        while len(grad.shape) < len(self._layer.state.shape): grad = grad.unsqueeze(-1)
        return grad

    def _grad_bias(self):
        """Returns the interaction's gradient wrt the bias"""

        # FIXME: we need to broadcast the bias tensor to the same shape as the layer tensor, to make sure that the correct dimensions are added together.

        grad = self._layer.state.mean(dim=0)

        if len(grad.shape) > len(self._bias.shape):
            dims = tuple(range(len(self._bias.shape), len(grad.shape)))
            grad = grad.sum(dim=dims)
            
        return grad


class DenseInteraction(Interaction):
    """Dense ('fully connected') interaction between two layers

    A dense interaction is defined between three variables: two adjacent layers, and the corresponding weight tensor between the two.
    
    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_pre_shape). Type is float32.
    _layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_post_shape). Type is float32.
    _weight (DenseWeight): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.
    """

    def __init__(self, layer_pre, layer_post, dense_weight):
        """Creates an instance of DenseInteraction

        Args:
            layer_pre (Layer): pre-synaptic layer
            layer_post (Layer): post-synaptic layer
            dense_weight (DenseWeight): the dense weights between the pre- and post-synaptic layer
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = dense_weight

        Interaction.__init__(self, [layer_pre, layer_post], [dense_weight])

    def primitive_fn(self):
        """Returns the primitive of a dense interaction.
        
        Example:
            - layer_pre is of shape (16, 1, 28, 28), i.e. batch_size is 16, with 1 channel of 28 by 28 (e.g. input tensor for MNIST)
            - layer_post is of shape (16, 2048), i.e. batch_size is 16, with 2048 units
            - weight is of shape (1, 28, 28, 2048)
        pre * W is the tensor product of pre and W over the dimensions (1, 28, 28). The result is a tensor of shape (16, 2048).
        
        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)  # number of dimensions involved in the tensor product layer_pre * weight
        return torch.tensordot(layer_pre, self._weight.state, dims=dims_pre).mul(layer_post).flatten(start_dim=1).sum(dim=1)  # Hebbian term: layer_pre * weight * layer_post

    def _grad_layers_fns(self):
        """Overrides the default implementation of Interaction"""
        return [self._grad_pre, self._grad_post]

    def _grad_params_fns(self):
        """Overrides the default implementation of Interaction"""
        return [self._grad_weight]

    def _grad_pre(self):
        """Returns the gradient of the primitive function wrt the pre-synaptic layer.

        This is the usual weight * layer_post

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the gradient wrt layer_pre
        """

        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)  # number of dimensions involved in the tensor product
        dim_weight = len(self._weight.shape)
        permutation = tuple(range(dims_pre, dim_weight)) + tuple(range(dims_pre))
        return torch.tensordot(layer_post, self._weight.state.permute(permutation), dims=dims_post)

    def _grad_post(self):
        """Returns the gradient of the primitive function wrt the post-synaptic layer.

        This is the usual layer_pre * weight

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the gradient wrt layer_post
        """

        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.shape)  # number of dimensions involved in the tensor product
        return torch.tensordot(layer_pre, self._weight.state, dims=dims_pre)

    def _grad_weight(self):
        """Returns the gradient of the primitive function wrt the weight.

        This is the usual Hebbian term, dPhi/dtheta = layer_pre^T * layer_post

        Returns:
            Tensor of shape weight_shape and type float32: the gradient wrt the weights
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        batch_size = layer_pre.shape[0]
        return torch.tensordot(layer_pre, layer_post, dims=([0], [0])) / batch_size  # we divide by batch size because we want the mean gradient over the mini-batch


class ConvMaxPoolInteraction(Interaction):
    """Convolutional interaction between two layers, with 2*2 max pooling.

    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _weight (ConvWeight): convolutional weight tensor between layer_pre and layer_post. Type is float32.
    _padding (int): padding of the convolution.
    """

    def __init__(self, layer_pre, layer_post, conv_weight, padding=0):
        """Creates an instance of ConvMaxPoolInteraction

        Args:
            layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            conv_weight (ConvWeight): convolutional weights between layer_pre and layer_post.
            padding (int, optional): padding of the convolution. Default: 0
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = conv_weight
        self._padding = padding

        Interaction.__init__(self, [layer_pre, layer_post], [conv_weight])

    def primitive_fn(self):
        """Returns the primitive of a convolutional interaction with max pooling.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state

        return F.max_pool2d(F.conv2d(layer_pre, self._weight.state, padding=self._padding), 2).mul(layer_post).sum(dim=(3,2,1))

    def _grad_layers_fns(self):
        """Overrides the default implementation of Interaction"""
        return [self._grad_pre, self._grad_post]

    def _grad_params_fns(self):
        """Overrides the default implementation of Interaction"""
        return [self._grad_weight]

    def _grad_pre(self):
        """Returns the gradient of the primitive function wrt the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the gradient wrt layer_pre
        """

        layer_pre = self._layer_pre.state
        _, indices = F.max_pool2d(F.conv2d(layer_pre, self._weight.state, padding=self._padding), 2, return_indices=True)
        layer_post = self._layer_post.state
        layer_post = F.max_unpool2d(layer_post, indices, 2)  # unpooling operation
        return F.conv_transpose2d(layer_post, self._weight.state, padding=self._padding)

    def _grad_post(self):
        """Returns the gradient of the primitive function wrt the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the gradient wrt layer_post
        """

        layer_pre = self._layer_pre.state
        return F.max_pool2d(F.conv2d(layer_pre, self._weight.state, padding=self._padding), 2)

    def _grad_weight(self):
        """Returns the gradient of the primitive function wrt the weight.

        Returns:
            Tensor of shape weight_shape and type float32: the gradient wrt the weights
        """

        layer_pre = self._layer_pre.state
        _, indices = F.max_pool2d(F.conv2d(layer_pre, self._weight.state, padding=self._padding), 2, return_indices=True)

        layer_post = self._layer_post.state
        layer_post = F.max_unpool2d(layer_post, indices, 2)  # unpooling operation

        batch_size = layer_pre.shape[0]

        return F.conv2d(layer_pre.transpose(0, 1), layer_post.transpose(0, 1), padding=self._padding).transpose(0, 1) / batch_size  # we divide by batch size because we want the mean gradient over the mini-batch