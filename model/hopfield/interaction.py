import torch
import torch.nn.functional as F

from model.function.interaction import Function 



class BiasInteraction(Function):
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

        Function.__init__(self, [layer], [bias])

    def eval(self):
        """Energy function of the bias interaction

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        # FIXME: we need to broadcast the bias tensor to the same shape as the layer tensor, to make sure that the correct dimensions are multiplied together.

        bias = self._bias.get().unsqueeze(0)
        while len(bias.shape) < len(self._layer.state.shape): bias = bias.unsqueeze(-1)

        return - self._layer.state.mul(bias).flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        """Overrides the default implementation of Function"""
        dictionary = {self._layer: self._grad_layer}
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._bias: self._grad_bias}
        return dictionary[param]

    def _grad_layer(self):
        """Returns the interaction's gradient wrt the layer

        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32: the gradient
        """

        # FIXME: we need to broadcast the bias tensor to the same shape as the layer tensor, to make sure that the correct dimensions are added together.
        grad = - self._bias.get().unsqueeze(0)
        while len(grad.shape) < len(self._layer.state.shape): grad = grad.unsqueeze(-1)
        return grad

    def _grad_bias(self):
        """Returns the interaction's gradient wrt the bias"""

        # FIXME: we need to broadcast the bias tensor to the same shape as the layer tensor, to make sure that the correct dimensions are added together.

        grad = - self._layer.state.mean(dim=0)

        if len(grad.shape) > len(self._bias.shape):
            dims = tuple(range(len(self._bias.shape), len(grad.shape)))
            grad = grad.sum(dim=dims)
            
        return grad


class DenseHopfield(Function):
    """Dense ('fully connected') interaction between two layers

    A dense interaction is defined between three variables: two adjacent layers, and the corresponding weight tensor between the two.
    
    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_pre_shape). Type is float32.
    _layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_post_shape). Type is float32.
    _weight (DenseWeight): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.
    """

    def __init__(self, layer_pre, layer_post, dense_weight):
        """Creates an instance of DenseHopfield

        Args:
            layer_pre (Layer): pre-synaptic layer
            layer_post (Layer): post-synaptic layer
            dense_weight (DenseWeight): the dense weights between the pre- and post-synaptic layer
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = dense_weight

        Function.__init__(self, [layer_pre, layer_post], [dense_weight])

    def eval(self):
        """Returns the energy of a dense interaction.
        
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
        return - torch.tensordot(layer_pre, self._weight.get(), dims=dims_pre).mul(layer_post).flatten(start_dim=1).sum(dim=1)  # Hebbian term: layer_pre * weight * layer_post

    def grad_layer_fn(self, layer):
        """Overrides the default implementation of Function"""
        dictionary = {
            self._layer_pre: self._grad_pre,
            self._layer_post: self._grad_post
            }
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]

    def _grad_pre(self):
        """Returns the gradient of the energy function wrt the pre-synaptic layer.

        This is the usual - weight * layer_post

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the gradient wrt layer_pre
        """

        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)  # number of dimensions involved in the tensor product
        dim_weight = len(self._weight.shape)
        permutation = tuple(range(dims_pre, dim_weight)) + tuple(range(dims_pre))
        return - torch.tensordot(layer_post, self._weight.get().permute(permutation), dims=dims_post)

    def _grad_post(self):
        """Returns the gradient of the energy function wrt the post-synaptic layer.

        This is the usual - layer_pre * weight

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the gradient wrt layer_post
        """

        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.shape)  # number of dimensions involved in the tensor product
        return - torch.tensordot(layer_pre, self._weight.get(), dims=dims_pre)

    def _grad_weight(self):
        """Returns the gradient of the energy function wrt the weight.

        This is the usual Hebbian term, dE/dtheta = - layer_pre^T * layer_post

        Returns:
            Tensor of shape weight_shape and type float32: the gradient wrt the weights
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        batch_size = layer_pre.shape[0]
        return - torch.tensordot(layer_pre, layer_post, dims=([0], [0])) / batch_size  # we divide by batch size because we want the mean gradient over the mini-batch


class ConvAvgPoolHopfield(Function):
    """Convolutional interaction between two layers, with 2*2 average (or `mean') pooling.

    A convolutional interaction with average pooling is defined between three variables: two adjacent layers, and the corresponding convolutional weight tensor between the two.
    
    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_pre_shape). Type is float32.
    _layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_post_shape). Type is float32.
    _weight (ConvWeight): convolutional weight tensor between layer_pre and layer_post. Type is float32.
    _padding (int): padding of the convolution.
    """

    def __init__(self, layer_pre, layer_post, conv_weight, padding=0):
        """Creates an instance of ConvAvgPoolHopfield
        
        Args:
            layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            conv_weight (ConvWeight): convolutional weights between layer_pre and layer_post. Type is float32.
            padding (int, optional): padding of the convolution. Default: 0
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = conv_weight
        self._padding = padding

        Function.__init__(self, [layer_pre, layer_post], [conv_weight])

    def eval(self):
        """Returns the energy of a convolutional interaction with average pooling.
        
        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state

        return - F.avg_pool2d(F.conv2d(layer_pre, self._weight.get(), padding=self._padding), 2).mul(layer_post).sum(dim=(3,2,1))  # Hebbian term: layer_pre * weight * layer_post

    def grad_layer_fn(self, layer):
        """Overrides the default implementation of Function"""
        dictionary = {
            self._layer_pre: self._grad_pre,
            self._layer_post: self._grad_post
            }
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]

    def _grad_pre(self):
        """Returns the gradient of the energy function wrt the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the gradient wrt layer_pre
        """

        layer_post = self._layer_post.state
        layer_post = F.interpolate(layer_post, scale_factor=2) / 4.  # unpooling operation
        return - F.conv_transpose2d(layer_post, self._weight.get(), padding=self._padding)

    def _grad_post(self):
        """Returns the gradient of the energy function wrt the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the gradient wrt layer_post
        """

        layer_pre = self._layer_pre.state
        return - F.avg_pool2d(F.conv2d(layer_pre, self._weight.get(), padding=self._padding), 2)

    def _grad_weight(self):
        """Returns the gradient of the energy function wrt the weight.

        Returns:
            Tensor of shape weight_shape and type float32: the gradient wrt the weights
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        batch_size = layer_pre.shape[0]

        layer_post = F.interpolate(layer_post, scale_factor=2) / 4.  # unpooling operation

        return - F.conv2d(layer_pre.transpose(0, 1), layer_post.transpose(0, 1), padding=self._padding).transpose(0, 1) / batch_size  # we divide by batch size because we want the mean gradient over the mini-batch


class ConvMaxPoolHopfield(Function):
    """Convolutional interaction between two layers, with 2*2 max pooling.

    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _weight (ConvWeight): convolutional weight tensor between layer_pre and layer_post. Type is float32.
    _padding (int): padding of the convolution.
    """

    def __init__(self, layer_pre, layer_post, conv_weight, padding=0):
        """Creates an instance of ConvMaxPoolHopfield

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

        Function.__init__(self, [layer_pre, layer_post], [conv_weight])

    def eval(self):
        """Returns the energy of a convolutional interaction with max pooling.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state

        return - F.max_pool2d(F.conv2d(layer_pre, self._weight.get(), padding=self._padding), 2).mul(layer_post).sum(dim=(3,2,1))

    def grad_layer_fn(self, layer):
        """Overrides the default implementation of Function"""
        dictionary = {
            self._layer_pre: self._grad_pre,
            self._layer_post: self._grad_post
            }
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]

    def _grad_pre(self):
        """Returns the gradient of the energy function wrt the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the gradient wrt layer_pre
        """

        layer_pre = self._layer_pre.state
        _, indices = F.max_pool2d(F.conv2d(layer_pre, self._weight.get(), padding=self._padding), 2, return_indices=True)
        layer_post = self._layer_post.state
        layer_post = F.max_unpool2d(layer_post, indices, 2)  # unpooling operation
        return - F.conv_transpose2d(layer_post, self._weight.get(), padding=self._padding)

    def _grad_post(self):
        """Returns the gradient of the energy function wrt the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the gradient wrt layer_post
        """

        layer_pre = self._layer_pre.state
        return - F.max_pool2d(F.conv2d(layer_pre, self._weight.get(), padding=self._padding), 2)

    def _grad_weight(self):
        """Returns the gradient of the energy function wrt the weight.

        Returns:
            Tensor of shape weight_shape and type float32: the gradient wrt the weights
        """

        layer_pre = self._layer_pre.state
        _, indices = F.max_pool2d(F.conv2d(layer_pre, self._weight.get(), padding=self._padding), 2, return_indices=True)

        layer_post = self._layer_post.state
        layer_post = F.max_unpool2d(layer_post, indices, 2)  # unpooling operation

        batch_size = layer_pre.shape[0]

        return - F.conv2d(layer_pre.transpose(0, 1), layer_post.transpose(0, 1), padding=self._padding).transpose(0, 1) / batch_size  # we divide by batch size because we want the mean gradient over the mini-batch


class ConvSoftPoolHopfield(Function):
    """Convolutional interaction between two layers, with 2*2 soft pooling.

    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _weight (ConvWeight): convolutional weight tensor between layer_pre and layer_post. Type is float32.
    _padding (int): padding of the convolution.
    """

    def __init__(self, layer_pre, layer_post, conv_weight, padding=0, beta=3.):
        """Creates an instance of ConvSoftPoolHopfield

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
        self._beta = beta  # inverse of the temperature

        Function.__init__(self, [layer_pre, layer_post], [conv_weight])

    def eval(self):
        """Returns the energy of a convolutional interaction with max pooling.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy value of an example in the current mini-batch
        """

        def patch_wise_max(input, kernel_size, stride):
            # Find all pooling windows
            input_windows_tmp = input.unfold(2, kernel_size, stride)
            input_windows = input_windows_tmp.unfold(3, kernel_size, stride)   
            input_windows = input_windows.contiguous().view(*input_windows.size()[:-2], -1)

            # Take window-wise max values
            values, _ = torch.max(input_windows, dim=4, keepdim= True)

            # Reshape tensor back to input size
            values = values.repeat(*(len(values.size()) - 1) * (1,), kernel_size * kernel_size)
            values = values.contiguous().view(values.size(0), values.size(1), -1, values.size(-1))
            values = values.contiguous().permute(0, 1, 3, 2)
            values = values.contiguous().view(values.size(0), -1, values.size(-1))
            out = F.fold(values, output_size=input.size()[2:], kernel_size=kernel_size, stride=stride)
            
            return out

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        x = F.conv2d(layer_pre, self._weight.get(), padding=self._padding)
        '''e_x = torch.clamp(torch.exp(self._beta * x), float(0), float('inf'))
        normalization = F.avg_pool2d(e_x, 2)  # this is 1/4 times the normalization factor
        normalization = F.interpolate(normalization, scale_factor=2)
        p_x = e_x.div(normalization)  # this is 4 times the probabilities
        product = F.avg_pool2d(x.mul(p_x), 2)
        return torch.clamp(product.mul(layer_post).sum(dim=(3,2,1)), float(0), float('inf'))'''

        # Remove patch-wise max values at the numerator and at the denominator
        max_x = patch_wise_max(x, 2, 2).detach()
        e_x = torch.exp(x - max_x)
        num = F.avg_pool2d(x.mul(e_x), 2, 2)
        denom = F.avg_pool2d(e_x, 2, 2)
        return - num.div(denom).mul(layer_post).sum(dim=(3,2,1))


class ModernHopfield(Function):
    """Interaction modelling a modern Hopfield layer

    Attributes
    ----------
    _layer (Layer): the layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _weight (DenseWeight): weight tensor between layer and the modern Hopfield layer. Type is float32.
    """

    def __init__(self, layer, weight):
        """Creates an instance of ModernHopfield

        Args:
            layer (Layer): the layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            weight (DenseWeight): weight tensor between layer and the modern Hopfield layer. Type is float32.
        """

        # TODO: parametrize by a 'temperature' alpha

        self._layer = layer
        self._weight = weight

        Function.__init__(self, [layer], [weight])

    def eval(self):
        """Returns the energy of a modern Hopfield interaction

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy value of an example in the current mini-batch
        """

        layer = self._layer.state
        weight = self._weight.get()
        dims = len(self._layer.shape)  # number of dimensions involved in the tensor product
        return - torch.log(torch.exp(torch.tensordot(layer, weight, dims=dims)).flatten(start_dim=1).sum(dim=1))


'''class MaxPoolConvHopfield(Function):
    """Experimental Class. MaxPool is optional and before the convolution.

    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
    _weight (ConvWeight): convolutional weight tensor between layer_pre and layer_post. Type is float32.
    _padding (int): padding of the convolution.
    _max_pool (bool): whether or not we perform max_pooling before the convolution.
    """

    def __init__(self, layer_pre, layer_post, conv_weight, padding=0, max_pool=True):
        """Creates an instance of MaxPoolConvHopfield

        Args:
            layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            conv_weight (ConvWeight): convolutional weights between layer_pre and layer_post.
            padding (int, optional): padding of the convolution. Default: 0
            max_pool (bool, optional): whether or not we perform max_pooling before the convolution. Default: False
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = conv_weight
        self._padding = padding
        self._max_pool = max_pool

        Function.__init__(self, [layer_pre, layer_post], [conv_weight])

    def eval(self):
        """Returns the energy of a convolutional interaction with max pooling.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state

        if self._max_pool: layer_pre = F.max_pool2d(layer_pre, 2)

        return - F.conv2d(layer_pre, self._weight.get(), padding=self._padding).mul(layer_post).sum(dim=(3,2,1))

    def grad_layer_fn(self, layer):
        """Overrides the default implementation of Function"""
        dictionary = {
            self._layer_pre: self._grad_pre,
            self._layer_post: self._grad_post
            }
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]

    def _grad_pre(self):
        """Returns the gradient of the energy function wrt the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the gradient wrt layer_pre
        """


        layer_post = self._layer_post.state
        result = F.conv_transpose2d(layer_post, self._weight.get(), padding=self._padding)

        if self._max_pool:
            layer_pre = self._layer_pre.state
            _, indices = F.max_pool2d(layer_pre, 2, return_indices=True)
            result = - F.max_unpool2d(result, indices, 2)  # unpooling operation
        return result

    def _grad_post(self):
        """Returns the gradient of the energy function wrt the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the gradient wrt layer_post
        """

        layer_pre = self._layer_pre.state
        if self._max_pool: layer_pre = F.max_pool2d(layer_pre, 2)
        return - F.conv2d(layer_pre, self._weight.get(), padding=self._padding)

    def _grad_weight(self):
        """Returns the gradient of the energy function wrt the weight.

        Returns:
            Tensor of shape weight_shape and type float32: the gradient wrt the weights
        """

        layer_pre = self._layer_pre.state
        if self._max_pool: layer_pre = F.max_pool2d(layer_pre, 2)

        layer_post = self._layer_post.state

        batch_size = layer_pre.shape[0]

        return - F.conv2d(layer_pre.transpose(0, 1), layer_post.transpose(0, 1), padding=self._padding).transpose(0, 1) / batch_size  # we divide by batch size because we want the mean gradient over the mini-batch'''