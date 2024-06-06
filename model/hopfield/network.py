import numpy

from model.function.interaction import SumSeparableFunction
from model.variable.layer import InputLayer, LinearLayer
from model.hopfield.layer import HardSigmoidLayer, SigmoidLayer, SoftMaxLayer, dSiLULayer
from model.variable.parameter import Bias, DenseWeight, ConvWeight
from model.hopfield.interaction import BiasInteraction, DenseHopfield, ConvAvgPoolHopfield, ConvMaxPoolHopfield, ConvSoftPoolHopfield, ModernHopfield



def create_layer(shape, activation='hard-sigmoid'):
    """Adds a layer to the network

    Args:
        shape (tuple of ints): shape of the layer
        activation (str, optional): the layer's activation function, either the identity ('linear'), the 'hard-sigmoid', or the `silu'. Default: 'hard-sigmoid'.
    """

    if activation == 'linear': layer = LinearLayer(shape)
    elif activation == 'hard-sigmoid': layer = HardSigmoidLayer(shape)
    elif activation == 'sigmoid': layer = SigmoidLayer(shape)
    elif activation == 'softmax': layer = SoftMaxLayer(shape)
    elif activation == 'silu': layer = dSiLULayer(shape)
    elif activation == 'input': layer = InputLayer(shape)
    else: raise ValueError("expected `linear', `hard-sigmoid' or `silu' but got {}".format(activation))

    return layer

def create_edge(layers, interaction_type, indices, gain, shape=None, padding=0):
    """Adds an interaction between two layers of the network.

    Adding an interaction also adds the associated parameter (weight or bias)

    Args:
        interaction_type (str): either `bias', `dense', `conv_avg_pool' or `conv_max_pool'
        indices (list of int): indices of layer_pre (the `pre-synaptic' layer) and layer_post (the `post-synaptic' layer)
        gain (float32): the gain (scaling factor) of the param at initialization
        shape (tuple of ints, optional): the shape of the param tensor. Required in the case of convolutional params. Default: None
        padding (int, optional): the padding of the convolution, if applicable. Default: 0
    """

    if interaction_type == "bias":
        layer = layers[indices[0]]
        if shape == None: shape = layer.shape  # if no shape is provided for the bias, we use the layer's shape by default
        param = Bias(shape, gain=gain, device=None)
        interaction = BiasInteraction(layer, param)
    elif interaction_type == "dense":
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = DenseWeight(layer_pre.shape, layer_post.shape, gain, device=None)
        interaction = DenseHopfield(layer_pre, layer_post, param)
    elif interaction_type == "conv_avg_pool":
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = ConvWeight(shape, gain, device=None)
        interaction = ConvAvgPoolHopfield(layer_pre, layer_post, param, padding)
    elif interaction_type == "conv_max_pool":
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = ConvWeight(shape, gain, device=None)
        interaction = ConvMaxPoolHopfield(layer_pre, layer_post, param, padding)
    elif interaction_type == "conv_soft_pool":
        layer_pre = layers[indices[0]]
        layer_post = layers[indices[1]]
        param = ConvWeight(shape, gain, device=None)
        interaction = ConvSoftPoolHopfield(layer_pre, layer_post, param, padding)
    elif interaction_type == "modern_hopfield":
        layer = layers[indices[0]]
        shape = (1024,)
        param = DenseWeight(layer.shape, shape, gain, device=None)
        interaction = ModernHopfield(layer, param)
    else:
        raise ValueError("expected `bias', `dense', `conv_avg_pool', `conv_max_pool' or `conv_soft_pool' but got {}".format(interaction_type))
    
    return param, interaction



class DeepHopfieldEnergy(SumSeparableFunction):
    """Energy function of a deep Hopfield network (DHN)

    The underlying model consists of multiple layers. Successive layers are densely connected.
    """

    def __init__(self, layer_shapes, weight_gains):
        """Creates an instance of a deep Hopfield network (DHN)

        Args:
            layer_shapes (list of tuple of ints): the shapes of the tensors representing the layers of the network
            weight_gains (list of float32): the gains of the weights used at initialization
        """

        self._layer_shapes = layer_shapes
        self._weight_gains = weight_gains

        # shapes of the layers
        input_shape = layer_shapes[0]
        hidden_shapes = layer_shapes[1:-1]
        output_shape = layer_shapes[-1]

        # build the layers of the network
        input_layer = InputLayer(input_shape)
        hidden_layers = [HardSigmoidLayer(shape) for shape in hidden_shapes]
        output_layer = LinearLayer(output_shape)
        layers = [input_layer] + hidden_layers + [output_layer]

        # build the biases of the network
        biases = [Bias(layer.shape, gain=0., device=None) for layer in layers]  # the bias has the same shape as the layer
        bias_interactions = [BiasInteraction(layer, bias) for layer, bias in zip(layers, biases)]

        # build the weights of the network
        weights = [DenseWeight(shape_pre, shape_post, gain, device=None) for shape_pre, shape_post, gain in zip(layer_shapes[:-1], layer_shapes[1:], weight_gains)]
        weight_interactions = [DenseHopfield(layer_pre, layer_post, weight) for layer_pre, layer_post, weight in zip(layers[:-1], layers[1:], weights)]
            
        params = biases[1:] + weights
        interactions = bias_interactions[1:] + weight_interactions

        # creates an instance of a SumSeparableFunction
        SumSeparableFunction.__init__(self, layers, params, interactions)

    def __str__(self):
        return 'Deep Hopfield Network -- layer shapes={}, weight gains={}'.format(self._layer_shapes, self._weight_gains)



class ConvHopfieldEnergy28(SumSeparableFunction):
    """Energy function of a convolutional Hopfield network (CHN) with 28x28 pixel input images

    The model consists of 3 layers:
        0. input layer has shape (num_inputs, 28, 28)
        1. first hidden layer has shape (num_hidden_1, 12, 12)
        2. second hidden layer has shape (num_hidden_2, 4, 4)
        3. output layer has shape (num_outputs,)
        
    The first two weight tensors are 5x5 convolutional kernels (with padding 0), followed by 2x2 pooling.
    The last weight tensor is dense.
    """

    def __init__(self, num_inputs=1, num_hiddens_1=32, num_hiddens_2=64, num_outputs=10, activation='hard-sigmoid', pool_type='conv_max_pool', weight_gains=[0.6, 0.6, 1.5]):
        """Creates an instance of a convolutional Hopfield network 28x28 (CHN28)

        Args:
            num_inputs (int, optional): number of input filters. Default: 1
            num_hiddens_1 (int, optional): number of filters in the first hidden layer. Default: 32
            num_hiddens_2 (int, optional): number of filters in the second hidden layer. Default: 64
            num_outputs (int): number of output units. Default: 10
            activation (str, optional): activation function used for the hidden layers. Default: 'hard-sigmoid'
        """

        self._size = [num_inputs, num_hiddens_1, num_hiddens_2, num_outputs]
        self._activation = activation
        self._pool_type = pool_type
        self._weight_gains = weight_gains

        # creates the layers of the network
        layer_shapes = [(num_inputs, 28, 28), (num_hiddens_1, 12, 12), (num_hiddens_2, 4, 4), (num_outputs,)]
        activations = ['input', activation, activation, 'linear']
        layers = [create_layer(shape, activation) for shape, activation in zip(layer_shapes, activations)]

        # adds the biases of the network
        num_layers = len(layer_shapes)
        bias_shapes = [(num_hiddens_1,), (num_hiddens_2,), (num_outputs,)]
        bias_gains = [0.5/numpy.sqrt(num_inputs*5*5), 0.5/numpy.sqrt(num_hiddens_1*5*5), 0.5/numpy.sqrt(num_hiddens_2*4*4)]
        biases = [Bias(shape, gain=gain, device=None) for shape, gain in zip(bias_shapes, bias_gains)]
        bias_interactions = [BiasInteraction(layer, bias) for layer, bias in zip(layers[1:], biases)]   

        params = biases
        interactions = bias_interactions
        # adds the weights of the network
        edges = [(0, 1), (1, 2), (2, 3)]
        weight_shapes = [(num_hiddens_1, num_inputs, 5, 5), (num_hiddens_2, num_hiddens_1, 5, 5), (num_hiddens_2, 4, 4, num_outputs)]
        weight_types = [pool_type, pool_type, 'dense']
        paddings = [0, 0, None]
        for indices, weight_type, gain, shape, padding, in zip(edges, weight_types, weight_gains, weight_shapes, paddings):
            param, interaction = create_edge(layers, weight_type, indices, gain, shape, padding)
            params.append(param)
            interactions.append(interaction)
        
        # creates an instance of a SumSeparableFunction
        SumSeparableFunction.__init__(self, layers, params, interactions)

    def __str__(self):
        return 'ConvHopfieldEnergy28 -- size={}, activation={}, pooling={}, gains={}'.format(self._size, self._activation, self._pool_type, self._weight_gains)



class ConvHopfieldEnergy32(SumSeparableFunction):
    """Energy function of a convolutional Hopfield network (CHN) with 32x32 pixel input images

    The model consists of 5 layers:
        0. input layer has shape (num_inputs, 32, 32)
        1. first hidden layer has shape (num_hidden_1, 16, 16)
        2. second hidden layer has shape (num_hidden_2, 8, 8)
        3. third hidden layer has shape (num_hidden_3, 4, 4)
        4. fourth hidden layer has shape (num_hidden_4, 2, 2)
        5. output layer has shape (num_outputs,)
    
    If num_outputs is None or 0, the model has no output layer.
        
    The first four weight tensors are 3x3 convolutional kernels with padding 1, followed by 2x2 pooling.
    The last weight tensor (if it exists) is dense.
    """

    def __init__(self, num_inputs, num_outputs, num_hiddens_1=128, num_hiddens_2=256, num_hiddens_3=512, num_hiddens_4=512, activation='hard-sigmoid', pool_type='conv_max_pool', weight_gains=[0.5, 0.5, 0.5, 0.5, 0.5]):
        """Creates an instance of a convolutional Hopfield network 32x32 (CHN32).

        Args:
            num_inputs (int): number of input filters
            num_outputs (int or None): number of output units
            num_hiddens_1 (int, optional): number of filters in the first hidden layer. Default: 128
            num_hiddens_2 (int, optional): number of filters in the second hidden layer. Default: 256
            num_hiddens_3 (int, optional): number of filters in the third hidden layer. Default: 512
            num_hiddens_4 (int, optional): number of filters in the fourth hidden layer. Default: 512
            activation (str, optional): activation function used for the hidden layers. Default: 'hard-sigmoid'
            pool_type (str, optional): the type of pooling operation used. Default: 'conv_max_pool'
            weight_gains (list of float, optional): the numbers used to scale the weights, layer-wise. Default: gain=0.5 for each weight
        """

        self._size = [num_inputs, num_hiddens_1, num_hiddens_2, num_hiddens_3, num_hiddens_4, num_outputs]
        self._activation = activation
        self._pool_type = pool_type
        self._weight_gains = weight_gains

        # layers of the network
        layer_shapes = [(num_inputs, 32, 32), (num_hiddens_1, 16, 16), (num_hiddens_2, 8, 8), (num_hiddens_3, 4, 4), (num_hiddens_4, 2, 2), (num_outputs,)]
        activations = ['input', activation, activation, activation, activation, 'linear']

        # biases of the network
        bias_shapes = [(num_hiddens_1,), (num_hiddens_2,), (num_hiddens_3,), (num_hiddens_4,), (num_outputs,)]
        bias_gains = [0.5/numpy.sqrt(num_inputs*3*3), 0.5/numpy.sqrt(num_hiddens_1*3*3), 0.5/numpy.sqrt(num_hiddens_2*3*3), 0.5/numpy.sqrt(num_hiddens_3*3*3), 0.5/numpy.sqrt(num_hiddens_4*2*2)]

        # weights of the network
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        weight_shapes = [(num_hiddens_1, num_inputs, 3, 3), (num_hiddens_2, num_hiddens_1, 3, 3), (num_hiddens_3, num_hiddens_2, 3, 3), (num_hiddens_4, num_hiddens_3, 3, 3), (num_hiddens_4, 2, 2, num_outputs)]
        weight_types = [pool_type, pool_type, pool_type, pool_type, 'dense']
        paddings = [1, 1, 1, 1, None]

        if num_outputs == None or num_outputs == 0:
            layer_shapes = layer_shapes[:-1]
            activations = activations[:-1]
            bias_shapes = bias_shapes[:-1]
            bias_gains = bias_gains[:-1]
            edges = edges[:-1]
            weight_types = weight_types[:-1]
            # weight_gains = weight_gains[:-1]  # FIXME: careful not to add this line or RO breaks
            weight_shapes = weight_shapes[:-1]
            paddings = paddings[:-1]

        # create the layers, biases and weights
        layers = [create_layer(shape, activation) for shape, activation in zip(layer_shapes, activations)]

        biases = [Bias(shape, gain=gain, device=None) for shape, gain in zip(bias_shapes, bias_gains)]  # the bias has the same shape as the layer
        bias_interactions = [BiasInteraction(layer, bias) for layer, bias in zip(layers[1:], biases)]

        params = biases
        interactions = bias_interactions

        for indices, weight_type, gain, shape, padding, in zip(edges, weight_types, weight_gains, weight_shapes, paddings):
            param, interaction = create_edge(layers, weight_type, indices, gain, shape, padding)
            params.append(param)
            interactions.append(interaction)

        # creates an instance of a SumSeparableFunction
        SumSeparableFunction.__init__(self, layers, params, interactions)

    def __str__(self):
        return 'ConvHopfieldEnergy32 -- size={}, activation={}, pooling={}, gains={}'.format(self._size, self._activation, self._pool_type, self._weight_gains)