import numpy

from network.network import Network
from network.variable.layer import LinearLayer, HardSigmoidLayer
from network.variable.parameter import Bias, DenseWeight, ConvWeight
from network.interaction import BiasInteraction, DenseInteraction, ConvMaxPoolInteraction



class HopfieldNetwork(Network):
    """Hopfield network"""
    
    def create_layer(self, shape, activation='hard-sigmoid'):
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
        else: raise ValueError("expected `linear', `hard-sigmoid' or `silu' but got {}".format(activation))

        Network.add_layer(self, layer)

    def create_edge(self, interaction_type, indices, gain, shape=None, padding=0):
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
            layer = self._layers[indices[0]]
            if shape == None: shape = layer.shape  # if no shape is provided for the bias, we use the layer's shape by default
            param = Bias(shape, gain=gain, device=self._device)
            interaction = BiasInteraction(layer, param)
        elif interaction_type == "dense":
            layer_pre = self._layers[indices[0]]
            layer_post = self._layers[indices[1]]
            param = DenseWeight(layer_pre.shape, layer_post.shape, gain, device=self._device)
            interaction = DenseInteraction(layer_pre, layer_post, param)
        elif interaction_type == "conv_avg_pool":
            layer_pre = self._layers[indices[0]]
            layer_post = self._layers[indices[1]]
            param = ConvWeight(shape, gain, device=self._device)
            interaction = ConvAvgPoolInteraction(layer_pre, layer_post, param, padding)
        elif interaction_type == "conv_max_pool":
            layer_pre = self._layers[indices[0]]
            layer_post = self._layers[indices[1]]
            param = ConvWeight(shape, gain, device=self._device)
            interaction = ConvMaxPoolInteraction(layer_pre, layer_post, param, padding)
        elif interaction_type == "conv_soft_pool":
            layer_pre = self._layers[indices[0]]
            layer_post = self._layers[indices[1]]
            param = ConvWeight(shape, gain, device=self._device)
            interaction = ConvSoftPoolInteraction(layer_pre, layer_post, param, padding)
        elif interaction_type == "modern_hopfield":
            layer = self._layers[indices[0]]
            shape = (1024,)
            param = DenseWeight(layer.shape, shape, gain, device=self._device)
            interaction = ModernHopfieldInteraction(layer, param)
        else:
            raise ValueError("expected `bias', `dense', `conv_avg_pool', `conv_max_pool' or `conv_soft_pool' but got {}".format(interaction_type))

        Network.add_param(self, param)
        Network.add_interaction(self, interaction)



class ConvHopfieldNet32(HopfieldNetwork):
    """
    Deep convolutional Hopfield network (DCHN)

    The model consists of 5 layers:
        0. input layer has shape (num_inputs, 32, 32)
        1. first hidden layer has shape (num_hidden_1, 16, 16)
        2. second hidden layer has shape (num_hidden_2, 8, 8)
        3. third hidden layer has shape (num_hidden_3, 4, 4)
        4. fourth hidden layer has shape (num_hidden_4, 2, 2)
        5. output layer has shape (num_outputs,)
        
    The first four layers are 3x3 convolutional layers with padding 1, followed by 2x2 pooling.
    The last (output) layer is dense.
    """

    def __init__(self, num_inputs, num_outputs, num_hiddens_1=128, num_hiddens_2=256, num_hiddens_3=512, num_hiddens_4=512, activation='hard-sigmoid', pool_type='conv_max_pool', weight_gains=[0.5, 0.5, 0.5, 0.5, 0.5]):
        """Creates an instance of a convolutional Hopfield network 32x32.

        Args:
            num_inputs (int): number of input filters
            num_outputs (int): number of output units
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

        # creates an instance of Network
        Network.__init__(self)

        # creates the layers of the network
        layer_shapes = [(num_inputs, 32, 32), (num_hiddens_1, 16, 16), (num_hiddens_2, 8, 8), (num_hiddens_3, 4, 4), (num_hiddens_4, 2, 2), (num_outputs,)]
        activations = ['linear', activation, activation, activation, activation, 'linear']
        for shape, activation in zip(layer_shapes, activations):
            self.create_layer(shape, activation)

        # adds the biases of the network
        indices = [1, 2, 3, 4, 5]
        bias_shapes = [(num_hiddens_1,), (num_hiddens_2,), (num_hiddens_3,), (num_hiddens_4,), (num_outputs,)]
        bias_gains = [0.5/numpy.sqrt(num_inputs*3*3), 0.5/numpy.sqrt(num_hiddens_1*3*3), 0.5/numpy.sqrt(num_hiddens_2*3*3), 0.5/numpy.sqrt(num_hiddens_3*3*3), 0.5/numpy.sqrt(num_hiddens_4*2*2)]
        for index, shape, gain in zip(indices, bias_shapes, bias_gains):
            self.create_edge('bias', (index,), gain, shape)

        # adds the weights of the network
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        weight_shapes = [(num_hiddens_1, num_inputs, 3, 3), (num_hiddens_2, num_hiddens_1, 3, 3), (num_hiddens_3, num_hiddens_2, 3, 3), (num_hiddens_4, num_hiddens_3, 3, 3), (num_hiddens_4, 2, 2, num_outputs)]
        weight_types = [pool_type, pool_type, pool_type, pool_type, 'dense']
        paddings = [1, 1, 1, 1, None]
        for indices, weight_type, gain, shape, padding, in zip(edges, weight_types, weight_gains, weight_shapes, paddings):
            self.create_edge(weight_type, indices, gain, shape, padding)

        # tells the network which layer is the output layer
        self.pack()

    def __str__(self):
        return 'ConvHopfieldNet32 -- size={}, activation={}, pooling={}, gains={}'.format(self._size, self._activation, self._pool_type, self._weight_gains)