import numpy
import torch

from model.function.interaction import SumSeparableFunction
from model.variable.layer import InputLayer, LinearLayer
from model.forward.layer import ReLULayer
from model.variable.parameter import Bias, DenseWeight
from model.forward.interaction import DenseForward


class NeuralNetwork(SumSeparableFunction):
    """Neural network"""
    
    def forward_fn(self, layer):
        """ """
        return self._dic[layer]


class ReLUNeuralNet(NeuralNetwork):
    """
    ReLU Neural network (ReLU NN)

    The model consists of multiple layers, Successive layers are densely connected.
    """

    def __init__(self, layer_shapes, weight_gains):
        """Creates an instance of a ReLU Neural Network

        Args:
            layer_shapes (list of tuple of ints): the shapes of the tensors representing the layers of the network
            weight_gains (list of float32): the gains of the weights used at initialization
        """

        self._layer_shapes = layer_shapes
        self._weight_gains = weight_gains

        # creates the layers of the network
        input_shape = layer_shapes[0]
        hidden_shapes = layer_shapes[1:-1]
        output_shape = layer_shapes[-1]
        input_layer = InputLayer(input_shape)  # input layer
        hidden_layers = [ReLULayer(shape) for shape in hidden_shapes]  # hidden layers
        output_layer = LinearLayer(output_shape)  # output layer
        layers = [input_layer] + hidden_layers + [output_layer]

        # adds the weights and biases of the network
        params, interactions = [], []
        for layer_pre, layer_post, gain in zip(layers[:-1], layers[1:], weight_gains):
            weight = DenseWeight(layer_pre.shape, layer_post.shape, gain, device=None)
            bias = Bias(layer_post.shape, gain=0., device=None)  # the bias has the same shape as the layer
            interaction = DenseForward(layer_pre, layer_post, weight, bias)
            params.append(weight)
            params.append(bias)
            interactions.append(interaction)

        # creates an instance of NeuralNetwork
        NeuralNetwork.__init__(self, layers, params, interactions)
        self._dic = dict()
        for layer, interaction in zip(layers[1:], interactions): self._dic[layer] = interaction.forward

    def __str__(self):
        return 'ReLU Neural Network -- layer shapes={}, weight gains={}'.format(self._layer_shapes, self._weight_gains)