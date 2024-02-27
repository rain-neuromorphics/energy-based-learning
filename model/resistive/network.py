import torch

from model.function.interaction import SumSeparableFunction
from model.resistive.layer import ResistiveInputLayer, NonlinearResistiveLayer
from model.variable.layer import LinearLayer
from model.variable.parameter import Bias, DenseWeight, ConvWeight
# from model.resistive.parameter import TiedDenseWeight as DenseWeight
from model.function.interaction import BiasInteraction
from model.resistive.interaction import DenseResistive



class DeepResistiveEnergy(SumSeparableFunction):
    """Energy function (power dissipation) of a deep resistive network (DRN)

    The model consists of multiple layers, Successive layers are densely connected.
    """

    def __init__(self, layer_shapes, weight_gains, input_gain):
        """Creates an instance of a dense Hopfield network

        Args:
            layer_shapes (list of tuple of ints): the shapes of the tensors representing the layers of the network
            weight_gains (list of float32): the gains of the weights used at initialization
            input_gain (float): the gain of input variables (input voltage sources)
        """

        self._input_amplifier = input_gain

        self._layer_shapes = layer_shapes
        self._weight_gains = weight_gains

        # build the layers of the network
        input_shape = layer_shapes[0]
        hidden_shapes = layer_shapes[1:-1]
        output_shape = layer_shapes[-1]

        input_layer = ResistiveInputLayer(input_shape, gain=input_gain, device=None)  # input layer
        hidden_layers = [NonlinearResistiveLayer(shape, device=None) for shape in hidden_shapes]  # hidden layers
        output_layer = LinearLayer(output_shape, device=None)  # output layer
        layers = [input_layer] + hidden_layers + [output_layer]

        # build the biases
        biases = [Bias(shape, 0., device=None) for shape in layer_shapes]
        bias_interactions = [BiasInteraction(layer, bias) for layer, bias in zip(layers, biases)]

        # build the weights of the network
        # outs = [True] * (len(edges)-1) + [False]
        weights = [DenseWeight(shape_pre, shape_post, gain, device=None, clamp=True) for shape_pre, shape_post, gain in zip(layer_shapes[:-1], layer_shapes[1:], weight_gains)]
        weight_interactions = [DenseResistive(layer_pre, layer_post, weight) for layer_pre, layer_post, weight in zip(layers[:-1], layers[1:], weights)]
        
        params = biases[1:] + weights
        interactions = bias_interactions[1:] + weight_interactions

        # creates an instance of Network
        SumSeparableFunction.__init__(self, layers, params, interactions)

    def __str__(self):
        return 'Deep Resistive Network -- layer shapes={}, weight gains={}, input_gain={}'.format(self._layer_shapes, self._weight_gains, self._input_amplifier)