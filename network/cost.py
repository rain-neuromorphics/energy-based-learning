from abc import ABC, abstractmethod
import numpy
import torch
import torch.nn.functional as F

from network.variable.parameter import Bias, DenseWeight



class CostFunction(ABC):
    """Abstract class for cost functions

    Attributes
    ----------
    _network (Network): the network whose output plays the role of prediction
    _num_classes (int): the number of categories in the classification task
    _label (Tensor): label tensor. Shape is (batch size,). Type is int.
    _target (Tensor): one-hot-label tensor. Shape is (batch size, output_shape). Type is float.
    _params (list of Parameter): the list of parameter variables involved in the cost function

    Methods
    -------
    set_target(label)
        Set target values
    cost_fn():
        Returns the cost value of the current configuration
    compute_gradient():
        Computes the parameter gradient for the current mini-batch of examples and stores the value in the parameter's .grad attribute
    output_gradients():
        Returns the (negative) error gradients wrt the cost function (evaluated at the current state)
    error_fn()
        Computes the error value for the current state configuration
    top_five_error_fn()
        Computes the top-5 error rate for the current output configuration.
    """

    def __init__(self, network):
        """Initializes an instance of CostFunction

        Args:
            network (Network): the network whose output plays the role of prediction
        """

        self._network = network
        self._num_classes = self._network.read_output().shape[1]  # number of categories in the classification task
        self._label = None  # FIXME
        self._target = None  # FIXME

        self._params = []

    def set_target(self, label):
        """Set label and target values

        Args:
            label: Tensor of shape (batch size,) and type int. Labels associated to the inputs in the batch.
        """

        # TODO: check that the batch_size of the labels is the same as that of the output layer

        output = self._get_output()  # the output layer's state of the network
        device = output.device  # device on which the output layer Tensor is, and on which we put the layer and target Tensors
        self._label = label.to(device)
        self._target = F.one_hot(self._label, num_classes=self._num_classes).type(torch.float32)  # convert the label into its one-hot code

    @abstractmethod
    def cost_fn(self):
        """Returns the value of the cost function evaluated at the current state

        Returns:
            Vector of size (batch_size,) and of type float32. Each coordinate is the cost value of an example in the current mini-batch
        """
        pass
    
    def params(self):
        """Returns the list of parameter variables of the cost function"""

        return self._params

    def error_fn(self):
        """Returns the error value for the current output configuration.

        Returns:
            Tensor of shape (batch_size,) and type bool. Vector of error values for each of the examples in the current mini-batch
        """

        output = self._get_output()  # state of output layer
        prediction = torch.argmax(output, dim=1)  # the predicted category is the index of the output unit that has the highest value
        label = self._label
        return torch.ne(prediction, label)

    def top_five_error_fn(self):
        """Returns the top-5 accuracy for the current output configuration.

        Returns:
            Tensor of shape (batch_size,) and type bool. Vector of top-5 error values for each of the examples in the current mini-batch
        """

        output = self._get_output()  # state of output layer
        _, indices = torch.topk(output, 5, dim=1)  # top-5 categories indices of the output unit
        label = self._label.unsqueeze(-1)
        return torch.all(torch.ne(indices, label), dim=1)

    def compute_gradient(self):
        """Computes the gradient for the current mini-batch of examples and stores the value in the parameter's .grad attribute

        We assume that when this function is called, the network is at equilibrium for nudging=0.
        """

        # Compute the parameter gradients
        param_grads = self._compute_grad()

        # Set the gradients of the parameters
        for param, grad in zip(self.params(), param_grads): param.state.grad = grad

    def output_gradients(self):
        """Returns the (negative) error gradients wrt the cost function (evaluated at the current state)

        Default implementation, valid for any cost function

        Returns:
            Vector of size (batch_size, output_size) and of type float32. Each row is the output gradient of an example in the current mini-batch
        """

        # FIXME: probably wouldn't work with TBP and RBP

        output = self._network.read_output()
        output.requires_grad = True
        cost_sum = torch.sum( self.cost_fn() )
        grad = torch.autograd.grad(cost_sum, output)[0]
        output.requires_grad = False
        return - grad
    
    def _compute_grad(self):
        """Computes the parameter gradients of the cost function for the current mini-batch of examples

        Default implementation that uses autograd
        
        Returns:
            list of Tensor of shape param_shape: the parameter gradients
        """

        if len(self.params()) == 0: return []

        # FIXME: probably wouldn't work with TBP and RBP

        for param in self.params(): param.state.requires_grad = True
        cost_mean = torch.mean( self.cost_fn() )
        params = [param.state for param in self.params()]
        param_grads = torch.autograd.grad(cost_mean, params)
        for param in self.params(): param.state.requires_grad = False

        return param_grads
    
    @abstractmethod
    def _get_output(self):
        """Returns the output of the network"""
        pass



class SquaredError(CostFunction):
    """Class for the squared error cost function between the output layer and the target layer.

    Methods
    -------
    cost_fn()
        Returns the squared error between the output layer and the target
    output_gradients()
        Returns the gradient of the cost function wrt the outputs
    """

    def cost_fn(self):
        """Returns the cost value (the squared error) of the current state configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of cost values for each of the examples in the current mini-batch
        """

        output = self._get_output()  # state of output layer: shape is (batch_size, num_classes)
        target = self._target  # desired output: shape is (batch_size, num_classes)
        return 0.5 * ((output - target) ** 2).sum(dim=1)  # Vector of shape (batch_size,)
    
    def output_gradients(self):
        """Returns the (negative) gradient of the cost function wrt the outputs

        Overrides the default implementation
        """
        return self._target - self._get_output()
    
    def _get_output(self):
        """Returns the output of the network"""
        return self._network.read_output()

    def __str__(self):
        return 'MSE (Mean Squared Error)'