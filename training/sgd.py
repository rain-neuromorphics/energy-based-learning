from abc import ABC, abstractmethod
from itertools import accumulate
import torch



class GradientEstimator(ABC):
    """
    Abstract class for computing or estimating the parameter gradients of a cost function in a network

    This class is used in particular:
    - to perform SGD (stochastic gradient descent),
    - to check the GDD (gradient-descending dynamics) property.

    Methods
    -------
    compute_gradient()
        Compute the parameter gradients for the current mini-batch and assign them to the .grad attribute of the parameters
    detailed_gradients()
        Compute and return the sequence of time-dependent layer- and parameter- gradients on the current mini-batch
    """

    @abstractmethod
    def compute_gradient(self):
        """Compute the parameter gradients for the current mini-batch and assign them to the .grad attribute of the parameters"""
        pass

    @abstractmethod
    def detailed_gradients(self):
        """Compute the sequence of time-dependent layer- and parameter- gradients on the current mini-batch"""
        pass



class Eqprop(GradientEstimator):
    """
    Class used to estimate the parameter gradients of a cost function in a network via Equilibrium Propagation (Eqprop)

    Attributes
    ----------
    _network (Network): the network to train
    _cost_function (CostFunction): the cost function to optimize
    _max_iterations (int): the maximum number of iterations allowed to converge to equilibrium in the second phase of Eqprop
    training_mode (str): either Positively-perturbed Eqprop, Negatively-perturbed Eqprop or Centered Eqprop
    nudging (float): the nudging value used to train via Eqprop
    use_constant_force (bool): whether nudging is achieved using a constant force F = beta * dC(s_free)/ds, or a varying force F = beta * dC(s)/ds
    use_alternative_formula (bool): which equilibrium propagation formula is used to estimating the parameter gradients, either the 'standard' formula (False) or the 'alternative' formula (True)

    Methods
    -------
    compute_gradient()
        Compute the parameter gradients for the current mini-batch via Eqprop, and assign them to the .grad attribute of the parameters
    detailed_gradients()
        Compute and return the sequence of time-dependent layer- and parameter- Eqprop gradients on the current mini-batch
    """

    def __init__(self, network, cost_function, max_iterations=100, training_mode='centered', nudging=0.5, use_constant_force=False, use_alternative_formula=False):
        """Creates an instance of Eqprop

        Args:
            network (Network): the network to optimize via Eqprop
            cost_function (CostFunction): the cost function to optimize
            max_iterations (int, optional): the maximum number of iterations allowed to converge to equilibrium in the second phase of Eqprop. Default: 100
            training_mode (str, optional): either Positively-perturbed Eqprop, Negatively-perturbed Eqprop or Centered Eqprop. Default: 'centered'
            nudging (float, optional): the nudging value used to train via Eqprop. Default: 0.5
            use_constant_force (bool, optional): whether nudging is achieved using a constant force F = beta * dC(s_free)/ds, or a varying force F = beta * dC(s)/ds. Default: False
            use_alternative_formula (bool, optional): which equilibrium propagation formula is used to estimating the parameter gradients, either the 'standard' formula (False) or the 'alternative' formula (True). Default: False
        """

        self._network = network
        self._cost_function = cost_function
        self.max_iterations = max_iterations

        self._nudging = nudging
        self._training_mode = training_mode
        self._set_nudgings()

        self._use_constant_force = use_constant_force
        self._use_alternative_formula = use_alternative_formula

    @property
    def nudging(self):
        """Get and sets the nudging value used for training"""
        return self._nudging

    @nudging.setter
    def nudging(self, nudging):
        if nudging > 0.:
            self._nudging = nudging
            self._set_nudgings()
        else: raise ValueError("expected a positive nudging value, but got {}".format(nudging))

    @property
    def training_mode(self):
        """Get and sets the training mode"""
        return self._training_mode

    @training_mode.setter
    def training_mode(self, training_mode):
        if training_mode in ['positive', 'negative', 'centered']:
            self._training_mode = training_mode
            self._set_nudgings()
        else: raise ValueError("expected 'positive', 'negative' or 'centered' but got {}".format(training_mode))

    @property
    def use_constant_force(self):
        """Get and sets the use_constant_force attribute"""
        return self._use_constant_force

    @use_constant_force.setter
    def use_constant_force(self, use_constant_force):
        if use_constant_force in [True, False]: self._use_constant_force = use_constant_force
        else: raise ValueError("expected True or False, but got {}".format(use_constant_force))

    @property
    def use_alternative_formula(self):
        """Get and sets the use_alternative_formula attribute"""
        return self._use_alternative_formula

    @use_alternative_formula.setter
    def use_alternative_formula(self, use_alternative_formula):
        if use_alternative_formula in [True, False]: self._use_alternative_formula = use_alternative_formula
        else: raise ValueError("expected True or False, but got {}".format(use_alternative_formula))

    def compute_gradient(self):
        """Estimates the gradient for the current mini-batch of examples and stores the value in the parameter's .grad attribute

        Gradient depends on the attributes training_mode (Positively-perturbed Eqprop, Negatively-perturbed Eqprop, Centered Eqprop) and nudging
        We assume that when this function is called, the network is at equilibrium for nudging=0.
        """

        layers_free = [layer.state for layer in self._network.layers()]  # hack: we store the `free state' (i.e. the equilibrium state of the layers with nudging=0)

        # First phase: compute the first equilibrium state
        layers_first = self._compute_equilibrium(self._first_nudging)
        
        # Second phase: compute the second equilibrium state
        for layer, state in zip(self._network.layers(), layers_free): layer.state = state  # hack: we start the second phase from the `free state' again
        layers_second = self._compute_equilibrium(self._second_nudging)

        # Compute the parameter gradients with either the standard Eqprop formula, or the alternative Eqprop formula
        if self._use_alternative_formula:
            param_grads = self._alternative_param_grads(layers_free, layers_first, layers_second)
        else:
            param_grads = self._standard_param_grads(layers_first, layers_second)

        # Set the gradients of the parameters
        for param, grad in zip(self._network.params(), param_grads): param.state.grad = grad
    
    def detailed_gradients(self, cumulative=True):
        """Compute and return the sequence of time-dependent layer- and parameter- Eqprop gradients on the current mini-batch

        Calling this method leaves the network's state unchanged
        For the method to return the correct detailed gradients of Eqprop, the network must be at equilibrium when calling the method

        Args:
            cumulative (bool, optional): if True, computes the cumulative gradients ; if False, computes the gradients increases. Default: False.

        Returns:
            grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
        """

        layers_free = [layer.state for layer in self._network.layers()]  # we store the `free state' (i.e. the equilibrium state of the layers with nudging=0)

        # First phase: compute the pre-activations and activations along the first trajectory
        pre_activations_first, activations_first = self._compute_trajectory(self._first_nudging)
        
        # Second phase: compute the pre-activations and activations along the second trajectory
        for layer, state in zip(self._network.layers(), layers_free): layer.state = state  # we start over from the `free state' again
        pre_activations_second, activations_second = self._compute_trajectory(self._second_nudging)

        # Compute the layer gradients
        layer_grads = [self._layer_grads(first, second) for first, second in zip(pre_activations_first, pre_activations_second)]
        layer_grads = list(map(list, zip(*layer_grads)))  # transpose the list of lists: transform the time-wise layer-wise gradients into layer-wise time-wise gradients

        # Compute the parameter gradients with either the standard Eqprop formula, or the alternative Eqprop formula
        if self._use_alternative_formula:
            param_grads = [self._alternative_param_grads(layers_free, first, second) for first, second in zip(activations_first, activations_second)]
        else:
            param_grads = [self._standard_param_grads(first, second) for first, second in zip(activations_first, activations_second)]
        param_grads = list(map(list, zip(*param_grads)))  # transpose the list of lists: transform the time-wise parameter-wise gradients into parameter-wise time-wise gradients

        # Store the layer-wise and parameter-wise time-wise gradients in a dictionary
        grads = dict()
        for layer, gradients in zip(self._network.layers(free_only=False), layer_grads): grads[layer.name] = gradients
        for param, gradients in zip(self._network.params(), param_grads): grads[param.name] = gradients
        
        # Transform the time-wise gradients into time-wise increases if required
        if not cumulative:
            for layer in self._network.layers(free_only=False):
                layer_grads = grads[layer.name]
                grads[layer.name] = [layer_grads[0]] + [j-i for i, j in zip(layer_grads[:-1], layer_grads[1:])]
            for param in self._network.params():
                param_grads = grads[param.name]
                grads[param.name] = [param_grads[0]] + [j-i for i, j in zip(param_grads[:-1], param_grads[1:])]

        for layer, state in zip(self._network.layers(), layers_free): layer.state = state  # we reset the network to the `free state', where it was initially
        
        return grads

    def _compute_equilibrium(self, nudging):
        """Compute the equilibrium state corresponding to a given nudging value

        Args:
            nudging (float): nudging value used to converge to equilibrium

        Returns:
            layers: dictionary of Tensors. The state of the layers at equilibrium
        """

        for iteration in range(self.max_iterations):
            # set the force at the output layer proportionally to the output gradients
            if iteration == 0 or (not self._use_constant_force):
                output_gradients = self._cost_function.output_gradients()
                self._network.set_output_force(nudging * output_gradients)
            # perform one step of the network's dynamics
            self._network._step_asynchronous(backward=True)

        layers = {layer.name: layer.state for layer in self._network.layers(free_only=False)}

        return layers

    def _compute_trajectory(self, nudging):
        """Compute the trajectory (pre-activations and activations) corresponding to a given nudging value

        Args:
            nudging (float): nudging value used to compute the trajectory

        Returns:
            pre_activations: list of dictionary of Tensors. The layer pre-activations at each step of the trajectory
            activations: list of dictionary of Tensors. The layer activations at each step of the trajectory
        """

        pre_activations = []
        activations = []

        for iteration in range(self.max_iterations):
            # set the force at the output layer proportionally to the output gradients
            if iteration == 0 or (not self._use_constant_force):
                output_gradients = self._cost_function.output_gradients()
                self._network.set_output_force(nudging * output_gradients)
            # perform one step of the network's dynamics and store the layers' pre-activations and activations
            pre_activations.append(
                {layer.name: layer.pre_activate() for layer in self._network.layers(free_only=False)}
                )
            self._network.step_synchronous()
            activations.append(
                {layer.name: layer.state for layer in self._network.layers(free_only=False)}
                )
        
        return pre_activations, activations

    def _standard_param_grads(self, layers_first, layers_second):
        """Compute the parameter gradients using the standard Eqprop formula

        Args:
            layers_first (dictionary of Tensors): the activations of the layers at the first state
            layers_second (dictionary of Tensors): the activations of the layers at the second state

        Returns:
            param_grads: list of Tensors. The parameter gradients
        """

        # Compute the energy gradients of the first state
        for layer in self._network.layers(): layer.state = layers_first[layer.name]
        grads_first = [param.energy_grad() for param in self._network.params()]

        # Compute the energy gradients of the second state
        for layer in self._network.layers(): layer.state = layers_second[layer.name]
        grads_second = [param.energy_grad() for param in self._network.params()]

        # Compute the parameter gradients
        param_grads = [(second - first) / (self._second_nudging - self._first_nudging) for first, second in zip(grads_first, grads_second)]

        return param_grads

    def _alternative_param_grads(self, layers_free, layers_first, layers_second):
        """Compute the parameter gradients using the alternative Eqprop formula

        Args:
            layers_free (list of Tensors): the activations of the layers at the free state
            layers_first (dictionary of Tensors): the activations of the layers at the first state
            layers_second (dictionary of Tensors): the activations of the layers at the second state

        Returns:
            param_grads: list of Tensors. The parameter gradients
        """

        direction = {layer.name: (layers_first[layer.name] - layers_second[layer.name]) / (self._second_nudging - self._first_nudging) for layer in self._network.layers(free_only=False)}
        for layer, state in zip(self._network.layers(), layers_free): layer.state = state  # hack: we need to reset the state of the network at the `free state'
        param_grads = [param.second_fn(direction) for param in self._network.params()]

        return param_grads

    def _layer_grads(self, layers_first, layers_second):
        """Compute the layer gradients given the pre-activations of the first and second states

        Args:
            layers_first (dictionary of Tensors): the pre-activations of the layers at the first state
            layers_second (dictionary of Tensors): the pre-activations of the layers at the second state

        Returns:
            layer_grads: list of Tensors. The layer gradients
        """

        # Compute the layer gradients
        batch_size = self._network._input_layer.state.size(0)
        layer_grads = [(layers_first[layer.name] - layers_second[layer.name]) / ((self._second_nudging - self._first_nudging) * batch_size) for layer in self._network.layers(free_only=False)]

        return layer_grads

    def _set_nudgings(self):
        """Sets the nudging values of the first and second states, depending on the attributes training_mode and nudging

        first_nudging: nudging value used to compute the first state of equilibrium propagation
        second_nudging: nudging value used to compute the second state of equilibrium propagation
        """

        if self._training_mode == "positive":
            self._first_nudging = 0.
            self._second_nudging = self._nudging
        elif self._training_mode == "negative":
            self._first_nudging = -self._nudging
            self._second_nudging = 0.
        elif self._training_mode == "centered":
            self._first_nudging = -self._nudging
            self._second_nudging = self._nudging
        else:
            # TODO: this raise of error is redundant
            raise ValueError("expected 'positive', 'negative' or 'centered' but got {}".format(self._training_mode))

    def __str__(self):
        force = 'constant' if self._use_constant_force else 'adaptive'
        formula = 'alternative' if self._use_alternative_formula else 'standard'
        return 'Eqprop -- mode={}, nudging={}, force={}, formula={}, num_iterations={}'.format(self._training_mode, self._nudging, force, formula, self.max_iterations)


class AutoDiff(GradientEstimator):
    """
    Class used to compute the parameter gradients of a cost function in a network via automatic differentiation (AutoDiff)

    Attributes
    ----------
    network (Network): the network to train
    cost_function (CostFunction): the cost function to optimize
    max_iterations (int): the number of iterations performed to compute the gradients

    Methods
    -------
    compute_gradient()
        Computes the parameter gradients for the current mini-batch via automatic differentiation, and assign them to the .grad attribute of the parameters
    detailed_gradients()
        Compute and return the sequence of time-dependent layer- and parameter- gradients via AutoDiff on the current mini-batch
    """

    def __init__(self, network, cost_function, max_iterations=100):
        """Creates an instance of AutoDiff

        Args:
            network (Network): the network to train
            cost_function (CostFunction): the cost function to optimize
            max_iterations (int, optional): the number of iterations performed to compute the gradients. Default: 100
        """

        self._network = network
        self._cost_function = cost_function
        self.max_iterations = max_iterations

    def compute_gradient(self):
        """Computes the parameter gradients for the current mini-batch via automatic differentiation, and assign them to the .grad attribute of the parameters"""

        for param in self._network.params(): param.state.requires_grad = True

        for _ in range(self.max_iterations): self._network._step_asynchronous(backward=True)
        
        cost_mean = self._cost_function.cost_fn().mean()
        param_grads = torch.autograd.grad(cost_mean, [param.state for param in self._network.params()])

        for param in self._network.params(): param.state.requires_grad = False
        for layer in self._network.layers(): layer.state = layer.state.detach()

        for param, grad in zip(self._network.params(), param_grads): param.state.grad = grad

    def detailed_gradients(self, cumulative=True):
        """Compute and return the sequence of time-dependent layer- and parameter- gradients via AutoDiff on the current mini-batch.
        
        For the GDD property to be satisfied, the network must be at equilibrium when calling the method.
        The Pros of this implementation is that it is linear wrt self.max_iterations (whereas the other implementation is quadratic)
        The Cons of this implementation is that it uses the backward method, and thus violates the rule that the .grad attribute should only be used to assign gradients (not to retrieve gradients)

        # TODO: can we merge ideas from this implementation with the other implementation, so that we get a linear algorithm that does not use backward()?

        Args:
            cumulative (bool, optional): if True, computes the cumulative gradients ; if False, computes the gradients increases. Default: False.

        Returns:
            grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
        """

        # We store the initial state of the layers and parameters, so we can reset them at the end of the method
        layers_state = [layer.state for layer in self._network.layers()]
        params_state = [param.state for param in self._network.params()]

        for param in self._network.params(): param.state.requires_grad = True
        for layer in self._network.layers(): layer.state.requires_grad = True
        
        # During the forward pass, we keep all the tensors of the computational graph in a 'trajectories' dictionary
        trajectories = dict()  
        for layer in self._network.layers(): trajectories[layer.name] = []
        for param in self._network.params(): trajectories[param.name] = []
        
        for _ in range(self.max_iterations):
            for param in self._network.params():
                # at each time step of the forward pass, we create a new parameter tensor,
                # so we can compute the partial derivative of the loss wrt the parameter at this time step in the backward pass
                param.state = param.state + torch.zeros_like(param.state)
                param.state.retain_grad()
                trajectories[param.name].append(param.state)
            self._network.step_synchronous()
            for layer in self._network.layers():
                layer.state.retain_grad()
                trajectories[layer.name].append(layer.state)

        loss = self._cost_function.cost_fn().mean()
        loss.backward()
        
        # After the backward pass, we read all the gradients in the .grad attributes of the variables (layers and parameters)
        # and we store these gradients in the 'grads' dictionary
        grads = dict()
        for layer in self._network.layers():
            layer_grads = [state.grad if state.grad is not None else torch.zeros_like(state) for state in trajectories[layer.name]]
            layer_grads = list(reversed(layer_grads))
            if cumulative: layer_grads = list(accumulate(layer_grads))
            grads[layer.name] = layer_grads
        for param in self._network.params():
            param_grads = [state.grad if state.grad is not None else torch.zeros_like(state) for state in trajectories[param.name]]
            param_grads = list(reversed(param_grads))
            if not cumulative: param_grads = [param_grads[0]] + [j-i for i, j in zip(param_grads[:-1], param_grads[1:])]
            grads[param.name] = param_grads

        # Finally, we reset the state of the layers and parameters where they were initially
        for layer, state in zip(self._network.layers(), layers_state): layer.state = state
        for param, state in zip(self._network.params(), params_state): param.state = state
        
        for layer in self._network.layers(): layer.state.requires_grad = False
        for param in self._network.params(): param.state.requires_grad = False

        return grads

    def __str__(self):
        return 'AutoDiff -- num_iterations={}'.format(self.max_iterations)


class ImplicitDiff(GradientEstimator):
    """
    Class used to compute the parameter gradients of a cost function in a network via implicit differentiation (ImplicitDiff).
    
    The algorithm implemented here is also known as `recurrent backpropagation', or the Almeida-Pineda algorithm.
    
    Attributes
    ----------
    _network (Network): the network to train
    _cost_function (CostFunction): the cost function to optimize
    max_iterations (int): the number of iterations performed to compute the gradients

    Methods
    -------
    compute_gradient()
        Computes the parameter gradients for the current mini-batch via implicit differentiation, and assign them to the .grad attribute of the parameters
    detailed_gradients()
        Compute and return the sequence of time-dependent layer- and parameter- gradients via ImplicitDiff on the current mini-batch
    """

    def __init__(self, network, cost_function, max_iterations=100):
        """Creates an instance of ImplicitDiff

        Args:
            network (Network): the network to train
            cost_function (CostFunction): the cost function to optimize
            max_iterations (int, optional): the number of iterations performed to compute the gradients. Default: 100
        """

        self._network = network
        self._cost_function = cost_function
        self.max_iterations = max_iterations

    def compute_gradient(self):
        """Computes the parameter gradients for the current mini-batch via implicit differentiation, and assign them to the .grad attribute of the parameters
        
        It is assumed that the network is at equilibrium when calling the method
        """

        equilibrium_state = [layer.state for layer in self._network.layers()]  # we store the state of the network at equilibrium

        for layer in self._network.layers(): layer.state.requires_grad = True
        for param in self._network.params(): param.state.requires_grad = True

        # Compute the layer gradients at time t=0
        loss = self._cost_function.cost_fn().mean()
        layer_grads = torch.autograd.grad(loss, equilibrium_state, allow_unused=True, retain_graph=True)
        layer_grads = [torch.zeros_like(layer.state) if grad == None else grad for layer, grad in zip(self._network.layers(), layer_grads)]
        param_grads = [torch.zeros_like(param.state) for param in self._network.params()]

        # Compute the parameter gradients at time t>0
        self._network.step_synchronous()
        next_state = [layer.state for layer in self._network.layers()]

        for _ in range(self.max_iterations):
            # performs one step of Recurrent Backpropagation (the Almeida-Pineda algorithm) to compute the time-dependent layer- and parameter- gradients
            layer_grads_next = torch.autograd.grad(next_state, equilibrium_state, grad_outputs = layer_grads, retain_graph=True)
            param_grads_next = torch.autograd.grad(next_state, [param.state for param in self._network.params()], grad_outputs = layer_grads, retain_graph=True)
            layer_grads = layer_grads_next
            param_grads = [param + param_next for param, param_next in zip(param_grads, param_grads_next)]
            
        for layer in self._network.layers():
            layer.state = layer.state.detach()
            layer.state.requires_grad = False
        for param in self._network.params(): param.state.requires_grad = False

        for param, grad in zip(self._network.params(), param_grads): param.state.grad = grad

    def detailed_gradients(self, cumulative=True):
        """Compute and return the sequence of time-dependent layer- and parameter- gradients via ImplicitDiff on the current mini-batch

        It is assumed that the network is at equilibrium when calling the method

        Args:
            cumulative (bool, optional): if True, computes the cumulative gradients ; if False, computes the gradients increases. Default: False.

        Returns:
            grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
        """

        equilibrium_state = [layer.state for layer in self._network.layers()]  # we store the state of the network at equilibrium

        grads = dict()  # dictionary of time-dependent layer- and parameter- gradients
        for layer in self._network.layers(): grads[layer.name] = list()
        for param in self._network.params(): grads[param.name] = list()

        for layer_state in equilibrium_state: layer_state.requires_grad = True
        for param in self._network.params(): param.state.requires_grad = True

        # Compute the layer gradients at time t=0
        loss = self._cost_function.cost_fn().mean()
        layer_grads = torch.autograd.grad(loss, equilibrium_state, allow_unused=True, retain_graph=True)

        for layer, grad in zip(self._network.layers(), layer_grads):
            if grad == None: grad = torch.zeros_like(layer.state)  # if the grad is None, then the corresponding parameter is not a part of the computational graph, therefore its gradient is zero
            grads[layer.name].append(grad)

        # Compute the layer- and parameter- gradients at time t>0
        self._network.step_synchronous()
        next_state = [layer.state for layer in self._network.layers()]

        for iteration in range(self.max_iterations):
            # performs one step of Recurrent Backpropagation (the Almeida-Pineda algorithm) to compute the time-dependent layer- and parameter- gradients
            layer_grads = [grads[layer.name][-1] for layer in self._network.layers()]
            layer_grads_next = torch.autograd.grad(next_state, equilibrium_state, grad_outputs = layer_grads, retain_graph=True)
            param_grads_next = torch.autograd.grad(next_state, [param.state for param in self._network.params()], grad_outputs = layer_grads, retain_graph=True)
            if iteration < self.max_iterations-1:
                for layer, grad in zip(self._network.layers(), layer_grads_next): grads[layer.name].append(grad)
            for param, grad in zip(self._network.params(), param_grads_next): grads[param.name].append(grad)

        # We reset the state of the network as it was initially
        for layer, state in zip(self._network.layers(), equilibrium_state):
            layer.state = state
            layer.state.requires_grad = False
        for param in self._network.params():
            param.state.requires_grad = False

        # We transform the values of the grads dictionary into Tensors
        if cumulative: grads = {name: list(accumulate(sequence)) for name, sequence in grads.items()}

        return grads

    def __str__(self):
        return 'ImplicitDiff -- num_iterations={}'.format(self.max_iterations)


class ContrastiveLearning(Eqprop):
    """
    Class used to compute the parameter updates of Contrastive Learning (CL) in a network

    This class more generally implements the 'coupled learning' (CpL) algorithm, a modified version of contrastive learning with weakly clamped targets in the training phase
    The standard contrastive learning (CL) algorithm corresponds to training_mode = 'positive' and nudging = 1.0

    Since CL is algorithmically similar to equilibrium propagation (EP), the ContrastiveLearning class inherits from the Eqprop class and overrides the methods that implement the differences between the two algorithms.

    # TODO: explain that this implementation of CL is generalized to any cost function (not just the MSE)
    """

    def __init__(self, network, cost_function, max_iterations=15, training_mode='positive', nudging=1.0):
        """Creates an instance of ContrastiveLearning

        Args:
            network (Network): the network to train via contrastive learning
            cost_function (CostFunction): the cost function to optimize
            max_iterations (int, optional): the maximum number of iterations allowed to converge to equilibrium in the second phase of CL. Default: 15
            training_mode (str, optional): either Positively-perturbed, Negatively-perturbed or Centered. Default: 'positive'
            nudging (float, optional): the nudging value used to train via Eqprop. Default: 1.0
        """

        Eqprop.__init__(self, network, cost_function, max_iterations=max_iterations, training_mode=training_mode, nudging=nudging, use_constant_force=True, use_alternative_formula=False)

    def _compute_equilibrium(self, nudging):
        """Compute the equilibrium state corresponding to a given nudging value

        Overrides the implementation of Eqprop to change the nudging method

        Args:
            nudging (float): nudging value used to converge to equilibrium

        Returns:
            layers: dictionary of Tensors. The state of the layers at equilibrium
        """

        self._network._output_layer.is_free = False  # FIXME
        for iteration in range(self.max_iterations):
            # set the state of the output layer somewhere between the free output values and the target values
            if iteration == 0 or (not self._use_constant_force):
                output_gradients = self._cost_function.output_gradients()
                output = self._network.read_output() + nudging * output_gradients
                self._network.set_output(output)
            # perform one step of the network's dynamics
            self._network.step_synchronous()
        self._network._output_layer.is_free = True  # FIXME
        layers = {layer.name: layer.state for layer in self._network.layers(free_only=False)}

        return layers

    def _compute_trajectory(self, nudging):
        """Compute the trajectory (pre-activations and activations) and parameter gradients at each step
        
        Overrides the implementation of Eqprop to change the nudging method

        Args:
            nudging (float): nudging value used to compute the trajectory

        Returns:
            pre_activations: list of dictionary of Tensors. The layer pre-activations at each step of the trajectory
            activations: list of dictionary of Tensors. The layer activations at each step of the trajectory
        """

        pre_activations = []
        activations = []

        self._network._output_layer.is_free = False  # FIXME
        for iteration in range(self.max_iterations):
            # set the force at the output layer proportionally to the output gradients
            if iteration == 0 or (not self._use_constant_force):
                output_gradients = self._cost_function.output_gradients()
                output = self._network.read_output() + nudging * output_gradients
                self._network.set_output(output)
            # perform one step of the network's dynamics and store the layers' pre-activations and activations
            pre_activations.append(
                {layer.name: layer.pre_activate() for layer in self._network.layers(free_only=False)}
                )
            self._network.step_synchronous()
            activations.append(
                {layer.name: layer.state for layer in self._network.layers(free_only=False)}
                )
        self._network._output_layer.is_free = True  # FIXME
        
        return pre_activations, activations

    def __str__(self):
        return 'Contrastive Learning -- mode={}, nudging={}, num_iterations={}'.format(self._training_mode, self._nudging, self.max_iterations)