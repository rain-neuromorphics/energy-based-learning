from abc import ABC, abstractmethod
from itertools import accumulate
import torch

from model.function.interaction import Function, SumSeparableFunction
from model.minimizer.minimizer import ParamUpdater, GradientDescentUpdater



class Nudging(Function):
    """Class to scale a function by a nudging factor

    Attributes
    ----------
    _function (Function): the function to scale by a nudging factor
    nudging (float): the nudging value
    """

    # FIXME: how to deal with the case where the class is a QFunction or LFunction?

    def __init__(self, function):
        """Initializes an instance of Nudging

        Args:
            function (Function): the function to scale by a nudging factor
        """
        self._function = function
        self._nudging = 0.

        Function.__init__(self, function.layers(), function.params())

    @property
    def nudging(self):
        """Get and sets the nudging value"""
        return self._nudging

    @nudging.setter
    def nudging(self, nudging):
        self._nudging = nudging

    def eval(self):
        """Value of the nudging function. This is the function's value times the nudging value.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the value of an example in the current mini-batch
        """
        return self._nudging * self._function.eval()

    def a_coef_fn(self, layer):
        """Returns the function that computes the coefficient a for a given layer"""
        a_coef_fn = self._function.a_coef_fn(layer)
        return lambda: self._nudging * a_coef_fn()

    def b_coef_fn(self, layer):
        """Returns the function that computes the coefficient b for a given layer"""
        b_coef_fn = self._function.b_coef_fn(layer)
        return lambda: self._nudging * b_coef_fn()



class AugmentedFunction(SumSeparableFunction):
    """
    Class to augment an 'energy' function by a 'cost' function.

    Attributes
    ----------
    nudging (float): nudging value

    Methods
    -------
    eval()
        Returns the value of the augmented function (for the current configuration)
    """

    def __init__(self, energy_fn, cost_fn):
        """Creates an instance of AugmentedFunction"""

        layers = energy_fn.layers()
        params = energy_fn.params()

        nudging = Nudging(cost_fn)
        interactions = [energy_fn, nudging]

        SumSeparableFunction.__init__(self, layers, params, interactions)
        self._nudging = nudging
        self._energy_fn = energy_fn

        # FIXME: what if the cost function does not have the same layers and/or params as the energy function?


    @property
    def nudging(self):
        """Get and sets the nudging value"""
        return self._nudging.nudging

    @nudging.setter
    def nudging(self, nudging):
        self._nudging.nudging = nudging

    def eval(self):
        """Returns the value of the augmented function for the current configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of values for each of the examples in the current mini-batch
        """

        return self._energy_fn.eval() + self._nudging.eval()



class GradientEstimator(ABC):
    """
    Abstract class for computing or estimating the parameter gradients in a bilevel optimization problem

    A bilevel optimization problem is a problem of the form:

    minimize C(s(theta))
    subject to s(theta) = argmin_s E(theta,s)

    where theta and s are variables, and C(s) and E(theta,s) are scalar functions. Specifally:
    - theta is a set of 'parameter variables'
    - s is a set of 'layer variables'
    - C(s) is a 'cost function' 
    - E(theta,s) is an 'energy function'

    This abstract class allows to compute (or estimate) the gradient of C(s(theta)) wrt theta. It is used in particular:
    - to perform SGD (stochastic gradient descent) in an energy-based system,
    - to check the matching gradients property (MGP) in an energy-based system.

    Methods
    -------
    compute_gradient()
        Compute and return the parameter gradients
    detailed_gradients()
        Compute and return the sequence of time-dependent layer- and parameter- gradients
    """

    @abstractmethod
    def compute_gradient(self):
        """Compute the parameter gradients"""
        pass

    @abstractmethod
    def detailed_gradients(self):
        """Compute the sequence of time-dependent layer- and parameter- gradients"""
        pass



class EquilibriumProp(GradientEstimator):
    """
    Class for estimating the parameter gradients of a cost function in an energy-based system via equilibrium propagation (EP)

    Equilibrium propagation (EP) estimates the gradient (wrt to theta) of C(s(theta)) under the constraint that s(theta) = argmin_s E(theta,s)

    The EP gradient estimator depends on two scalars: nudging_1 and nudging_2. It is given by the formula
    estimator(nudging_1, nudging_2) := [ dE(theta,s(nudging_2))/dtheta - dE(theta,s(nudging_1))/dtheta ] / [ nudging_2 - nudging_1 ]
    where s(nudging) = argmin_s [ E(theta,s) + nudging * C(s) ]

    The scalars nudging_1 and nudging_2 are determined by the attributes 'variant' and 'nudging' of the EquilibriumProp class, as follows:
    * if the variant is 'positive', then nudging_1 = 0 and nudging_2 = + nudging
    * if the variant is 'negative', then nudging_1 = - nudging and nudging_2 = 0
    * if the variant is 'centered', then nudging_1 = - nudging and nudging_2 = + nudging

    Attributes
    ----------
    _params (list of Parameters): the parameters whose gradients we want to estimate via equilibrium propagation
    _layers (list of Layers): the layers that minimize the augmented energy function E(theta,s) + nudging * C(s)
    _energy_minimizer (Minimizer): the algorithm used to minimize the augmented energy function (in the perturbed phase of EP)
    variant (str): either 'positive' (positively-perturbed EP), 'negative' (negatively-perturbed EP) or 'centered' (centered EP)
    nudging (float): the nudging value used to estimate the parameter gradients via EP
    use_alternative_formula (bool): which equilibrium propagation formula is used to estimating the parameter gradients. Either the 'standard' formula (False) or the 'alternative' formula (True)

    Methods
    -------
    compute_gradient()
        Compute and return the parameter gradients via EP
    detailed_gradients()
        Compute and return the sequence of time-dependent layer- and parameter- EP gradients
    """

    def __init__(self, params, layers, energy_fn, cost_fn, energy_minimizer, variant='centered', nudging=0.25, use_alternative_formula=False):
        """Creates an instance of equilibrium propagation

        Args:
            params (list of Parameters): the parameters whose gradients we want to estimate via equilibrium propagation
            layers (list of Layers): the Layers that minimize the augmented energy function
            energy_fn (Function): the energy function
            cost_fn (Function): the cost function
            energy_minimizer (Minimizer): the algorithm used to minimize the augmented energy function
            variant (str, optional): either 'positive' (positively-perturbed EP), 'negative' (negatively-perturbed EP) or 'centered' (centered EP). Default: 'centered'
            nudging (float, optional): the nudging value used to estimate the parameter gradients via EP. Default: 0.25
            use_alternative_formula (bool, optional): which equilibrium propagation formula is used to estimating the parameter gradients. Either the 'standard' formula (False) or the 'alternative' formula (True). Default: False
        """

        self._params = params
        self._layers = layers

        self._param_updaters = [ParamUpdater(param, energy_fn) for param in params]
        # self._param_updaters_cost = [ParamUpdater(param, cost_fn) for param in cost_fn.params()]
        self._layer_updaters = [GradientDescentUpdater(layer, energy_fn) for layer in layers]  # FIXME: one should be using the layer updaters of the energy minimizer

        self._augmented_fn = energy_fn  # AugmentedFunction(energy_fn, cost_fn)
        self._energy_minimizer = energy_minimizer
        self._cost_fn = cost_fn

        self._nudging = nudging
        self._variant = variant
        self._set_nudgings()

        self._use_alternative_formula = use_alternative_formula

    @property
    def nudging(self):
        """Get and sets the nudging value used for estimating the parameter gradients via equilibrium propagation"""
        return self._nudging

    @nudging.setter
    def nudging(self, nudging):
        if nudging > 0.:
            self._nudging = nudging
            self._set_nudgings()
        else: raise ValueError("expected a positive nudging value, but got {}".format(nudging))

    @property
    def variant(self):
        """Get and sets the training mode ('positive', 'negative' or 'centered')"""
        return self._variant

    @variant.setter
    def variant(self, variant):
        if variant in ['positive', 'negative', 'centered']:
            self._variant = variant
            self._set_nudgings()
        else: raise ValueError("expected 'positive', 'negative' or 'centered' but got {}".format(variant))

    @property
    def use_alternative_formula(self):
        """Get and sets the use_alternative_formula attribute"""
        return self._use_alternative_formula

    @use_alternative_formula.setter
    def use_alternative_formula(self, use_alternative_formula):
        if use_alternative_formula in [True, False]: self._use_alternative_formula = use_alternative_formula
        else: raise ValueError("expected True or False, but got {}".format(use_alternative_formula))

    def compute_gradient(self):
        """Estimates the parameter gradient via equilibrium propagation

        The gradient depends on the EP variant (positively-perturbed EP, negatively-perturbed EP, or centered EP) and the nudging strength.
        To get a better gradient estimate, this method must be called when the layers are at their 'free state' (equilibrium for nudging=0).
        
        Returns:
            param_grads: list of Tensor of shape param_shape and type float32. The parameter gradients
        """

        # TODO: this implementation is likely suboptimal in the case of e.g. the Readout cost function
        cost_grads = [self._cost_fn._grad(param, mean=True) for param in self._cost_fn.params()]  # compute the direct gradient of C, if C explicitly depends on parameters
        
        # First phase: compute the first equilibrium state of the layers
        layers_free = [layer.state for layer in self._layers]  # hack: we store the `free state' (i.e. the equilibrium state of the layers with nudging=0)
        self._augmented_fn.nudging = self._first_nudging
        layers_first = self._energy_minimizer.compute_equilibrium()
        
        # Second phase: compute the second equilibrium state of the layers
        for layer, state in zip(self._layers, layers_free): layer.state = state  # hack: we start the second phase from the `free state' again
        self._augmented_fn.nudging = self._second_nudging
        layers_second = self._energy_minimizer.compute_equilibrium()

        # Compute the parameter gradients with either the standard EquilibriumProp formula, or the alternative EquilibriumProp formula
        if self._use_alternative_formula:
            param_grads = self._alternative_param_grads(layers_free, layers_first, layers_second)
        else:
            param_grads = self._standard_param_grads(layers_first, layers_second)

        return param_grads + cost_grads
    
    def detailed_gradients(self, cumulative=True):
        """Compute and return the sequence of time-dependent layer- and parameter- EP gradients

        Calling this method leaves the state of the layers unchanged
        For the method to return the correct detailed gradients of EP, the layers must be at their 'free state' (equilibrium for nudging=0) when calling the method

        Args:
            cumulative (bool, optional): if True, computes the cumulative gradients ; if False, computes the gradients increases. Default: True.

        Returns:
            grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
        """

        # First phase: compute the layers' activations along the first trajectory
        layers_free = [layer.state for layer in self._layers]  # we store the `free state' (i.e. the equilibrium state of the layers with nudging=0)
        self._augmented_fn.nudging = self._first_nudging
        trajectory_first = self._energy_minimizer.compute_trajectory()
        
        # Second phase: compute the layers' activations along the second trajectory
        for layer, state in zip(self._layers, layers_free): layer.state = state  # we start over from the `free state' again
        self._augmented_fn.nudging = self._second_nudging
        trajectory_second = self._energy_minimizer.compute_trajectory()

        # Compute the layer gradients
        trajectory_first = [dict(zip(trajectory_first, v)) for v in zip(*trajectory_first.values())]  # transform the dictionary of lists into a list of dictionaries
        trajectory_second = [dict(zip(trajectory_second, v)) for v in zip(*trajectory_second.values())]  # transform the dictionary of lists into a list of dictionaries
        layer_grads = [self._layer_grads(first, second) for first, second in zip(trajectory_first[:-1], trajectory_second[:-1])]
        layer_grads = list(map(list, zip(*layer_grads)))  # transpose the list of lists: transform the time-wise layer-wise gradients into layer-wise time-wise gradients

        # Compute the parameter gradients with either the standard EquilibriumProp formula, or the alternative EquilibriumProp formula
        if self._use_alternative_formula:
            param_grads = [self._alternative_param_grads(layers_free, first, second) for first, second in zip(trajectory_first[1:], trajectory_second[1:])]
        else:
            param_grads = [self._standard_param_grads(first, second) for first, second in zip(trajectory_first[1:], trajectory_second[1:])]
        param_grads = list(map(list, zip(*param_grads)))  # transpose the list of lists: transform the time-wise parameter-wise gradients into parameter-wise time-wise gradients

        # Store the layer-wise and parameter-wise time-wise gradients in a dictionary
        grads = dict()
        for layer, gradients in zip(self._layers, layer_grads): grads[layer.name] = gradients
        for param, gradients in zip(self._params, param_grads): grads[param.name] = gradients
        
        # Transform the time-wise gradients into time-wise increases if required
        if not cumulative:
            for layer in self._layers:
                layer_grads = grads[layer.name]
                grads[layer.name] = [layer_grads[0]] + [j-i for i, j in zip(layer_grads[:-1], layer_grads[1:])]
            for param in self._params:
                param_grads = grads[param.name]
                grads[param.name] = [param_grads[0]] + [j-i for i, j in zip(param_grads[:-1], param_grads[1:])]

        # Reset the layers to their `free state' values, where they were initially
        for layer, state in zip(self._layers, layers_free): layer.state = state
        
        return grads

    def _standard_param_grads(self, layers_first, layers_second):
        """Compute the parameter gradients using the standard EquilibriumProp formula

        Args:
            layers_first (dictionary of Tensors): the activations of the layers at the first state
            layers_second (dictionary of Tensors): the activations of the layers at the second state

        Returns:
            param_grads: list of Tensors. The parameter gradients
        """

        # Compute the cost gradients of the free state
        # for layer, free in zip(self._layers, layers_free): layer.state = free
        # grads_free = [updater.grad() for updater in self._param_updaters_cost]

        # Compute the energy gradients of the first state
        for layer in self._layers: layer.state = layers_first[layer.name]
        grads_first = [updater.grad() for updater in self._param_updaters]

        # Compute the energy gradients of the second state
        for layer in self._layers: layer.state = layers_second[layer.name]
        grads_second = [updater.grad() for updater in self._param_updaters]

        # Compute the parameter gradients
        param_grads = [(second - first) / (self._second_nudging - self._first_nudging) for first, second in zip(grads_first, grads_second)]

        return param_grads

    def _alternative_param_grads(self, layers_free, layers_first, layers_second):
        """Compute the parameter gradients using the alternative EquilibriumProp formula

        Args:
            layers_free (list of Tensors): the activations of the layers at the free state
            layers_first (dictionary of Tensors): the activations of the layers at the first state
            layers_second (dictionary of Tensors): the activations of the layers at the second state

        Returns:
            param_grads: list of Tensors. The parameter gradients
        """

        direction = {layer.name: (layers_second[layer.name] - layers_first[layer.name]) / (self._second_nudging - self._first_nudging) for layer in self._layers}
        for layer, state in zip(self._layers, layers_free): layer.state = state  # hack: we need to reset the state of the layers at the `free state'
        param_grads = [updater.second_fn(direction) for updater in self._param_updaters]

        return param_grads

    def _layer_grads(self, layers_first, layers_second):
        """Compute the layer gradients given the activations of the first and second states

        Args:
            layers_first (dictionary of Tensors): the activations of the layers at the first state
            layers_second (dictionary of Tensors): the activations of the layers at the second state

        Returns:
            layer_grads: list of Tensors. The layer gradients
        """

        # FIXME: the gradients of the output layer are wrong because we need to set the correct 'output force' (nudging) at each time step.

        # Compute the energy gradients of the first state
        for layer in self._layers: layer.state = layers_first[layer.name]
        grads_first = [updater.grad() for updater in self._layer_updaters]

        # Compute the energy gradients of the second state
        for layer in self._layers: layer.state = layers_second[layer.name]
        grads_second = [updater.grad() for updater in self._layer_updaters]

        # Compute the layer gradients
        batch_size = self._layers[0].state.size(0)
        layer_grads = [(second - first) / ((self._second_nudging - self._first_nudging) * batch_size) for first, second in zip(grads_first, grads_second)]

        return layer_grads

    def _set_nudgings(self):
        """Sets the nudging values of the first and second states, depending on the attributes variant and nudging

        first_nudging: nudging value used to compute the first state of equilibrium propagation
        second_nudging: nudging value used to compute the second state of equilibrium propagation
        """

        if self._variant == "positive":
            self._first_nudging = 0.
            self._second_nudging = self._nudging
        elif self._variant == "negative":
            self._first_nudging = -self._nudging
            self._second_nudging = 0.
        elif self._variant == "centered":
            self._first_nudging = -self._nudging
            self._second_nudging = self._nudging
        else:
            # TODO: this raise of error is redundant
            raise ValueError("expected 'positive', 'negative' or 'centered' but got {}".format(self._variant))

    def __str__(self):
        formula = 'alternative' if self._use_alternative_formula else 'standard'
        return 'Equilibrium propagation -- mode={}, nudging={}, formula={}'.format(self._variant, self._nudging, formula)


class Backprop(GradientEstimator):
    """
    Class used to compute the parameter gradients of a cost function in an energy-based system via backpropagation (automatic differentiation)

    Attributes
    ----------
    _params (list of Parameters): the parameters whose gradients we want to estimate via backpropagation
    _layers (list of Layers): the layers that minimize the augmented energy function
    cost_fn (CostFunction): the cost function to optimize
    energy_minimizer (Minimizer): the algorithm used to minimize the augmented energy function

    Methods
    -------
    compute_gradient()
        Computes the parameter gradients via backpropagation
    detailed_gradients()
        Compute and return the sequence of time-dependent layer- and parameter- gradients via backpropagation
    """

    def __init__(self, params, layers, cost_fn, energy_minimizer):
        """Creates an instance of backpropagation

        Args:
            params (list of Parameters): the parameters whose gradients we want to estimate via backpropagation
            layers (list of Layers): the Layers that minimize the augmented energy function
            cost_fn (CostFunction): the cost function whose parameter gradients we want to compute
            energy_minimizer (Minimizer): the algorithm used to minimize the augmented energy function
        """

        self._params = params
        self._layers = layers
        self._cost_fn = cost_fn
        self._energy_minimizer = energy_minimizer

    def compute_gradient(self):
        """Computes the parameter gradients via backpropagation
        
        Returns:
            param_grads: list of Tensor of shape param_shape and type float32. The parameter gradients
        """

        # TODO: this implementation is likely suboptimal in the case of e.g. the Readout cost function
        cost_grads = [self._cost_fn._grad(param, mean=True) for param in self._cost_fn.params()]  # compute the direct gradient of C, if C explicitly depends on parameters

        for param in self._params: param.state.requires_grad = True

        self._energy_minimizer.compute_equilibrium()
        cost_mean = self._cost_fn.eval().mean()
        param_grads = torch.autograd.grad(cost_mean, [param.state for param in self._params])

        for param in self._params: param.state.requires_grad = False
        for layer in self._layers: layer.state = layer.state.detach()

        return list(param_grads) + cost_grads

    '''def detailed_gradients(self):
        """Compute and return the sequence of time-dependent layer- and parameter- gradients via backpropagation

        For the MGP property to be satisfied, the layers must be at equilibrium when calling the method.
        The Pros of this implementation is that it does not use the backward method, and thus does not violate the rule that the .grad attribute should only be used to assign gradients (not to retrieve gradients)
        The Cons of this implementation is that it is quadratic wrt the number of iterations (whereas the other implementation is linear in time)

        # TODO: can we merge ideas from this implementation with the other implementation, so that we get a linear algorithm that does not use backward()?

        Returns:
            grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
        """

        equilibrium_state = [layer.state for layer in self._layers]  # we store the state of the layers, so we can reset it to its initial state at the end of the method

        layers_grads = {layer.name: [] for layer in self._layers}  # dictionary of time-dependent layer-gradients
        params_grads = {param.name: [] for param in self._params}  # dictionary of time-dependent parameter-gradients

        for iteration in range(self.max_iterations):

            # We compute the gradient (wrt the layers) of the cost of the state of the output layer at time 'iteration', starting from the equilibrium state (the 'projected cost function')
            for layer, state in zip(self._layers, equilibrium_state): layer.state = state  # we start from the equilibrium state
            for layer_state in equilibrium_state: layer_state.requires_grad = True
            for _ in range(iteration): self._energy_minimizer.step()
            loss = self._cost_fn.eval().mean()
            grads = torch.autograd.grad(loss, equilibrium_state, allow_unused=True)
            for layer_state in equilibrium_state: layer_state.requires_grad = False

            # We update the layers_grads dictionary
            for layer, grad in zip(self._layers, grads):
                if grad == None: grad = torch.zeros_like(layer.state)  # if the grad is None, then the corresponding layer is not a part of the computational graph, therefore its gradient is zero
                layers_grads[layer.name].append(grad)

            # We compute the gradient (wrt the parameters) of the cost of the state of the output layer at time 'iteration', starting from the equilibrium state (the 'projected cost function')
            for layer, state in zip(self._layers, equilibrium_state): layer.state = state  # we start from the equilibrium state
            for param in self._params: param.state.requires_grad = True
            for _ in range(iteration+1): self._energy_minimizer.step()
            loss = self._cost_fn.eval().mean()
            grads = torch.autograd.grad(loss, [param.state for param in self._params], allow_unused=True)
            for param in self._params: param.state.requires_grad = False
            
            # We update the params_grads dictionary
            for param, grad in zip(self._params, grads):
                if grad == None: grad = torch.zeros_like(param.state)  # if the grad is None, then the corresponding parameter is not a part of the computational graph, therefore its gradient is zero
                params_grads[param.name].append(grad)

        # We transform the values of the grads dictionary into Tensors
        grads = dict()
        for layer in self._layers: grads[layer.name] = torch.cumsum(torch.stack(layers_grads[layer.name]), dim=0)
        for param in self._params: grads[param.name] = torch.stack(params_grads[param.name])
        
        for layer, state in zip(self._layers, equilibrium_state): layer.state = state  # we set the state of the layers where it was initially
        
        return grads'''

    def detailed_gradients(self, cumulative=True):
        """Compute and return the sequence of time-dependent layer- and parameter- gradients via Backprop.
        
        For the matching gradient property (MGP) to be satisfied, the layers must be at equilibrium when calling the method.

        Args:
            cumulative (bool, optional): if True, computes the cumulative gradients ; if False, computes the gradients increases. Default: True.

        Returns:
            grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
        """

        # We store the initial state of the layers and parameters, so we can reset them at the end of the method
        layers_state = [layer.state for layer in self._layers]
        params_state = [param.state for param in self._params]

        for param in self._params: param.state.requires_grad = True
        for layer in self._layers: layer.state.requires_grad = True
        
        # During the forward pass, we keep all the tensors of the computational graph in a 'trajectories' dictionary
        trajectories = self._energy_minimizer.compute_trajectory()
        
        # We perform the backward pass, and before that we ask all the variables (layers and parameters) to retain the partial deriavtive computed in the backward pass
        for trajectory in trajectories.values():
            for variable in trajectory:
                variable.retain_grad()
        loss = self._cost_fn.eval().mean()
        loss.backward()
        
        # After the backward pass, we read all the gradients in the .grad attributes of the variables (layers and parameters)
        # and we store these gradients in the 'grads' dictionary
        grads = dict()
        for layer in self._layers:
            layer_grads = [state.grad if state.grad is not None else torch.zeros_like(state) for state in trajectories[layer.name]]
            layer_grads = list(reversed(layer_grads))
            if cumulative: layer_grads = list(accumulate(layer_grads))
            grads[layer.name] = layer_grads[:-1]
        for param in self._params:
            param_grads = [state.grad if state.grad is not None else torch.zeros_like(state) for state in trajectories[param.name]]
            param_grads = list(reversed(param_grads))
            if not cumulative: param_grads = [param_grads[0]] + [j-i for i, j in zip(param_grads[:-1], param_grads[1:])]
            grads[param.name] = param_grads[:-1]

        # Finally, we reset the state of the layers and parameters where they were initially
        for layer, state in zip(self._layers, layers_state): layer.state = state
        for param, state in zip(self._params, params_state): param.state = state
        
        for layer in self._layers: layer.state.requires_grad = False
        for param in self._params: param.state.requires_grad = False

        return grads

    def __str__(self):
        return 'Backpropagation'


class RecurrentBackprop(GradientEstimator):
    """
    Class used to compute the parameter gradients of a cost function in an energy-based system via implicit differentiation (RecurrentBackprop).
    
    The algorithm implemented here is also known as `recurrent backpropagation', or the Almeida-Pineda algorithm.
    
    Attributes
    ----------
    _params (list of Parameters): the parameters whose gradients we want to estimate via implicit differentiation
    _layers (list of Layers): the Layers that minimize an energy function
    _cost_fn (CostFunction): the cost function to optimize
    energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function

    Methods
    -------
    compute_gradient()
        Computes the parameter gradients via implicit differentiation
    detailed_gradients()
        Compute and return the sequence of time-dependent layer- and parameter- gradients via RecurrentBackprop
    """

    def __init__(self, params, layers, cost_fn, energy_minimizer):
        """Creates an instance of RecurrentBackprop

        Args:
            params (list of Parameters): the parameters whose gradients we want to estimate via equilibrium propagation
            layers (list of Layers): the Layers that minimize an energy function
            cost_fn (CostFunction): the cost function to optimize
            energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function
        """

        self._params = params
        self._layers = layers
        self._cost_fn = cost_fn
        self._energy_minimizer = energy_minimizer

    def compute_gradient(self):
        """Computes the parameter gradients via implicit differentiation
        
        To compute the correct gradients, the layers must be in their 'free state' (equilibrium) when calling the method

        Returns:
            param_grads: list of Tensor of shape param_shape and type float32. The parameter gradients
        """

        # TODO: we use all the layers (including the input layer). Need to check if this is fine

        # TODO: this implementation is likely suboptimal in the case of e.g. the Readout cost function
        cost_grads = [self._cost_fn._grad(param, mean=True) for param in self._cost_fn.params()]  # compute the direct gradient of C, if C explicitly depends on parameters

        equilibrium_state = [layer.state for layer in self._layers]  # we store the state of the layers at equilibrium

        for layer in self._layers: layer.state.requires_grad = True
        for param in self._params: param.state.requires_grad = True

        # Compute the layer gradients at time t=0
        loss = self._cost_fn.eval().mean()
        layer_grads = torch.autograd.grad(loss, equilibrium_state, allow_unused=True, retain_graph=True)
        layer_grads = [torch.zeros_like(layer.state) if grad == None else grad for layer, grad in zip(self._layers, layer_grads)]
        param_grads = [torch.zeros_like(param.state) for param in self._params]

        # Compute the parameter gradients at time t>0
        self._energy_minimizer.step()
        next_state = [layer.state for layer in self._layers]

        for _ in range(self._energy_minimizer.num_iterations):
            # performs one step of Recurrent Backpropagation (the Almeida-Pineda algorithm) to compute the time-dependent layer- and parameter- gradients
            layer_grads_next = torch.autograd.grad(next_state, equilibrium_state, grad_outputs = layer_grads, retain_graph=True)
            param_grads_next = torch.autograd.grad(next_state, [param.state for param in self._params], grad_outputs = layer_grads, retain_graph=True)
            layer_grads = layer_grads_next
            param_grads = [param + param_next for param, param_next in zip(param_grads, param_grads_next)]
            
        for layer in self._layers:
            layer.state = layer.state.detach()
            layer.state.requires_grad = False
        for param in self._params: param.state.requires_grad = False

        return list(param_grads) + cost_grads

    def detailed_gradients(self, cumulative=True):
        """Compute and return the sequence of time-dependent layer- and parameter- gradients via RecurrentBackprop

        To compute the correct gradients, the layers must be at equilibrium when calling the method

        Args:
            cumulative (bool, optional): if True, computes the cumulative gradients ; if False, computes the gradients increases. Default: True.

        Returns:
            grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
        """

        equilibrium_state = [layer.state for layer in self._layers]  # we store the state of the layers at equilibrium

        grads = dict()  # dictionary of time-dependent layer- and parameter- gradients
        for layer in self._layers: grads[layer.name] = list()
        for param in self._params: grads[param.name] = list()

        for layer_state in equilibrium_state: layer_state.requires_grad = True
        for param in self._params: param.state.requires_grad = True

        # Compute the layer gradients at time t=0
        loss = self._cost_fn.eval().mean()
        layer_grads = torch.autograd.grad(loss, equilibrium_state, allow_unused=True, retain_graph=True)

        for layer, grad in zip(self._layers, layer_grads):
            if grad == None: grad = torch.zeros_like(layer.state)  # if the grad is None, then the corresponding parameter is not a part of the computational graph, therefore its gradient is zero
            grads[layer.name].append(grad)

        # Compute the layer- and parameter- gradients at time t>0
        self._energy_minimizer.step()
        next_state = [layer.state for layer in self._layers]

        for iteration in range(self._energy_minimizer.num_iterations):
            # performs one step of Recurrent Backpropagation (the Almeida-Pineda algorithm) to compute the time-dependent layer- and parameter- gradients
            layer_grads = [grads[layer.name][-1] for layer in self._layers]
            layer_grads_next = torch.autograd.grad(next_state, equilibrium_state, grad_outputs = layer_grads, retain_graph=True)
            param_grads_next = torch.autograd.grad(next_state, [param.state for param in self._params], grad_outputs = layer_grads, retain_graph=True)
            if iteration < self._energy_minimizer.num_iterations-1:
                for layer, grad in zip(self._layers, layer_grads_next): grads[layer.name].append(grad)
            for param, grad in zip(self._params, param_grads_next): grads[param.name].append(grad)

        # We reset the state of the layers as it was initially
        for layer, state in zip(self._layers, equilibrium_state):
            layer.state = state
            layer.state.requires_grad = False
        for param in self._params:
            param.state.requires_grad = False

        # We transform the values of the grads dictionary into Tensors
        if cumulative: grads = {name: list(accumulate(sequence)) for name, sequence in grads.items()}

        return grads

    def __str__(self):
        return 'Recurrent backpropagation'



class ContrastiveLearning(EquilibriumProp):
    """
    Class for estimating the parameter gradients of a cost function in an energy-based system via contrastive learning (CL)

    This class more generally implements the perturbation method of 'coupled learning', a modified version of contrastive learning and equilibrium propagation with weakly clamped outputs in the training phase.
    Since equilibrium propagation, contrastive learning and coupled learning are algorithmically similar, the ContrastiveLearning class inherits from the EquilibriumProp class and overrides the perturbation method, i.e. the difference between the two algorithms.

    This implementation of contrastive learning is generalized to any cost function (not just the MSE)
    """

    def __init__(self, params, layers, energy_fn, cost_fn, energy_minimizer, variant='positive', nudging=1.0):
        """Creates an instance of contrastive learning

        Args:
            params (list of Parameters): the parameters whose gradients we want to estimate via equilibrium propagation
            layers (list of Layers): the Layers that minimize the energy function
            energy_fn (Function): the energy function
            cost_fn (Function): the cost function
            energy_minimizer (Minimizer): the algorithm used to minimize the energy function
            variant (str, optional): either 'positive' (positively-perturbed CpL), 'negative' (negatively-perturbed CpL) or 'centered' (centered CpL). Default: 'positive'
            nudging (float, optional): the nudging value used to estimate the parameter gradients via EP. Default: 1.0
        """

        EquilibriumProp.__init__(self, params, layers, energy_fn, cost_fn, energy_minimizer, variant, nudging, use_alternative_formula=False)

        self._cost_fn = cost_fn

    def compute_gradient(self):
        """Calculates the direction of the weight updates of contrastive learning

        The weight updates depends on the coupled learning variant (positively-perturbed CpL, negatively-perturbed CpL, or centered CpL) and the nudging strength.
        
        Returns:
            param_grads: list of Tensor of shape param_shape and type float32. The directions of the weight updates
        """

        # TODO: this implementation is likely suboptimal in the case of e.g. the Readout cost function
        cost_grads = [self._cost_fn._grad(param, mean=True) for param in self._cost_fn.params()]  # compute the direct gradient of C, if C explicitly depends on parameters

        # First phase: compute the first equilibrium state of the layers
        layers_free = [layer.state for layer in self._layers]  # hack: we store the `free state' (i.e. the equilibrium state of the layers with nudging=0)
        output_gradients = self._cost_fn._grad(self._layers[-1], mean=False)  # FIXME
        self._layers[-1].state = self._layers[-1].state - self._first_nudging * output_gradients  # FIXME
        layers_first = self._energy_minimizer.compute_equilibrium()
        
        # Second phase: compute the second equilibrium state of the layers
        for layer, state in zip(self._layers, layers_free): layer.state = state  # hack: we start the second phase from the `free state' again
        output_gradients = self._cost_fn._grad(self._layers[-1], mean=False)  # FIXME
        self._layers[-1].state = self._layers[-1].state - self._second_nudging * output_gradients  # FIXME
        layers_second = self._energy_minimizer.compute_equilibrium()

        param_grads = self._standard_param_grads(layers_first, layers_second)

        return param_grads + cost_grads

    def __str__(self):
        return 'Contrastive learning -- mode={}, nudging={}'.format(self._variant, self._nudging)