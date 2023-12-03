from abc import ABC, abstractmethod
import torch



class Statistic(ABC):
    """Abstract class for statistics

    Attributes
    ----------

    Methods
    -------
    do_measurement()
        Do a measurement and adjusts the value of the statistic accordingly
    get()
        Returns the current value of the statistic
    reset()
        Resets the statistic to zero
    """

    @abstractmethod
    def do_measurement(self):
        """Do a measurement and adjusts the value of the statistic accordingly"""
        pass

    @abstractmethod
    def get(self):
        """Returns the current value of the statistic"""
        pass

    @abstractmethod
    def reset(self):
        """Resets the statistic to zero"""
        pass


class Counter(Statistic):
    """Class to count how many examples of the dataset have been processed

    Attributes
    ----------
    _num_examples (int): the number of examples processed so far
    _dataset_size (int): the number of examples in the dataset

    Methods
    -------
    _count_fn()
        Counts the number of examples in the current mini-batch
    """

    display = True
    name = None
    option = None
    display_name = None

    def __init__(self, network, dataset_size):
        """Initializes an instance of Counter

        Args:
            network (Network): the network where we count the number of examples processed
            dataset_size (int): the number of examples in the dataset
        """

        self._network = network
        self._num_examples = 0

        self._dataset_size = dataset_size

    def _count_fn(self):
        """Count function"""
        return self._network.layers()[0].state.shape[0]

    def do_measurement(self):
        """Measures the statistics and adds it to the sum"""
        self._num_examples += self._count_fn()

    def get(self):
        """Returns the number of examples processed so far"""
        return self._num_examples

    def reset(self):
        """Resets the number of examples processed to zero"""
        self._num_examples = 0

    def __str__(self):
        string = 'Example {:5d}/{}'.format(self._num_examples, self._dataset_size)
        return string
        

class ErrorFinder(Statistic):
    """
    Class used to find the examples in a dataset that a given network misclassifies

    Attributes
    ----------
    name (str): the name of the statistic
    _network (Network): the model to evaluate
    _list_indices (list of int): list of indices of misclassified images in the dataset
    display (bool): whether or not this statistic is displayed in the summeary logs of each epoch  # FIXME

    Methods
    -------
    do_measurement()
        Measures the statistics and adds it to the sum
    get()
        Returns the list of indices
    reset()
        Resets the list of indices to an empty list
    """

    display = False
    name = None  # 'Error_finder'
    option = None
    display_name = None

    def __init__(self, network, evaluator):
        """Initializes an instance of ErrorFinder

        Args:
            network (Network): the network used to classify the examples of the dataset
            evaluator (Evaluator): the evaluator of the network
        """

        self._network = network
        self._evaluator = evaluator

        self._list_indices = []

    def do_measurement(self):
        """Returns the indices of the misclassified images in idx"""

        idx = self._evaluator.idx
        mask = self._network.error_fn()
        indices = idx[mask].numpy()

        self._list_indices.extend(indices)

    def get(self):
        """Returns the list of indices"""
        return self._list_indices

    def reset(self):
        """Resets the list of indices to an empty list"""
        self._list_indices = []

    def __str__(self):
        num_mistakes = len(self._list_indices)
        return 'Number of mistakes = {}'.format(num_mistakes)




class MeanStat(Statistic, ABC):
    """
    Abstract class used to accumulate statistics over the mini-batches to get statistics over the entire dataset.

    Attributes
    ----------
    name (str): the name of the statistics
    sum (float32): cumulative sum of the statistics
    num_steps (int): number of times the function "do_meaurement" has been called.
    percentage (bool): whether the statistics is played in percentage or not
    string (str): the string to be formatted.

    Methods
    -------
    do_measurement()
        Measures the statistics and adds it to the sum
    get()
        Returns the mean of the statistics
    reset()
        Resets sum and num_steps to zero
    """

    def __init__(self, display_name=None, name=None, precision=3, percentage=False, display=False):
        """
        Initializes an instance of MeanStat.
        
        Args:
            name (str): the name of the statistics
            precision (int, optional): the number of decimals to display. Default: 3.
            percentage (bool, optional): whether the statistics is displayed in percentage or not. Default: False.
        """

        self.display_name = display_name

        self.name = name

        self._sum = 0.
        self._num_examples = 0

        self._display_string = display_name + ' = {:.' + str(precision) + 'f}'
        if percentage: self._display_string += '%'

        self._percentage = percentage

        self.display = display

    @abstractmethod
    def _measure_fn(self):
        """function that measures the statistics"""
        pass

    def do_measurement(self):
        """Measures the statistics and adds it to the sum"""
        amount = self._measure_fn()
        self._sum += amount.sum().item()
        self._num_examples += amount.numel()

    def get(self):
        """Returns the mean statistics"""
        mean = self._sum / self._num_examples
        if self._percentage: mean *= 100.
        return mean

    def reset(self):
        """Resets the variables to zero"""
        self._sum = 0.
        self._num_examples = 0

    def __str__(self):
        mean = self.get()
        return self._display_string.format(mean)


class EnergyStat(MeanStat):
    """
    Class used to measure the mean value of the energy function (when the network is at equilibrium) over the dataset.
    """

    option = None

    def __init__(self, network):
        """Initializes an instance of EnergyStat

        Args:
            network (Network): the network whose energy function is measured at equilibrium
        """

        self._network = network
        display_name = 'Energy'
        name = 'Energy'
        precision = 2
        percentage = False
        display = True

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Energy function"""
        return self._network.energy_fn()


class CostStat(MeanStat):
    """
    Class used to measure the mean cost (when the network is at equilibrium, i.e. the `loss') over the dataset.
    """

    option = None

    def __init__(self, cost_function):
        """Initializes an instance of CostStat

        Args:
            cost_function (CostFunction): the cost function whose value we measure
        """

        self._cost_function = cost_function
        display_name = 'Cost'
        name = 'Cost'
        precision = 5
        percentage = False
        display = True

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Cost function"""
        return self._cost_function.cost_fn()


class ErrorStat(MeanStat):
    """
    Class used to measure the mean error rate over the dataset.
    """

    def __init__(self, cost_function):
        """Initializes an instance of ErrorStat

        Args:
            cost_function (CostFunction): the cost function associated to the error rate
        """

        self.option = None

        self._cost_function = cost_function
        display_name = 'Error'
        name = 'Error'
        precision = 3
        percentage = True
        display = True

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Error function"""
        return self._cost_function.error_fn().type(torch.float)


class TopFiveErrorStat(MeanStat):
    """
    Class used to measure the mean top-5 error rate over the dataset.
    """

    def __init__(self, cost_function):
        """Initializes an instance of TopFiveErrorStat

        Args:
            cost_function (CostFunction): the cost function associated to the top-5 error rate
        """

        self.option = None

        self._cost_function = cost_function
        display_name = 'Top5Error'
        name = 'Top5Error'
        precision = 3
        percentage = True
        display = True

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Top-5 error function"""
        return self._cost_function.top_five_error_fn().type(torch.float)


class ViolationStat(MeanStat):
    """
    Class used to measure the mean violation of the equilibrium condition (at the end of the relaxation phase to equilibrium) over the dataset.
    """

    def __init__(self, layer):
        """Creates and instance of ViolationStat

        Args:
            layer (Layer): the layer whose equilibrium condition's violation we want to track
        """

        self._layer = layer
        self.option = layer.name

        display_name = 'Violation_{}'.format(layer.name)
        name = 'Violation'
        precision = 5
        percentage = False
        display = False

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Violation function"""
        return self._layer.violation()


class SaturationStat(MeanStat):
    """
    Class used to measure the mean saturation over the dataset.

    The `saturation' is the fraction of individual variables that are cliped to the min or the max of the variable's state interval
    """

    def __init__(self, variable):
        """Creates and instance of SaturationStat

        Args:
            variable (Variable): the variable whose saturation we want to track
        """

        self.option = variable.name
        self._variable = variable

        display_name = 'Saturation_{}'.format(variable.name)
        name = 'Saturation'
        precision = 1
        percentage = True
        display = False

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Saturation function"""
        return (self._variable.state == 0.).type(torch.float)


class NormStat(MeanStat):
    """
    Class used to measure the mean norm of a layer over the dataset.
    """

    def __init__(self, variable):
        """Creates and instance of NormStat

        Args:
            variable (Variable): the variable whose norm we want to track
        """

        self.option = variable.name
        self._variable = variable

        display_name = 'Norm_{}'.format(variable.name)
        name = 'Norm'
        precision = 3
        percentage = False
        display = False

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Norm function"""
        return torch.abs(self._variable.state)


class GradientStat(MeanStat):
    """
    Class used to measure the gradient of a given variable.
    """

    def __init__(self, variable):
        """Creates and instance of GradientStat

        Args:
            variable (Variable): the variable whose gradient we want to monitor
        """

        self.option = variable.name
        self._variable = variable

        display_name = 'Gradient_{}'.format(variable.name)
        name = 'Gradient'
        precision = 5
        percentage = False
        display = False

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Gradient function"""
        return torch.abs(self._variable.state.grad)

    def get_variable(self):
        self._variable


class NumIterationsStat(MeanStat):
    """
    Class used to count the mean number of iterations to converge to equilibrium during a relaxation phase
    """
    
    option = None

    def __init__(self, epoch_processor):
        """Creates an instance of NumIterationsStat

        Args:
            epoch_processor (Epoch): the trainer or evaluator
        """

        self._epoch_processor = epoch_processor
        display_name = 'Num_iterations'
        name = 'Num_iterations'
        precision = 1
        percentage = False
        display = True

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Number of Iterations function"""
        return torch.tensor([self._epoch_processor.num_iterations])