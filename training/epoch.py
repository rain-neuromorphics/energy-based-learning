import sys
import torch



class Epoch:
    """
    Class used to process a dataset with a network once (one 'epoch') to change the internal state of the network and/or compute statistics.
    Two important subclasses of Epoch are Evaluator and Trainer.

    Attributes
    ----------
    _stats (list of list of Statistic): the list of all lists of statistics to be computed over the dataset

    Methods
    -------
    add_statistic(stat, list_idx)
        Adds a statistic to the collection of index list_idx
    dataset_size()
        Returns the size of the dataset
    _reset()
        Sets all the statistics to zero
    _do_measurements(list_idx)
        Do the measurements for each of the statistics in list of index list_idx
    """

    def __init__(self, num_lists):
        """Creates an instance of Epoch

        Args:
            num_lists (int): the number of lists of statistics
        """

        self._stats = []
        for _ in range(num_lists): self._stats.append([])

    def add_statistic(self, stat, list_idx=0):
        """Adds a statistic to the list of statistics

        Args:
            stat (Statistic): the statistic to be added
            list_idx (int, optional): the index of the list in which we add the statistic. Default: 0
        """

        self._stats[list_idx].append(stat)

    def dataset_size(self):
        """Returns the size of the dataset"""
        return len(self._dataloader.dataset)

    def _all_stats(self):
        """Returns the list of all stats, both from the 'eval' and 'train' collections"""
        return [stat for list_stats in self._stats for stat in list_stats]

    def _reset(self):
        """Sets all the statistics to zero"""

        for stat in self._all_stats(): stat.reset()

    def _do_measurements(self, list_idx=0):
        """Do measurements in all stats of the collection of index list_idx

        Args:
            list_idx (int, optional): the index of the list where we measure all the statistic. Default: 0
        """

        for stat in self._stats[list_idx]: stat.do_measurement()

    def __str__(self):
        list_of_strings = [str(stat) for stat in self._all_stats() if stat.display]
        string = ', '.join(list_of_strings)
                
        return string



class Evaluator(Epoch):
    """
    Class used to evaluate a network on a dataset

    Attributes
    ----------
    _network (SumSeparableFunction): the model to evaluate
    _dataloader (Dataloader): the dataset on which to evaluate the model
    energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference

    idx (tensor of int): vector of indices of the data examples in the current mini-batch

    Methods
    -------
    run(verbose)
        Evaluates the network over the dataset
    """

    def __init__(self, network, cost_fn, dataloader, energy_minimizer):
        """Initializes an instance of Evaluator

        Args:
            network (Network): the model to evaluate
            cost_fn (CostFunction): the cost function to optimize
            dataloader (Dataloader): the dataset on which to evaluate the model. An IndexedDataset that loads data in the form of triplets (x, y, idx)
            energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference
        """

        Epoch.__init__(self, 1)

        self._network = network
        self._cost_fn = cost_fn
        self._dataloader = dataloader

        self._energy_minimizer = energy_minimizer

    @property
    def idx(self):
        """Gets the indices of the examples in the last mini batch processed"""
        return self._idx

    def run(self, verbose=False):
        """Evaluate the model over the dataset.

        Args:
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after processing the entire dataset. Default: False.
        """

        self._reset()  # sets all the statistics to zero

        for x, y, idx in self._dataloader:

            # Inference (free phase relaxation)
            self._network.set_input(x, reset=True)
            self._energy_minimizer.compute_equilibrium()

            # Measure statistics
            self._idx = idx
            self._cost_fn.set_target(y)
            self._do_measurements()

            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write(str(self))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.write(str(self))
            sys.stdout.write('\n')

    def __str__(self):
        return 'TEST  -- ' + Epoch.__str__(self)



class Trainer(Epoch):
    """
    Class used to train a network on a dataset

    Attributes
    ----------
    _network (SumSeparableFunction): the network to train
    _dataloader (Dataloader): the dataset on which to train the network
    _differentiator (GradientEstimator): the method used to train the network
    energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference

    _stats (list of Statistic): _stats[0] is the list of statistics to measure after inference (evaluation), and _stats[1] is the list of statistics to measure after computing the gradient (training)

    Methods
    -------
    run(verbose)
        Train the network for one epoch over the dataset
    """

    def __init__(self, network, cost_fn, dataloader, differentiator, optimizer, energy_minimizer):
        """Initializes an instance of Trainer

        Args:
            network (Network): the network to train
            cost_fn (CostFunction): the cost function to optimize
            dataloader (Dataloader): the dataset on which to train the network
            differentiator (GradientEstimator): either EquilibriumProp or Backprop
            optimizer (str): the optimizer used to optimize.
            energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference
        """


        Epoch.__init__(self, 2)

        self._network = network
        self._params = network._params + cost_fn.params()
        self._cost_fn = cost_fn
        self._dataloader = dataloader
        self._differentiator = differentiator
        self._optimizer = optimizer
        self._energy_minimizer = energy_minimizer

    def run(self, verbose=False):
        """Train the model for one epoch over the dataset.

        Args:
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after every epoch. Default: False.
        """

        self._reset()  # sets all the statistics to zero

        for x, y in self._dataloader:

            # inference (free phase relaxation)
            self._network.set_input(x, reset=False)  # we set the input, and we let the state of the network where it was at the end of the previous batch
            self._energy_minimizer.compute_equilibrium()  # we let the network settle to equilibrium (free state)
            self._cost_fn.set_target(y)  # we present the correct (desired) output
            self._do_measurements(0)  # we measure the statistics of the free state (energy value, cost value, error value, ...)

            # training step
            grads = self._differentiator.compute_gradient()  # compute the parameter gradients
            for param, grad in zip(self._params, grads): param.state.grad = grad  # Set the gradients of the parameters
            self._do_measurements(1)  # measure the statistics of training
            self._optimizer.step()  # perform one step of gradient descent on the parameters (of both the energy function E and the cost function C)
            for param in self._params: param.clamp_()  # clamp the parameters' states in their range of permissible values, if adequate

            if verbose:  # log the characteristics of training for the current epoch, up to the current mini-batch
                sys.stdout.write('\r')
                sys.stdout.write(str(self))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.write(str(self))
            sys.stdout.write('\n')

    def __str__(self):
        return 'TRAIN -- ' + Epoch.__str__(self)