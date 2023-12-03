from datetime import datetime
import pickle
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from training.statistics import Counter, ErrorFinder, EnergyStat, CostStat, ErrorStat, TopFiveErrorStat, ViolationStat, NormStat, SaturationStat, GradientStat


class Monitor:
    """
    Class used to monitor the training process.

    Attributes
    ----------
    _network (Network): the network to train
    _trainer (Trainer): used to train the network on the training set
    _scheduler (lr_scheduler): used to adjust the learning rates after every training epoch
    _evaluator (Evaluator): used to evaluate the network on the test set
    _path (str): the directory where to save the model and the characteristics of the training process
    _series (list of TimeSeries): the time series of statistics that are monitored during training
    _test_error_curve (TimeSeries): the test error curve

    Methods
    -------
    run(num_epochs, verbose=False, use_tensorboard=True)
        Trains the network for num_epochs epochs
    save_network()
        Saves the model in the path
    save_series()
        Saves the state of the training process in the path
    """

    def __init__(self, network, cost_function, trainer, scheduler, evaluator, path=None, gdd=None):
        """Creates an instance of Optimizer

        Args:
            network (Network): the network to train
            trainer (Trainer): used to train the network on the training set
            evaluator (Evaluator): used to evaluate the network on the test set
            scheduler (lr_scheduler): used to adjust the learning rates after every training epoch
            path (str, optional): the directory where to save the model and the characteristics of the training process. Default: None
        """

        self._network = network
        self._cost_function = cost_function
        self._trainer = trainer
        self._scheduler = scheduler
        self._evaluator = evaluator
        # self._gdd = gdd  # FIXME

        self._path = datetime.now().strftime("%Y%m%d-%H%M%S") if path is None else path  # If no path is given, uses the date of creation of the monitor

        self._series = []
        self._test_error_curve = None  # initialized in _build_series()

        self._build_series()


    def run(self, num_epochs, verbose=False, use_tensorboard=True):
        """Launch a run for num_epochs epochs

        Logs statistics about the run either after every batch or after every epoch.

        Args:
            num_epochs (int): number of epochs of training
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after every epoch. Default: False
            use_tensorboard (bool, optional): if True, uses a summary writer to monitor with tensorboard. Default: True
        """

        if use_tensorboard: writer = SummaryWriter(self._path)

        start_time = time.time()

        for epoch in range(num_epochs):

            print('Epoch {}'.format(epoch + 1))

            # Training
            self._trainer.run(verbose)
            self._scheduler.step()

            # Evaluation
            self._evaluator.run(verbose)

            # Update the statistics of training and evaluation
            for series in self._series: series.update()
            if use_tensorboard: self._update_summary_writer(writer, epoch)

            if not verbose:  # Print the statistics of training and evaluation at every epoch
                print(str(self._trainer))
                print(str(self._evaluator))
            
            self.save_series()  # saves the training curves
            if self._test_error_curve.is_minimum(): self.save_network()  # saves the network's parameters

            # Print the total duration
            seconds = time.time() - start_time
            minutes, seconds = seconds // 60, seconds % 60
            hours, minutes = minutes // 60, minutes % 60
            print('Duration = {:.0f} hours {:.0f} min {:.0f} sec \n'.format(hours, minutes, seconds))

    def save_network(self):
        """Saves the network's parameters"""

        model_path = self._path + '/model.pt'
        self._network.save(model_path)

    def save_series(self):
        """Saves the time series (`training curves')"""

        time_series_path = self._path + '/time_series.pkl'
        with open(time_series_path, 'wb') as handle:
            dictionary = {series.name: series.get() for series in self._series if series.name}
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _add_statistic(self, statistic, train, list_idx=0):
        """Adds a statistic to either the trainer or the evaluator, and adds the corresponding time series to the monitor

        Special cases:
        * if the statistic is a ChangeStat of a Layer, also adds the corresponding series to _series_layer_changes
        * if the statistic is a test ErrorStat, also sets the corresponding series as _test_error_curve

        Args:
            statistic (Statistic): the statistic to add
            train (bool): whether we add the statistic to the trainer (True) or the evaluator (False)
            list_idx (int, optional): the index of the list that we add the statistic to. Default: 0
        """

        if train: self._trainer.add_statistic(statistic, list_idx)
        else: self._evaluator.add_statistic(statistic)

        series = TimeSeries(statistic, train)
        self._series.append(series)

        if not train and isinstance(statistic, ErrorStat): self._test_error_curve = series

    def _build_series(self):
        """Prepares the statistics and the corresponding time series to monitor"""

        network = self._network
        cost_function = self._cost_function
        train_set_size = self._trainer.dataset_size()
        test_set_size = self._evaluator.dataset_size()

        # the statistics to add to the trainer
        stats_train_0 = [
        Counter(network, train_set_size),
        EnergyStat(network),
        CostStat(cost_function),
        ErrorStat(cost_function),
        TopFiveErrorStat(cost_function),
        ]
        stats_train_0 += [NormStat(layer) for layer in network.layers()]
        stats_train_0 += [SaturationStat(layer) for layer in network.layers()]
        # stats_train_1 = [GradientStat(layer) for layer in network.layers()]  # FIXME
        stats_train_1 = [GradientStat(param) for param in network.params()]

        for stat in stats_train_0: self._add_statistic(stat, train=True)
        for stat in stats_train_1: self._add_statistic(stat, train=True, list_idx=1)

        # the statistics to add to the evaluator
        stats_test = [
        Counter(network, test_set_size),
        EnergyStat(network),
        CostStat(cost_function),
        ErrorStat(cost_function),
        TopFiveErrorStat(cost_function),
        # ErrorFinder(network, evaluator),
        ]
        stats_test += [NormStat(layer) for layer in network.layers()]
        stats_test += [SaturationStat(layer) for layer in network.layers()]

        for stat in stats_test: self._add_statistic(stat, train=False)

    def _update_summary_writer(self, writer, epoch):
        """Add the statistics to the summary writer to monitor with tensorboard

        Args:
            writer (SummaryWriter): tensorboard summary writer to update
            epoch (int): epoch of training where the statistics have been recorded
        """

        for param in self._network.params(): writer.add_histogram(param.name, param.state, epoch+1)

        for series in self._series:
            if series.name: writer.add_scalar(series.name, series.last_value(), epoch+1)

        # FIXME
        # figs = self._gdd.produce_curves()
        # for name, fig in figs.items(): writer.add_image('GDD {}'.format(name), fig, epoch+1)



class TimeSeries:
    """Class for time series of statistics during training

    Attributes
    ----------
    name (str): the name of the time series
    _statistic (Statistic): the underlying statistic whose value is measured regularly to define the time series
    _time_series (list): the time series. Each entry corresponds to the value of the statistic at a given epoch

    Methods
    -------
    update()
        Appends the current value of the statistic to the time series
    get()
        Returns the time series of statistics
    last_value()
        Returns the last value of the time series of statistics
    minimum()
        Returns the minimum of the time series
    is_minimum()
        Checks if the last value of the time series is strictly less than all the previous values
    """

    def __init__(self, statistic, train=True):
        """Creates an instance of Series

        Args:
            statistic (Statistic): the underlying statistic whose time series we compute during training
            train (bool, optional): whether this is a time series during training (training time) or evaluation (test time). Default: True
        """

        self._statistic = statistic
        self._time_series = []

        self._name = None
        if self._statistic.display_name:
            name = self._statistic.name
            if train: name += '/train'
            else: name += '/test'
            if self._statistic.option: name += '_'+self._statistic.option
            self._name = name

    @property
    def name(self):
        """Gets the name of the time series"""
        return self._name

    def update(self):
        """Appends the current value of the statistic to the time series"""
        self._time_series.append(self._statistic.get())

    def get(self):
        """Returns the time series"""
        return self._time_series

    def last_value(self):
        """Returns the last value of the time series"""
        return self._time_series[-1]

    def minimum(self):
        """Returns the minimum of the time series"""
        return min(self._time_series)

    def is_minimum(self):
        """Checks if the last value of the time series is strictly less than all the previous values

        Returns:
            bool: whether or not the last value of the time series is the strict minimum of the series
        """

        epoch = len(self._time_series)
        if epoch <= 1: return True

        last_value = self._time_series[-1]
        min_value = min(self._time_series[:-1])
        return last_value < min_value


class Optimizer(torch.optim.SGD):

    def __init__(self, network, cost_function, learning_rates, momentum=0, weight_decay=0):

        self._learning_rates = learning_rates
        self._momentum = momentum
        self._weight_decay = weight_decay

        params = network.params() + cost_function.params()
        params = [{"params": param.state, "lr": lr} for param, lr in zip(params, learning_rates)]
        torch.optim.SGD.__init__(self, params, lr=0.1, momentum=momentum, weight_decay=weight_decay)

    def __str__(self):
        return 'SGD -- initial learning rates = {}, momentum={}, weight_decay={}'.format(self._learning_rates, self._momentum, self._weight_decay)