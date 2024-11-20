import argparse
import numpy
import torch

from datasets import load_dataloaders
from model.hopfield.network import DeepHopfieldEnergy
from model.function.network import Network
from model.function.cost import SquaredError
from model.hopfield.minimizer import FixedPointMinimizer
from training.sgd import EquilibriumProp, Backprop, AugmentedFunction
from training.epoch import Trainer, Evaluator
from training.monitor import Monitor, Optimizer

parser = argparse.ArgumentParser(description='Deep Hopfield Networks')
parser.add_argument('--model', type = str, default = 'dhn-1h', help="The DHN architecture: either 'dhn-1h', 'dhn-2h' or 'dhn-3h'")

args = parser.parse_args()



if __name__ == "__main__":
    
    # Expected results
    # DHN-1H: 1.79% test error
    # DHN-2H: 1.65% test error
    # DHN-3H: 1.65% test error

    model = args.model

    # Hyperparameters
    if model == 'dhn-1h':
        batch_size = 16
        layer_shapes = [(1, 28, 28), (1024,), (10,)]
        weight_gains = [0.7, 0.7]
        num_iterations_inference = 15
        num_iterations_training = 15
        nudging = 0.2
        learning_rates_weights = [0.05, 0.05]
        learning_rates_biases = [0.05, 0.05]
        num_epochs = 100
    elif model == 'dhn-2h':
        batch_size = 16
        layer_shapes = [(1, 28, 28), (1024,), (1024,), (10,)]
        weight_gains = [0.7, 0.7, 0.7]
        num_iterations_inference = 50
        num_iterations_training = 20
        nudging = 0.2
        learning_rates_weights = [0.2, 0.05, 0.01]
        learning_rates_biases = [0.2, 0.05, 0.01]
        num_epochs = 100
    elif model == 'dhn-3h':
        batch_size = 16
        layer_shapes = [(1, 28, 28), (1024,), (1024,), (1024,), (10,)]
        weight_gains = [0.7, 0.7, 0.7, 0.7]
        num_iterations_inference = 100
        num_iterations_training = 10
        nudging = 0.2
        learning_rates_weights = [0.2, 0.05, 0.01, 0.002]
        learning_rates_biases = [0.2, 0.05, 0.01, 0.002]
        num_epochs = 100
    else:
        raise ValueError("expected 'dhn-1h', 'dhn-2h' or 'dhn-3h' but got {}".format(model))

    # Load the training and test data (MNIST)
    dataset = 'MNIST'
    training_loader, test_loader = load_dataloaders(dataset, batch_size, augment_32x32=False, normalize=False)

    # Build the network (DHN: deep Hopfield network)
    energy_fn = DeepHopfieldEnergy(layer_shapes, weight_gains)

    # Set the device on which we run and train the network
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    energy_fn.set_device(device)
    
    # Define the cost function (mean squared error)
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredError(output_layer)
    
    network = Network(energy_fn)

    # Define the energy minimizer and gradient estimator (equilibrium propagation)
    params = energy_fn.params()
    layers = energy_fn.layers()
    free_layers = network.free_layers()

    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    energy_minimizer_training = FixedPointMinimizer(augmented_fn, free_layers)
    estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training)
    estimator.nudging = nudging
    estimator.variant = 'centered'

    energy_minimizer_training.num_iterations = num_iterations_training
    energy_minimizer_training.mode = 'asynchronous'

    # Build the optimizer (SGD)
    learning_rates = learning_rates_biases + learning_rates_weights
    momentum = 0.
    weight_decay = 0. * 1e-4
    optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

    # Define the trainer (to perform one epoch of training) and the evaluator (to evaluate the model on the test set)
    energy_minimizer_inference = FixedPointMinimizer(energy_fn, free_layers)
    energy_minimizer_inference.num_iterations = num_iterations_inference
    energy_minimizer_inference.mode = 'asynchronous'

    trainer = Trainer(network, cost_fn, training_loader, estimator, optimizer, energy_minimizer_inference)
    evaluator = Evaluator(network, cost_fn, test_loader, energy_minimizer_inference)
    
    # Define the scheduler for the learning rates
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Define the path and the monitor to perform the run
    path = '/'.join(['papers/fast-drn', model, 'EP'])
    monitor = Monitor(energy_fn, cost_fn, trainer, scheduler, evaluator, path)

    # Print the characteristics of the run
    print('Dataset: {} -- batch_size={}'.format(dataset, batch_size))
    print('Network: ', energy_fn)
    print('Cost function: ', cost_fn)
    print('Energy minimizer during inference: ', energy_minimizer_inference)
    print('Energy minimizer during training: ', energy_minimizer_training)
    print('Gradient estimator: ', estimator)
    print('Parameter optimizer: ', optimizer)
    print('Number of epochs = {}'.format(num_epochs))
    print('Path = {}'.format(path))
    print('Device = {}'.format(device))
    print()

    # Launch the experiment
    monitor.run(num_epochs, verbose=True)