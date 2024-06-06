import argparse
import numpy
import torch

from datasets import load_dataloaders
from model.resistive.network import DeepResistiveEnergy
from model.function.network import Network
from model.function.cost import SquaredError
from model.resistive.minimizer import QuadraticMinimizer
from training.sgd import EquilibriumProp, Backprop, AugmentedFunction
from training.epoch import Trainer, Evaluator
from training.monitor import Monitor, Optimizer

parser = argparse.ArgumentParser(description='Deep Resistive Networks')
parser.add_argument('--model', type = str, default = 'drn-1h', help="The DRN architecture: either 'drn-1h', 'drn-2h', 'drn-3h', 'drn-xs' or 'drn-xl'")
parser.add_argument('--algorithm', type = str, default = 'EP', help="The algorithm used to train the network: equilibrium propagation (EP) or backpropagation (BP)")

args = parser.parse_args()



if __name__ == "__main__":
    
    # Expected results
    # DRN-XS EP: 3.46% test error and 2.94% train error (0 hours 30 minutes)
    # DRN-XS BP: 3.30% test error and 2.62% train error (0 hours 32 minutes)
    # DRN-XL EP: 1.33% test error and 0.02% train error (10 hours 27 minutes)
    # DRN-XL BP: 1.30% test error and 0.01% train error (8 hours 19 minutes)
    # DRN-1H EP: 1.57% test error and 0.12% train error (2 hours 36 minutes)
    # DRN-1H BP: 1.54% test error and 0.12% train error (2 hours 48 minutes)
    # DRN-2H EP: 1.48% test error and 0.25% train error (4 hours 29 minutes)
    # DRN-2H BP: 1.45% test error and 0.22% train error (4 hours 53 minutes)
    # DRN-3H EP: 1.66% test error and 0.46% train error (6 hours 57 minutes)
    # DRN-3H BP: 1.50% test error and 0.33% train error (7 hours 55 minutes)

    model = args.model
    algorithm = args.algorithm

    # Hyperparameters
    if model == 'drn-xs':
        layer_shapes = [(2, 28, 28), (100,), (10,)]
        weight_gains = [1.0, 1.0]
        input_gain = 100.
        num_iterations_inference = 4
        num_iterations_training = 4
        nudging = 1.0
        learning_rates_weights = [0.006, 0.006]
        learning_rates_biases = [0.006, 0.006]
        num_epochs = 10
    elif model == 'drn-xl':
        layer_shapes = [(2, 28, 28), (32768,), (10,)]
        weight_gains = [1.0, 1.0]
        input_gain = 800.
        num_iterations_inference = 4
        num_iterations_training = 4
        nudging = 1.0
        learning_rates_weights = [0.006, 0.006]
        learning_rates_biases = [0.006, 0.006]
        num_epochs = 100
    elif model == 'drn-1h':
        layer_shapes = [(2, 28, 28), (1024,), (10,)]
        weight_gains = [1.0, 1.0]
        input_gain = 480.
        num_iterations_inference = 4
        num_iterations_training = 4
        nudging = 1.0
        learning_rates_weights = [0.006, 0.006]
        learning_rates_biases = [0.006, 0.006]
        num_epochs = 50
    elif model == 'drn-2h':
        layer_shapes = [(2, 28, 28), (1024,), (1024,), (10,)]
        weight_gains = [1.0, 1.0, 1.0]
        input_gain = 2000.
        num_iterations_inference = 5
        num_iterations_training = 5
        nudging = 1.0
        learning_rates_weights = [0.002, 0.006, 0.018]
        learning_rates_biases = [0.002, 0.006, 0.018]
        num_epochs = 50
    elif model == 'drn-3h':
        layer_shapes = [(2, 28, 28), (1024,), (1024,), (1024,), (10,)]
        weight_gains = [1.0, 1.0, 1.0, 1.0]
        input_gain = 4000.
        num_iterations_inference = 6
        num_iterations_training = 6
        nudging = 2.0
        learning_rates_weights = [0.005, 0.02, 0.08, 0.005]
        learning_rates_biases = [0.0005, 0.002, 0.008, 0.0005]
        num_epochs = 50
    else:
        raise ValueError("expected 'drn-1h', 'drn-2h', 'drn-3h' or 'drn-xl' but got {}".format(model))

    # Load the training and test data (MNIST)
    dataset = 'MNIST'
    batch_size = 4
    training_loader, test_loader = load_dataloaders(dataset, batch_size, augment_32x32=False, normalize=False)

    # Build the network (DRN: deep resistive network)
    energy_fn = DeepResistiveEnergy(layer_shapes, weight_gains, input_gain)

    # Set the device on which we run and train the network
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    energy_fn.set_device(device)
    
    # Define the cost function (mean squared error)
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredError(output_layer)
    
    network = Network(energy_fn)

    # Define the energy minimizer and gradient estimator (equilibrium propagation, or backpropagation)
    params = energy_fn.params()
    layers = energy_fn.layers()
    free_layers = network.free_layers()

    if algorithm == "EP":
        augmented_fn = AugmentedFunction(energy_fn, cost_fn)
        energy_minimizer_training = QuadraticMinimizer(augmented_fn, free_layers)
        estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training)
        estimator.nudging = nudging
        estimator.variant = 'centered'
    elif algorithm == "BP":
        energy_minimizer_training = QuadraticMinimizer(energy_fn, free_layers)
        estimator = Backprop(params, layers, cost_fn, energy_minimizer_training)
    else:
        raise ValueError("expected 'EP' or 'BP' but got {}".format(algorithm))

    energy_minimizer_training.num_iterations = num_iterations_training
    energy_minimizer_training.mode = 'asynchronous'

    # Build the optimizer (SGD)
    learning_rates = learning_rates_biases + learning_rates_weights
    momentum = 0.
    weight_decay = 0.
    optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

    # Define the trainer (to perform one epoch of training) and the evaluator (to evaluate the model on the test set)
    energy_minimizer_inference = QuadraticMinimizer(energy_fn, free_layers)
    energy_minimizer_inference.num_iterations = num_iterations_inference
    energy_minimizer_inference.mode = 'asynchronous'

    trainer = Trainer(network, cost_fn, params, training_loader, estimator, optimizer, energy_minimizer_inference)
    evaluator = Evaluator(network, cost_fn, test_loader, energy_minimizer_inference)
    
    # Define the scheduler for the learning rates
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Define the path and the monitor to perform the run
    path = '/'.join(['papers/fast-drn', model, algorithm])
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