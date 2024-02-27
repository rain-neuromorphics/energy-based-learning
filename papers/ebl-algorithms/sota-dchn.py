import argparse
import torch

from datasets import load_dataloaders
from model.hopfield.network import ConvHopfieldEnergy32
from model.function.network import Network
from model.function.cost import SquaredError
from model.hopfield.minimizer import FixedPointMinimizer
from training.sgd import EquilibriumProp, AugmentedFunction
from training.epoch import Trainer, Evaluator
from training.monitor import Monitor, Optimizer

parser = argparse.ArgumentParser(description='Best results obtained with a deep convolutional Hopfield network')
parser.add_argument('--dataset', type = str, default = 'CIFAR10', help="The dataset used. Either `CIFAR10' or `CIFAR100'")
parser.add_argument('--epochs', type = int, default = 100, help="The number of epochs of training")

args = parser.parse_args()


if __name__ == "__main__":

    dataset = args.dataset

    # The best hyperparameters found, depending on the dataset (CIFAR10 or CIFAR100)
    if dataset == 'CIFAR10':
        num_outputs = 10
        nudging = 0.1
        num_iterations_inference = 60
        num_iterations_training = 20
        weight_gains = [0.4, 0.7, 0.6, 0.3, 0.4]
        learning_rates_weights = [0.03, 0.03, 0.03, 0.03, 0.03]
        learning_rates_biases = [0.03, 0.03, 0.03, 0.03, 0.03]
        momentum = 0.9
        weight_decay = 2.5 * 1e-4
    elif dataset == 'CIFAR100':
        num_outputs = 100
        nudging = 0.25
        num_iterations_inference = 60
        num_iterations_training = 15
        weight_gains = [0.5, 0.4, 0.5, 0.8, 0.5]
        learning_rates_weights = [0.03, 0.04, 0.04, 0.04, 0.025]
        learning_rates_biases = [0.03, 0.04, 0.04, 0.04, 0.025]
        momentum = 0.9
        weight_decay = 3.5 * 1e-4
    else:
        raise ValueError("expected 'CIFAR10' or 'CIFAR100' but got {}".format(dataset))

    # Load the training and test data (either CIFAR-10 or CIFAR-100)
    batch_size = 128
    training_loader, test_loader = load_dataloaders(dataset, batch_size)

    # Build the network: a convolutional Hopfield network (DCHN) with 32x32-pixel input images
    num_inputs = 3
    energy_fn = ConvHopfieldEnergy32(num_inputs, num_outputs, weight_gains=weight_gains)

    # Set the device on which we run and train the network
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    energy_fn.set_device(device)
    
    # Define the cost function: mean squared error (MSE)
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredError(output_layer)
    network = Network(energy_fn)

    # Define the energy minimizer used in the perturbed phase of training
    params = energy_fn.params()
    layers = energy_fn.layers()
    free_layers = network.free_layers()
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    energy_minimizer_training = FixedPointMinimizer(augmented_fn, free_layers)
    energy_minimizer_training.num_iterations = num_iterations_training

    # Define the gradient estimator: centered equilibrium propagation (CEP)
    estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training)
    estimator.variant = 'centered'
    estimator.nudging = nudging

    # Build the optimizer (SGD with momentum and weight decay)
    learning_rates = learning_rates_biases + learning_rates_weights
    optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

    # Define the energy minimizer used at inference (fixed point minimizer)
    energy_minimizer_inference = FixedPointMinimizer(energy_fn, free_layers)
    energy_minimizer_inference.num_iterations = num_iterations_inference

    # Define the trainer (to perform one epoch of training) and the evaluator (to evaluate the model on the test set)
    trainer = Trainer(network, cost_fn, params, training_loader, estimator, optimizer, energy_minimizer_inference)
    evaluator = Evaluator(network, cost_fn, test_loader, energy_minimizer_inference)
    
    # Define the scheduler for the learning rates (cosine annealing scheduler)
    num_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=2.*1e-6)

    # Define the path and the monitor to perform the run
    path = '/'.join(['papers/ebl-algorithms/sota-dchn', dataset])
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