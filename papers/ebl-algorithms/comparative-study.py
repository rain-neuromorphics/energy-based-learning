import argparse
import torch

from datasets import load_dataloaders
from model.hopfield.network import ConvHopfieldEnergy32
from model.function.network import Network
from model.function.cost import SquaredError
from model.hopfield.minimizer import FixedPointMinimizer
from training.sgd import EquilibriumProp, ContrastiveLearning, Backprop, RecurrentBackprop, AugmentedFunction
from training.epoch import Trainer, Evaluator
from training.monitor import Monitor, Optimizer

parser = argparse.ArgumentParser(description='Comparative study of energy-based learning algorithms')
parser.add_argument('--dataset', type = str, default = 'MNIST', help="The dataset used. Either `MNIST', `FashionMNIST', `SVHN', `CIFAR10' or `CIFAR100'")
parser.add_argument('--algorithm', type = str, default = 'CEP', help="The algorithm used to train the network. Either 'CL', 'PEP', 'NEP', 'CEP', 'PCpL', 'NCpL', 'CCpL', 'TBP' or 'RBP'")
parser.add_argument('--gain', type = float, default = 0.5, help="The gain for the weights of the network, at initialization")
parser.add_argument('--device', type = str, default = None, help="The device where we run and train the network")

args = parser.parse_args()


def build_gradient_estimator(energy_fn, cost_fn, algorithm = 'CEP', nudging = 0.25, num_iterations = 15):
    """Build the gradient estimator: either contrastive learning (CL), equilibrium propagation (EP), coupled learning (CpL) or a variant of these algorithms.

    Variants:
    * P = postively-perturbed
    * N = negatively-perturbed
    * C = centered
    e.g. CEP = centered variant of equilibrium propagation

    TBP = truncated backpropagation
    RBP = recurrent backpropagation

    Args:
        energy_fn (SumSeparableFunction): the energy_fn to train
        cost_fn (CostFunction): the cost function to optimize
        algorithm (str, optional): the gradient estimator used. Either 'CL', 'PEP', 'NEP', 'CEP', 'PCpL', 'NCpL', 'CCpL', 'TBP' or 'RBP'. Default: 'CEP'
        nudging (float, optional): the nudging value used, for equilibrium propagation (EP) and coupled learning (CpL). Default: 0.25
        num_iterations (int, optional): the number of iterations used for training. Default: 15
    
    Returns:
        the gradient estimator
    """

    params = energy_fn.params()
    layers = energy_fn.layers()
    free_layers = network.free_layers()

    if algorithm == "CL":
        energy_minimizer = FixedPointMinimizer(energy_fn, free_layers[:-1])
        energy_minimizer.mode = 'asynchronous'
        estimator = ContrastiveLearning(params, layers, energy_fn, cost_fn, energy_minimizer)
        estimator.nudging = 1.0
        estimator.variant = 'positive'
    elif algorithm[1:] == "EP":
        augmented_fn = AugmentedFunction(energy_fn, cost_fn)
        energy_minimizer = FixedPointMinimizer(augmented_fn, free_layers)
        energy_minimizer.mode = 'asynchronous'
        estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer)  # FIXME
        estimator.nudging = nudging
        if algorithm == "PEP": estimator.variant = 'positive'
        elif algorithm == "NEP": estimator.variant = 'negative'
        elif algorithm == "CEP": estimator.variant = 'centered'
        else: raise ValueError("expected 'PEP', 'NEP' or 'CEP' but got {}".format(algorithm))
    elif algorithm[1:] == "CpL":
        energy_minimizer = FixedPointMinimizer(energy_fn, free_layers[:-1])
        energy_minimizer.mode = 'asynchronous'
        estimator = ContrastiveLearning(params, layers, energy_fn, cost_fn, energy_minimizer)
        estimator.nudging = nudging
        if algorithm == "PCpL": estimator.variant = 'positive'
        elif algorithm == "NCpL": estimator.variant = 'negative'
        elif algorithm == "CCpL": estimator.variant = 'centered'
        else: raise ValueError("expected 'PCpL', 'NCpL' or 'CCpL' but got {}".format(algorithm))
    elif algorithm == "TBP":
        energy_minimizer = FixedPointMinimizer(energy_fn, free_layers)
        energy_minimizer.mode = 'asynchronous'
        estimator = Backprop(params, layers, cost_fn, energy_minimizer)
    elif algorithm == "RBP":
        energy_minimizer = FixedPointMinimizer(energy_fn, free_layers)
        energy_minimizer.mode = 'synchronous'
        estimator = RecurrentBackprop(params, layers, cost_fn, energy_minimizer)
    else:
        raise ValueError("expected 'CL', 'PEP', 'NEP', 'CEP', 'PCpL', 'NCpL', 'CCpL', 'TBP' or 'RBP' but got {}".format(algorithm))

    energy_minimizer.num_iterations = num_iterations

    return energy_minimizer, estimator


if __name__ == "__main__":

    dataset = args.dataset
    algorithm = args.algorithm

    # Load the training and test data (either MNIST, Fashion-MNIST, SVHN, CIFAR-10 or CIFAR-100)
    batch_size = 128
    training_loader, test_loader = load_dataloaders(dataset, batch_size, augment_32x32=True)  # if the dataset is MNIST or Fashion-MNIST, we augment the input images to 32x32 pixels

    # Specify the number of input filters and output units of the network, depending on the dataset
    if dataset in ['MNIST', 'FashionMNIST']: num_inputs = 1
    else: num_inputs = 3

    if dataset == 'CIFAR100': num_outputs = 100
    else: num_outputs = 10

    # Build the network (a convolutional Hopfield net with 32x32-pixel input images)
    gain = args.gain
    weight_gains = [gain, gain, gain, gain, gain]
    energy_fn = ConvHopfieldEnergy32(num_inputs, num_outputs, weight_gains=weight_gains)

    # Set the device on which we run and train the network
    if args.device != None: device = args.device
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    energy_fn.set_device(device)
    
    # Define the cost function (mean squared error)
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredError(output_layer)
    network = Network(energy_fn)

    # Define the gradient estimator (either CL, EP, CpL, or a variant of these algorithms)
    nudging = 0.25
    num_iterations_training = 15
    energy_minimizer_training, estimator = build_gradient_estimator(energy_fn, cost_fn, algorithm, nudging, num_iterations_training)

    # Build the optimizer (SGD with momentum and weight decay)
    learning_rates_weights = [0.0625, 0.0375, 0.025, 0.02, 0.0125]
    learning_rates_biases = [0.0625, 0.0375, 0.025, 0.02, 0.0125]
    learning_rates = learning_rates_biases + learning_rates_weights
    momentum = 0.9
    weight_decay = 3 * 1e-4
    # optimizer = build_optimizer(energy_fn, learning_rates, momentum, weight_decay)
    optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

    # Define the trainer (to perform one epoch of training) and the evaluator (to evaluate the model on the test set)
    free_layers = network.free_layers()
    num_iterations_inference = 60
    energy_minimizer_inference = FixedPointMinimizer(energy_fn, free_layers)
    energy_minimizer_inference.num_iterations = num_iterations_inference
    trainer = Trainer(network, cost_fn, training_loader, estimator, optimizer, energy_minimizer_inference)
    evaluator = Evaluator(network, cost_fn, test_loader, energy_minimizer_inference)
    
    # Define the scheduler for the learning rates (cosine annealing scheduler)
    num_epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=2.*1e-6)

    # Define the path and the monitor to perform the run
    path = '/'.join(['papers/ebl-algorithms/comparison', dataset, algorithm])
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