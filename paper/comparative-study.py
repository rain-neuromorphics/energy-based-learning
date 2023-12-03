import argparse
import torch

from datasets import load_dataloaders
from network.hopfield import ConvHopfieldNet32
from network.cost import SquaredError
from training.sgd import Eqprop, AutoDiff, ImplicitDiff, ContrastiveLearning
from training.epoch import Trainer, Evaluator
from training.monitor import Monitor, Optimizer

parser = argparse.ArgumentParser(description='Comparative study of energy-based learning algorithms')
parser.add_argument('--dataset', type = str, default = 'MNIST', help="The dataset used. Either `MNIST', `FashionMNIST', `SVHN', `CIFAR10' or `CIFAR100'")
parser.add_argument('--algorithm', type = str, default = 'CEP', help="The algorithm used to train the network. Either 'CL', 'PEP', 'NEP', 'CEP', 'PCpL', 'NCpL', 'CCpL', 'TBP' or 'RBP'")
parser.add_argument('--gain', type = float, default = 0.5, help="The gain for the weights of the network, at initialization")
parser.add_argument('--device', type = str, default = None, help="The device where we run and train the network")

args = parser.parse_args()


def build_gradient_estimator(network, cost_function, algorithm = 'CEP', nudging = 0.25, num_iterations = 15):
    """Build the gradient estimator: either contrastive learning (CL), equilibrium propagation (EP), coupled learning (CpL) or a variant of these algorithms.

    Variants:
    * P = postively-perturbed
    * N = negatively-perturbed
    * C = centered
    e.g. CEP = centered variant of equilibrium propagation

    TBP = truncated backpropagation
    RBP = recurrent backpropagation

    Args:
        network (Network): the network to train
        cost_function (CostFunction): the cost function to optimize
        algorithm (str, optional): the gradient estimator used. Either 'CL', 'PEP', 'NEP', 'CEP', 'PCpL', 'NCpL', 'CCpL', 'TBP' or 'RBP'. Default: 'CEP'
        nudging (float, optional): the nudging value used, for equilibrium propagation (EP) and coupled learning (CpL). Default: 0.25
        num_iterations (int, optional): the number of iterations used for training. Default: 15
    
    Returns:
        the gradient estimator
    """

    if algorithm == "CL":
        estimator = ContrastiveLearning(network, cost_function)
    elif algorithm[1:] == "EP":
        estimator = Eqprop(network, cost_function)
        estimator.nudging = nudging
        if algorithm == "PEP": estimator.training_mode = 'positive'
        elif algorithm == "NEP": estimator.training_mode = 'negative'
        elif algorithm == "CEP": estimator.training_mode = 'centered'
        else: raise ValueError("expected 'PEP', 'NEP' or 'CEP' but got {}".format(algorithm))
    elif algorithm[1:] == "CpL":
        estimator = ContrastiveLearning(network, cost_function)
        estimator.nudging = nudging
        if algorithm == "PCpL": estimator.training_mode = 'positive'
        elif algorithm == "NCpL": estimator.training_mode = 'negative'
        elif algorithm == "CCpL": estimator.training_mode = 'centered'
        else: raise ValueError("expected 'PCpL', 'NCpL' or 'CCpL' but got {}".format(algorithm))
    elif algorithm == "TBP":
        estimator = AutoDiff(network, cost_function)
    elif algorithm == "RBP":
        estimator = ImplicitDiff(network, cost_function)
    else:
        raise ValueError("expected 'CL', 'PEP', 'NEP', 'CEP', 'PCpL', 'NCpL', 'CCpL', 'TBP' or 'RBP' but got {}".format(algorithm))

    estimator.max_iterations = num_iterations

    return estimator


if __name__ == "__main__":

    dataset = args.dataset
    algorithm = args.algorithm

    # Load the training and test data (either MNIST, FashionMNIST, SVHN, CIFAR-10 or CIFAR-100)
    batch_size = 128
    training_loader, test_loader = load_dataloaders(dataset, batch_size, augment_32x32=True)  # if the dataset is MNIST or FashionMNIST, we augment the input images to 32x32 pixels

    # Specify the number of input filters and output units of the network, depending on the dataset
    if dataset in ['MNIST', 'FashionMNIST']: num_inputs = 1
    else: num_inputs = 3

    if dataset == 'CIFAR100': num_outputs = 100
    else: num_outputs = 10

    # Build the network (a convolutional Hopfield net with 32x32-pixel input images)
    gain = args.gain
    weight_gains = [gain, gain, gain, gain, gain]
    network = ConvHopfieldNet32(num_inputs, num_outputs, weight_gains=weight_gains)

    # Set the device on which we run and train the network
    if args.device != None: device = args.device
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    network.set_device(device)
    
    # Define the cost function (mean squared error)
    cost_function = SquaredError(network)

    # Define the gradient estimator (either CL, EP, CpL, or a variant of these algorithms)
    nudging = 0.25
    num_iterations_training = 15
    estimator = build_gradient_estimator(network, cost_function, algorithm, nudging, num_iterations_training)

    # Build the optimizer (SGD with momentum and weight decay)
    learning_rates_weights = [0.0625, 0.0375, 0.025, 0.02, 0.0125]
    learning_rates_biases = [0.0625, 0.0375, 0.025, 0.02, 0.0125]
    learning_rates = learning_rates_biases + learning_rates_weights
    momentum = 0.9
    weight_decay = 3 * 1e-4
    # optimizer = build_optimizer(network, learning_rates, momentum, weight_decay)
    optimizer = Optimizer(network, cost_function, learning_rates, momentum, weight_decay)

    # Define the trainer (to perform one epoch of training) and the evaluator (to evaluate the model on the test set)
    num_iterations_inference = 60

    trainer = Trainer(network, cost_function, training_loader, estimator, optimizer)
    trainer.max_iterations = num_iterations_inference

    evaluator = Evaluator(network, cost_function, test_loader)
    evaluator.max_iterations = num_iterations_inference
    
    # Define the scheduler for the learning rates (cosine annealing scheduler)
    num_epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=2.*1e-6)

    # Define the path and the monitor to perform the run
    path = '/'.join(['paper/comparative-study', dataset, algorithm])
    monitor = Monitor(network, cost_function, trainer, scheduler, evaluator, path)

    # Print the characteristics of the run
    print('{} - batch_size={}'.format(dataset, batch_size))
    print(network)
    print(cost_function)
    print('number of iterations at inference = {}'.format(num_iterations_inference))
    print(estimator)
    print(optimizer)
    print('number of epochs = {}'.format(num_epochs))
    print('path = {}'.format(path))
    print('device = {}'.format(device))
    print()

    # Launch the experiment
    monitor.run(num_epochs, verbose=True)
    # monitor.save_network()