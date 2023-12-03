import argparse
import torch

from datasets import load_dataloaders
from network.hopfield import ConvHopfieldNet32
from network.cost import SquaredError
from training.sgd import Eqprop
from training.epoch import Trainer, Evaluator
from training.monitor import Monitor, Optimizer

parser = argparse.ArgumentParser(description='Best results obtained with a deep convolutional Hopfield network')
parser.add_argument('--dataset', type = str, default = 'CIFAR10', help="The dataset used. Either `CIFAR10' or `CIFAR100'")
parser.add_argument('--epochs', type = int, default = 100, help="The number of epochs of training")
parser.add_argument('--device', type = str, default = None, help="The device where we run and train the network")

args = parser.parse_args()


if __name__ == "__main__":

    dataset = args.dataset

    if dataset == 'CIFAR10':
        nudging = 0.1
        num_iterations_inference = 60
        num_iterations_training = 20
        weight_gains = [0.4, 0.7, 0.6, 0.3, 0.4]
        learning_rates_weights = [0.03, 0.03, 0.03, 0.03, 0.03]
        learning_rates_biases = [0.03, 0.03, 0.03, 0.03, 0.03]
        momentum = 0.9
        weight_decay = 2.5 * 1e-4
        batch_size = 128
    elif dataset == 'CIFAR100':
        nudging = 0.25
        num_iterations_inference = 60
        num_iterations_training = 15
        weight_gains = [0.5, 0.4, 0.5, 0.8, 0.5]
        learning_rates_weights = [0.03, 0.04, 0.04, 0.04, 0.025]
        learning_rates_biases = [0.03, 0.04, 0.04, 0.04, 0.025]
        momentum = 0.9
        weight_decay = 3.5 * 1e-4
        batch_size = 128
    else:
        raise ValueError("expected 'CIFAR10' or 'CIFAR100' but got {}".format(dataset))

    # Load the training and test data (either CIFAR-10 or CIFAR-100)
    training_loader, test_loader = load_dataloaders(dataset, batch_size)

    # Specify the number of input filters and output units of the network, depending on the dataset
    num_inputs = 3
    num_outputs = 10 if dataset == 'CIFAR10' else 100

    # Build the network (a convolutional Hopfield net with 32x32-pixel input images)
    network = ConvHopfieldNet32(num_inputs, num_outputs, weight_gains=weight_gains)

    # Set the device on which we run and train the network
    if args.device != None: device = args.device
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    network.set_device(device)
    
    # Define the cost function (mean squared error)
    cost_function = SquaredError(network)

    # Define the gradient estimator (equilibrium propagation)
    estimator = Eqprop(network, cost_function)
    estimator.nudging = nudging
    estimator.training_mode = 'centered'
    estimator.max_iterations = num_iterations_training

    # Build the optimizer (SGD with momentum and weight decay)
    learning_rates = learning_rates_biases + learning_rates_weights
    optimizer = Optimizer(network, cost_function, learning_rates, momentum, weight_decay)

    # Define the trainer (to perform one epoch of training) and the evaluator (to evaluate the model on the test set)
    trainer = Trainer(network, cost_function, training_loader, estimator, optimizer)
    trainer.max_iterations = num_iterations_inference

    evaluator = Evaluator(network, cost_function, test_loader)
    evaluator.max_iterations = num_iterations_inference
    
    # Define the scheduler for the learning rates (cosine annealing scheduler)
    num_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=2.*1e-6)

    # Define the path and the monitor to perform the run
    path = '/'.join(['paper/sota-dchn', dataset])
    monitor = Monitor(network, cost_function, trainer, scheduler, evaluator, path)

    # Print the characteristics of the run
    print('{} - batch_size={}'.format(dataset, batch_size))
    print(network)
    print(cost_function)
    print('num iterations inference = {}'.format(num_iterations_inference))
    print(estimator)
    print('num epochs = {}'.format(num_epochs))
    print('path = {}'.format(path))
    print('device={}'.format(device))
    print()

    # Launch the experiment
    monitor.run(num_epochs, verbose=True)
    # monitor.save_network()