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

parser = argparse.ArgumentParser(description='Simulation speedup of deep convolutional Hopfield networks')
parser.add_argument('--32bit', dest='use32bit', action='store_true', help="Use 32 bit precision for the units and weights")
parser.add_argument('--16bit', dest='use32bit', action='store_false', help="Use 16 bit precision for the units and weights")

args = parser.parse_args()


if __name__ == "__main__":

    if args.use32bit:
        # hyperparameters used in Laborieux et al. (2021)
        num_iterations_inference = 250
        num_iterations_training = 30
        mode = 'synchronous'

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    else:
        # hyperparameters used this work
        num_iterations_inference = 60
        num_iterations_training = 20
        mode = 'asynchronous'

    # Load the training and test data (CIFAR-10)
    dataset = 'CIFAR10'
    batch_size = 128
    training_loader, test_loader = load_dataloaders(dataset, batch_size)

    # Build the network (a convolutional Hopfield net with 32x32-pixel input images)
    num_inputs = 3
    num_outputs = 10
    weight_gains = [0.4, 0.7, 0.6, 0.3, 0.4]
    energy_fn = ConvHopfieldEnergy32(num_inputs, num_outputs, weight_gains=weight_gains)

    # Set the device on which we run and train the network
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    energy_fn.set_device(device)
    
    # Define the cost function (mean squared error)
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredError(output_layer)
    network = Network(energy_fn)

    # Define the energy minimizer used in the perturbed phase of training
    params = energy_fn.params()
    layers = energy_fn.layers()
    free_layers = network.free_layers()
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    energy_minimizer_training = FixedPointMinimizer(augmented_fn, free_layers)
    energy_minimizer_training.mode = mode
    energy_minimizer_training.num_iterations = num_iterations_training

    # Define the gradient estimator (equilibrium propagation)
    estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training)
    nudging = 0.1
    estimator.variant = 'centered'
    estimator.nudging = nudging

    # Build the optimizer (SGD with momentum and weight decay)
    learning_rates_weights = [0.03, 0.03, 0.03, 0.03, 0.03]
    learning_rates_biases = [0.03, 0.03, 0.03, 0.03, 0.03]
    momentum = 0.9
    weight_decay = 2.5 * 1e-4
    learning_rates = learning_rates_biases + learning_rates_weights
    optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

    # Define the trainer (to perform one epoch of training) and the evaluator (to evaluate the model on the test set)
    energy_minimizer_inference = FixedPointMinimizer(energy_fn, free_layers)
    energy_minimizer_inference.mode = mode
    energy_minimizer_inference.num_iterations = num_iterations_inference

    trainer = Trainer(network, cost_fn, params, training_loader, estimator, optimizer, energy_minimizer_inference)
    evaluator = Evaluator(network, cost_fn, test_loader, energy_minimizer_inference)
    
    # Define the scheduler for the learning rates (cosine annealing scheduler)
    num_epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=2.*1e-6)

    bits = '32bit' if args.use32bit else '16bit'

    # Define the path and the monitor to perform the run
    path = '/'.join(['papers/ebl-algorithms/speedup', bits])
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