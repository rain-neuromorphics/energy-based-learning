import argparse
import numpy
import torch

from datasets import load_dataloaders
from model.resistive.network import DeepResistiveEnergy
from model.forward.network import ReLUNeuralNet
from model.function.network import Network
from model.function.cost import SquaredError
from model.resistive.minimizer import QuadraticMinimizer
from model.forward.minimizer import ForwardPass
from training.sgd import AugmentedFunction, EquilibriumProp
from training.sgd import Backprop
from training.epoch import Trainer, Evaluator
from training.monitor import Monitor, Optimizer

parser = argparse.ArgumentParser(description='Comparison of deep resistive networks (DRNs) and ReLU neural networks (ReLU NNs)')
parser.add_argument('--dataset', type = str, default = 'MNIST', help="The dataset used. Either `MNIST', `KMNIST' or `FMNIST'")
parser.add_argument('--model', type = str, default = 'DRN-TBP', help="The network model. Either 'DRN-EP' (Deep Resistive Network trained with Equilibrium Propagation), 'DRN-TBP' (Deep Resistive Network trained with Truncated Backpropagation) or 'NN-BP' (ReLU Neural Network trained with Backpropagation)")
parser.add_argument('--layers', type = int, default = 1, help="The number of layers. Either 1, 2 or 3.")

args = parser.parse_args()



if __name__ == "__main__":
    
    # Expected results

    # MNIST DRN-1H EP:     1.53% test error and 0.07% train error
    # MNIST DRN-1H TBP:    1.51% test error and 0.04% train error
    # MNIST ReLU NN-1H BP: 1.67% test error and 0.22% train error
    #############################################################
    # MNIST DRN-2H EP:     1.60% test error and 0.20% train error
    # MNIST DRN-2H TBP:    1.46% test error and 0.20% train error
    # MNIST ReLU NN-2H BP: 1.34% test error and 0.03% train error
    #############################################################
    # MNIST DRN-3H EP:     1.77% test error and 0.36% train error
    # MNIST DRN-3H TBP:    1.47% test error and 0.30% train error
    # MNIST ReLU NN-3H BP: 1.48% test error and 0.02% train error

    # K-MNIST DRN-1H EP:     7.57% test error and 0.08% train error
    # K-MNIST DRN-1H TBP:    7.67% test error and 0.06% train error
    # K-MNIST ReLU NN-1H BP: 8.83% test error and 0.12% train error
    ###############################################################
    # K-MNIST DRN-2H EP:     8.40% test error and 0.39% train error
    # K-MNIST DRN-2H TBP:    8.27% test error and 0.42% train error
    # K-MNIST ReLU NN-2H BP: 8.14% test error and 0.05% train error
    ###############################################################
    # K-MNIST DRN-3H EP:     9.30% test error and 0.63% train error
    # K-MNIST DRN-3H TBP:    8.39% test error and 0.63% train error
    # K-MNIST ReLU NN-3H BP: 7.99% test error and 0.03% train error

    # F-MNIST DRN-1H EP:     10.38% test error and 6.14% train error
    # F-MNIST DRN-1H TBP:    10.31% test error and 5.91% train error
    # F-MNIST ReLU NN-1H BP: 10.20% test error and 5.13% train error
    ################################################################
    # F-MNIST DRN-2H EP:     10.29% test error and 6.20% train error
    # F-MNIST DRN-2H TBP:    9.88% test error and 4.88% train error
    # F-MNIST ReLU NN-2H BP: 9.42% test error and 2.78% train error
    ###############################################################
    # F-MNIST DRN-3H EP:     11.23% test error and 8.19% train error
    # F-MNIST DRN-3H TBP:    9.79% test error and 4.58% train error
    # F-MNIST ReLU NN-3H BP: 9.55% test error and 1.82% train error

    dataset = args.dataset
    model = args.model
    num_layers = args.layers

    if dataset == 'MNIST':
        dataset_gain = 8.
    elif dataset == 'FMNIST':
        dataset = 'FashionMNIST'
        dataset_gain = 5.
    elif dataset == 'KMNIST':
        dataset = 'KuzushijiMNIST'
        dataset_gain = 5.
    else:
        raise ValueError("expected 'MNIST', 'KMNIST' or 'FMNIST' but got {}".format(model))

    # Hyperparameters
    if model[:3] == 'DRN':
        if num_layers == 1:
            layer_shapes = [(2, 28, 28), (1024,), (10,)]
            model_gain = 60.
            num_iterations_inference = 4
            num_iterations_training = 4
            learning_rates_weights = [0.005, 0.005]
            learning_rates_biases = [0.005, 0.005]
            nudging = 0.5  # for EP only
        elif num_layers == 2:
            layer_shapes = [(2, 28, 28), (1024,), (1024,), (10,)]
            model_gain = 250.
            num_iterations_inference = 5
            num_iterations_training = 5
            learning_rates_weights = [0.002, 0.006, 0.005]
            learning_rates_biases = [0.002, 0.006, 0.005]
            nudging = 1.0  # for EP only
        elif num_layers == 3:
            layer_shapes = [(2, 28, 28), (1024,), (1024,), (1024,), (10,)]
            model_gain = 500.
            num_iterations_inference = 6
            num_iterations_training = 6
            learning_rates_weights = [0.005, 0.02, 0.08, 0.005]
            learning_rates_biases = [0.0005, 0.002, 0.008, 0.0005]
            nudging = 2.0  # for EP only
        else:
            raise ValueError("expected 1, 2 or 3 but got {}".format(num_layers))
    elif model == 'NN-BP':
        if num_layers == 1:
            layer_shapes = [(1, 28, 28), (512,), (10,)]
            learning_rates_weights = [0.005, 0.005]
            learning_rates_biases = [0.005, 0.005]
        elif num_layers == 2:
            layer_shapes = [(1, 28, 28), (512,), (512,), (10,)]
            learning_rates_weights = [0.005, 0.005, 0.005]
            learning_rates_biases = [0.005, 0.005, 0.005]
        elif num_layers == 3:
            layer_shapes = [(1, 28, 28), (512,), (512,), (512,), (10,)]
            learning_rates_weights = [0.005, 0.005, 0.005, 0.005]
            learning_rates_biases = [0.005, 0.005, 0.005, 0.005]
        else:
            raise ValueError("expected 1, 2 or 3 but got {}".format(num_layers))
    else:
        raise ValueError("expected 'DRN-EP', 'DRN-TBP' or 'NN-BP' but got {}".format(model))

    # Load the training and test data
    batch_size = 32
    training_loader, test_loader = load_dataloaders(dataset, batch_size, augment_32x32=False, normalize=False)

    # Build the energy function (DRN or ReLU NN)
    weight_gains = [1.0] * (len(layer_shapes) - 1)
    if model[:3] == 'DRN':
        input_gain = dataset_gain * model_gain
        energy_fn = DeepResistiveEnergy(layer_shapes, weight_gains, input_gain)
    elif model == 'NN-BP':
        energy_fn = ReLUNeuralNet(layer_shapes, weight_gains)
    else:
        raise ValueError("expected 'DRN-EP', 'DRN-TBP' or 'NN-BP' but got {}".format(model))

    # Set the device on which we run and train the network
    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    energy_fn.set_device(device)
    
    # Define the cost function (mean squared error)
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredError(output_layer)
    
    # Define the network
    network = Network(energy_fn)
    free_layers = network.free_layers()
    params = energy_fn.params()
    layers = energy_fn.layers()

    # Build the energy minimizers and gradient estimator
    if model == 'DRN-EP':
        # Define the energy minimizer at inference
        energy_minimizer_inference = QuadraticMinimizer(energy_fn, free_layers)
        energy_minimizer_inference.num_iterations = num_iterations_inference
        energy_minimizer_inference.mode = 'forward'
        # Define the energy minimizer during training
        augmented_fn = AugmentedFunction(energy_fn, cost_fn)
        energy_minimizer_training = QuadraticMinimizer(augmented_fn, free_layers)
        energy_minimizer_training.num_iterations = num_iterations_training
        energy_minimizer_training.mode = 'backward'
        # Define the gradient estimator (equilibrium propagation)
        estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training)
        estimator.nudging = nudging
        estimator.variant = 'centered'
    elif model == 'DRN-TBP':
        # Define the energy minimizer at inference
        energy_minimizer_inference = QuadraticMinimizer(energy_fn, free_layers)
        energy_minimizer_inference.num_iterations = num_iterations_inference
        energy_minimizer_inference.mode = 'forward'
        # Define the energy minimizer during training
        energy_minimizer_training = QuadraticMinimizer(energy_fn, free_layers)
        energy_minimizer_training.num_iterations = num_iterations_training
        energy_minimizer_training.mode = 'forward'
        # Define the gradient estimator (backpropagation)
        estimator = Backprop(params, layers, cost_fn, energy_minimizer_training)
    elif model == 'NN-BP':
        # Define the energy minimizers at inference and during training
        energy_minimizer_inference = ForwardPass(energy_fn, free_layers)
        energy_minimizer_training = ForwardPass(energy_fn, free_layers)
        # Define the gradient estimator (backpropagation)
        estimator = Backprop(params, layers, cost_fn, energy_minimizer_training)
    else:
        raise ValueError("expected 'DRN-EP', 'DRN-TBP' or 'NN-BP' but got {}".format(model))
    
    # Build the optimizer (SGD)
    learning_rates = learning_rates_biases + learning_rates_weights
    momentum = 0.9
    weight_decay = 0.
    optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

    # Define the trainer (to perform one epoch of training) and the evaluator (to evaluate the model on the test set)
    trainer = Trainer(network, cost_fn, params, training_loader, estimator, optimizer, energy_minimizer_inference)
    evaluator = Evaluator(network, cost_fn, test_loader, energy_minimizer_inference)
    
    # Define the scheduler for the learning rates
    num_epochs = 100
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Define the path and the monitor to perform the run
    path = '/'.join(['papers/universal-drn/drn-vs-dnn', model, str(num_layers)])
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

    for param in params:
        (std, mean) = torch.std_mean(param.state.to('cpu'))
        maxi = torch.max(param.state.to('cpu'))
        print('{}: mean={:.5f} std={:.5f}, max={:.5f}'.format(param.name, mean, std, maxi))