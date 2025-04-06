# A universal approximation theorem for nonlinear resistive networks

[Link to the paper](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.23.044009) ([Open-access arXiv version](https://arxiv.org/abs/2312.15063))

This paper proves that, under suitable assumptions of ideality, nonlinear resistive networks built from independent voltages sources, ohmic resistors, diodes, and voltage amplifiers can act as universal function approximators. In particular, it demonstrates how a neural network with rectified linear units (ReLU NN) can be translated into an approximately equivalent `deep resistive network' (DRN).

The paper also compares the performance of DRNs and ReLU NNs with 1, 2 and 3 hidden layers.

## Model architectures

We evaluate three DRN architectures, denoted as DRN-1H, DRN-2H and DRN-3H, corresponding to one, two and three hidden layers, respectively. These models are trained on MNIST and its variants, where each image is 28x28 pixels.

Layer shapes for each DRN model (input, hidden and output layers):
- `DRN-1H`: [(2, 28, 28), (1024,), (10,)]
- `DRN-2H`: [(2, 28, 28), (1024,), (1024,), (10,)]
- `DRN-3H`: [(2, 28, 28), (1024,), (1024,), (1024,), (10,)]

Note: DRNs use two input channels (i.e., duplicated input), unlike standard neural networks.

For comparison, we consider corresponding ReLU neural network models, denoted as ReLU NN-1H, ReLU NN-2H and ReLU NN-3H, with the following layer sizes:
- `ReLU NN-1H`: [(1, 28, 28), (512,), (10,)]
- `ReLU NN-2H`: [(1, 28, 28), (512,), (512,), (10,)]
- `ReLU NN-3H`: [(1, 28, 28), (512,), (512,), (512,), (10,)]

By construction, each DRN layer contains twice as many units as its ReLU NN counterpart (see the paper for more details).

## Reproducing the results of Table 1 and Table 4.

We conduct simulations using three training setups:
- DRN models trained with equilibrium propagation (EP)
- DRN models trained with truncated backpropagation (TBP)
- ReLU NN models trained with backpropagation (BP)

Experiments are performed on the following datasets:
- MNIST
- Kuzushiji MNIST (K-MNIST)
- Fashion MNIST (F-MNIST)

To reproduce results, e.g., training a DRN with one hidden layer using equilibrium propagation on MNIST, run:
``` bash
python papers/universal-drn/drn-vs-dnn.py --dataset='MNIST' --model='DRN-EP' --layers='1'
```