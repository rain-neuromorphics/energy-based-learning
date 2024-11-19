# A fast algorithm to simulate nonlinear resistive networks

[Link to the paper](https://proceedings.mlr.press/v235/scellier24a.html)

This paper describes a fast procedure for computing the steady state of a nonlinear resistive network comprising linear resistors, ideal diodes, voltages sources and current sources. The steady state of such networks is shown to minimize the power dissipation function, under linear inequality constraints (imposed by diodes) and linear equality constraints (imposed by voltage sources). An exact coordinate descent algorithm is derived to compute the minimum of this function with respect to the node electrical potentials.

## Reproducing the results of Deep Resistive Networks (DRNs, Table 1)

The primary contribution of the paper is to demonstrate that simulations are orders of magnitude faster than previous simulations of nonlinear resistive networks.

Five `deep resistive network' (DRN) models are considered. The sizes of the layers for each architecture are as follows:
- `DRN-XS`: [(2, 28, 28), (100,), (10,)], a model of the same size as the one of [Kendall et al. (2020)](https://arxiv.org/abs/2006.01981)
- `DRN-XL`: [(2, 28, 28), (32768,), (10,)], the largest model in terms of parameter count
- `DRN-1H`: [(2, 28, 28), (1024,), (10,)], a 1-hidden-layer model of standard size
- `DRN-2H`: [(2, 28, 28), (1024,), (1024,), (10,)], a 2-hidden-layer model
- `DRN-3H`: [(2, 28, 28), (1024,), (1024,), (1024,), (10,)], a 3-hidden-layer model

Two learning algorithms are considered for training:
- `EP`: equilibrium propagation,
- `BP`: truncated backpropagation, used as a baseline.

We train the five DRN models using EP on the MNIST dataset, and use BP as a baseline. To start a simulation of the training process of e.g. the DRN-1H with EP, run the command:

``` bash
python papers/fast-drn/drn.py --model='drn-1h' --algorithm='EP'
```

On a A100 GPU, the full run (50 epochs) should take ~ 2 hours 36 minutes.

## Reproducing the results of Deep Hopfield Networks (DHNS, Table 3 in Appendix E)

The paper also compares the DRN models with deep Hopfield network (DHN) models of the same size. These DHN models are trained using EP on the MNIST dataset.

Three DHN models are considered. The sizes of the layers for each architecture are as follows:
- `DHN-1H`: [(1, 28, 28), (1024,), (10,)]
- `DHN-2H`: [(1, 28, 28), (1024,), (1024,), (10,)]
- `DHN-3H`: [(1, 28, 28), (1024,), (1024,), (1024,), (10,)]

To train e.g. the DHN-1H model with EP, run the command:

``` bash
python papers/fast-drn/dhn.py --model='dhn-1h'
```