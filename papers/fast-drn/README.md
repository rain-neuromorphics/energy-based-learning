# A fast algorithm to simulate nonlinear resistive networks

[Link to the paper](https://arxiv.org/abs/2402.11674)

We provide the code to reproduce the simulation results of the paper.

## Reproduce the results of Table 1 (Deep Resistive Networks, or DRNs)

We consider five deep resistive network (DRN) models, denoted DRN-XS, DRN-XL, DRN-1H, DRN-2H and DRN-3H. The sizes of the layers for each architecture are as follows:
- `XS`: [(2, 28, 28), (100,), (10,)]
- `XL`: [(2, 28, 28), (32768,), (10,)]
- `1H`: [(2, 28, 28), (1024,), (10,)]
- `2H`: [(2, 28, 28), (1024,), (1024,), (10,)]
- `3H`: [(2, 28, 28), (1024,), (1024,), (1024,), (10,)]

We train these DRN models using equilibrium propagation (EP) and backpropagation (BP) on the MNIST dataset. To start a simulation of the training process of e.g. the DRN-1H with EP, run the command:

``` bash
python papers/fast-drn/drn.py --model='drn-1h' --algorithm='EP'
```

On a A100 GPU, the full run (50 epochs) should take ~ 2 hours 36 minutes.

## Reproduce the results of Table 3 in Appendix E (Deep Hopfield Networks, or DHNs)

We consider three deep Hopfield network (DHN) models, denoted DRN-1H, DRN-2H and DRN-3H. The sizes of the layers for each architecture are as follows:
- `1H`: [(2, 28, 28), (1024,), (10,)]
- `2H`: [(2, 28, 28), (1024,), (1024,), (10,)]
- `3H`: [(2, 28, 28), (1024,), (1024,), (1024,), (10,)]

We train these DHN models using equilibrium propagation (EP) on the MNIST dataset. To train e.g. the DHN-1H with EP, run the command:

``` bash
python papers/fast-drn/dhn.py --model='dhn-1h'
```