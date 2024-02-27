# A fast algorithm to simulate nonlinear resistive networks

[Link to the paper](https://arxiv.org/abs/2402.11674)

We provide the code to reproduce the results of Table 1. We consider four deep resistive network (DRN) models, denoted DRN-XL, DRN-1H, DRN-2H and DRN-3H. The sizes of the layers for each architecture are as follows:
- `XL`: [(2, 28, 28), (32768,), (10,)]
- `1H`: [(2, 28, 28), (1024,), (10,)]
- `2H`: [(2, 28, 28), (1024,), (1024,), (10,)]
- `3H`: [(2, 28, 28), (1024,), (1024,), (1024,), (10,)]

We train these DRN models using equilibrium propagation (EP) and backpropagation (BP) on the MNIST dataset. To launch a simulation of the training process of e.g. a DRN-1H model with EP, run the command:

``` bash
python papers/fast-drn/drn.py --model='drn-1h' --algorithm='EP'
```

On a A100 GPU, the full run (50 epochs) should take 3 hours.