# Energy-based learning algorithms for analog computing: a comparative study

[Link to the paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a52b0d191b619477cc798d544f4f0e4b-Abstract-Conference.html)

We provide the code to reproduce the simulation results of the paper.

## Reproduce the results of the comparative study of learning algorithms (Table 1 in the paper)

The study compares nine learning algorithms -- seven 'physical' learning algorithms (CL, PEP, NEP, CEP, PCpL, NCpL, CCpL) and two baselines (TBP, RBP) -- to train deep convolutional Hopfield networks (DCHNs) on five datasets (MNIST, Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100). The terminology is the following:
- `CL`: contrastive learning
- `EP`: equilibrium propagation
- `CpL`: coupled learning
- `TBP`: truncated backpropagation
- `RBP`: recurrent backpropagation

Equilibrium propagation (EP) and coupled learning (CpL) come in three variants each:
- `P`: positively-perturbed
- `N`: negatively-perturbed
- `C`: centered
For example, PEP refers to `positively-perturbed equilibrium propagation'.

To launch an experiment e.g. on CIFAR-100 using CEP, run the command:

``` bash
python papers/ebl-algorithms/comparative-study.py --dataset='CIFAR100' --algorithm='CEP'
```
On a A100 GPU, the full run (100 epochs) should take 3 hours.

To reproduce the results of the additional simulations on SVHN presented in Appendix E of the paper, e.g. with the NEP algorithm, run the command:

``` bash
python papers/ebl-algorithms/comparative-study.py --dataset='SVHN' --algorithm='NEP' --gain=0.7
```

To customize the run, you can use the following command-line options:
- `--dataset`: either 'MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10' or 'CIFAR100'
- `--algorithm`: the learning algorithm to train the network. Either CL, PEP, NEP, CEP, PCpL, NCpL, CCpL, TBP or RBP
- `--gain`: the gain to initialize the weights of the network. Default value is 0.5
- `--device`: specify the device where to run the simulation (e.g. 'cpu' or 'cuda')

## Reproduce the SOTA DCHN results (Table 2 in the paper)

To train a network (DCHN) on CIFAR-10 for 300 epochs (using CEP), run the command:
``` bash
python papers/ebl-algorithms/sota-dchn.py --dataset='CIFAR10' --epochs=300
```

## Reproduce the speedup results (Table 3 of appendix C)

To train a network using the hyperparameters of Laborieux et al. (2021) and 32-bit precision, run the command:
``` bash
python papers/ebl-algorithms/speedup.py --32bit
```