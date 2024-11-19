# Energy-based learning algorithms for analog computing: a comparative study

[Link to the paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a52b0d191b619477cc798d544f4f0e4b-Abstract-Conference.html)

This paper compares three `physical' learning algorithms for energy-based systems, namely:
- `CL`: contrastive learning
- `EP`: equilibrium propagation
- `CpL`: coupled learning

Equilibrium propagation (EP) and coupled learning (CpL) come in three variants each, depending on the sign of the perturbation used:
- `P`: positively-perturbed
- `N`: negatively-perturbed
- `C`: centered

This results in seven algorithms: CL, PEP, NEP, CEP, PCpL, NCpL and CCpL. For example, PEP refers to `positively-perturbed equilibrium propagation'.

These seven algorithms are also compared to two baselines:
- `TBP`: truncated backpropagation
- `RBP`: recurrent backpropagation

## Reproducing the results of the comparative study of learning algorithms (Table 1 in the paper)

The nine learning algorithms are compared on training deep convolutional Hopfield networks (DCHNs) on five datasets:
- MNIST
- Fashion-MNIST
- SVHN
- CIFAR-10
- CIFAR-100

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

## Reproducing the SOTA DCHN results (Table 2 in the paper)

The second contribution of the paper is to provide state-of-the-art results with DCHNs on all five datasets.

To train a network (DCHN) on CIFAR-10 for 300 epochs (using CEP), run the command:
``` bash
python papers/ebl-algorithms/sota-dchn.py --dataset='CIFAR10' --epochs=300
```

## Reproducing the speedup results (Table 3 of Appendix C)

The third contribution of the paper is to provide an algorithm for minimizing the energy function of the DCHN more rapidly than prior papers.

To train a network using the hyperparameters of [Laborieux et al. (2021)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.633674/full) and 32-bit precision, run the command:
``` bash
python papers/ebl-algorithms/speedup.py --32bit
```