# A universal approximation theorem for nonlinear resistive networks

[Link to the paper](https://arxiv.org/abs/2312.15063)

This paper proves, under suitable assumptions of ideality, that nonlinear resistive networks comprising linear resistors, ideal diodes, voltages sources and VCVS are universal function approximators. It shows in particular how to translate a ReLU neural network (NN) into an approximately equivalent `deep resistive network' (DRN). The paper also provides a comparison of the performance of DRN and ReLU NN models with 1, 2 and 3 hidden layers.

We consider three deep resistive network (DRN) models, denoted DRN-1H, DRN-2H and DRN-3H. The sizes of the layers for each architecture are as follows:
- `DRN-1H`: [(2, 28, 28), (1024,), (10,)]
- `DRN-2H`: [(2, 28, 28), (1024,), (1024,), (10,)]
- `DRN-3H`: [(2, 28, 28), (1024,), (1024,), (1024,), (10,)]

We also consider three ReLU neural network models, denoted ReLU NN-1H, ReLU NN-2H and ReLU NN-3H. The sizes of the layers for each architecture are as follows:
- `ReLU NN-1H`: [(1, 28, 28), (512,), (10,)]
- `ReLU NN-2H`: [(1, 28, 28), (512,), (512,), (10,)]
- `ReLU NN-3H`: [(1, 28, 28), (512,), (512,), (512,), (10,)]

## Reproducing the results of Table 1 and Table 4.

We performs simulations of:
- DRN models trained with equilibrium propagation (EP)
- DRN models trained with truncated backpropagation (TBP)
- ReLU NN models trained with backpropagation (BP)

We perform simulations on three datasets:
- MNIST
- Kuzushiji MNIST (K-MNIST)
- Fashion MNIST (F-MNIST)

To train e.g. a DRN with one hidden layer on the MNIST dataset, run the command:

``` bash
python papers/universal-drn/drn-vs-dnn.py --dataset='MNIST' --model='drn' --layers='1'
```