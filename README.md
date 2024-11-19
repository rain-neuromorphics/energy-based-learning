# Energy-Based Learning Framework

This repository aims at developing code for simulating energy-based systems and the training process of such systems using learning algorithms like equilibrium propagation. In particular, this repository contains the code to reproduce the results of the following papers:
- [Energy-based learning algorithms for analog computing: a comparative study](papers/ebl-algorithms/README.md)
- [A fast algorithm to simulate nonlinear resistive networks](papers/fast-drn/README.md)
- [A universal approximation theorem for nonlinear resistive networks](papers/universal-drn/README.md)


## Prerequisites

First, download the code and navigate to the project directory:
``` bash
git clone https://github.com/rain-neuromorphics/energy-based-learning
cd energy-based-learning
```
You will need to install PyTorch, Torchvision and TensorBoard.
You will also need to export the path to the main directory:
``` bash
export PYTHONPATH="${PYTHONPATH}:/path/to/the/main/directory"
```
To know the path of the main directory (your current location) you may use the command `pwd'.


## Getting in Touch and Citation

If you are interested, have any questions, comments, or would like to explore collaborative opportunities, do not hesitate to reach out to benjamin at rain dot ai.

Additionally, if you use this codebase in your research, we kindly ask you to acknowledge it by either referencing it or citing the relevant paper(s).