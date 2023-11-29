# osc_thesis
## Training of OSC on MNIST and Letters

## Setup

For installation, we rely on Conda.
We have good experiences using the [MiniConda Installation](https://docs.conda.io/en/latest/miniconda.html), but other conda versions should also work.
We provide a conda installation script to install all the dependencies.
Please run:

    conda env create -f environment.yaml

Afterward, activate the environment via:

    conda activate osc_thesis

This will setup everything that we need.
Particularly, it will install `pytorch`, `torchvision` including the `cudatoolkit` (you might want to skip this when no GPU is available to you).
Also, the `vast` package is installed from its [GitHub Source](https://github.com/Vastlab/vast.git)

## Data

The scripts rely on the EMNIST digits and EMNIST letters, which we set to automatically download from `torchvision.datasets`.
Currently, we follow the approach of using both Letters and MNIST from the EMNIST splits, removing extremely-similar letters (i, g, l, o).
Hence, both negative and unknown classes contain 11 different letters each.

