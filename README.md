# Verifier-Constrained Flow Expansion for Discovery Beyond the Data


[![arXiv](http://img.shields.io/badge/arxiv-2602.15984-red?logo=arxiv)](https://arxiv.org/abs/2602.15984)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Komod0D/flow-expansion/blob/main/tutorial.ipynb)

This repository contains the official implementation of the Flow Expansion algorithm, a method for verifier-constrained exploration for discovery beyond the data.

## Installation

Create conda environment:
```bash
conda env create -f environment.yaml
```
Activate environment:
```bash
conda activate genexp
```
In root directory, install package:
```bash
pip3 install -e .
```

## Usage

To use FE on your own models, follow the installation instructions above and check the ``tutorial.ipynb`` notebook for a typical first example. 

For more fine-grained control (especially if your model generates something other than a PyTorch Tensor):

### Wrap your model with either our FlowModel class or our DiffusionModel class (or subclass them)

We define a prototypical `FlowModel` class (and `DiffusionModel` class extending it) in `src/genexp/models.py` in order to wrap simple models. We assume your model is a velocity field predictor in the case of flow models, and a noise predictor in the case of diffusion models. If your flow/diffusion model does something different, simply subclass the respective class, and implement the `velocity_field`/`score_func` according to your model's functionality. Note that we use alpha/beta according to their definitions in the [Adjoint Matching paper](https://arxiv.org/abs/2409.08861), see section 2.

### Choose one of our Sampler classes (or subclass your own)

If your model outputs a PyTorch tensor, we recommend using our `EulerMaruyamaSampler` by simply passing the data shape. Otherwise, subclass the `Sampler` class, and the `Sample` object, in order to define exactly part of your data is used in the Adjoint Matching procedure. No matter how complex your data is, by implementing the `adjoint` property of your `Sample` object you define a tensor with respect to which gradients can be computed in FE.

## Citation

If you use this code in your research, please include the following citation in your work:


```
@inproceedings{de2026verifier,
	title={Verifier-Constrained Flow Expansion for Discovery Beyond the Data},
	author={De Santi*, Riccardo and Protopapas*, Kimon and Hsieh, Ya-Ping and Krause, Andreas},
	booktitle={International Conference on Learning Representations (ICLR)},
	year={2026},
	month={April},
	pdf={https://arxiv.org/pdf/2602.15984},
}
```
