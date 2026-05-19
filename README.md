# Verifier-Constrained Flow Expansion for Discovery Beyond the Data


[![arXiv](http://img.shields.io/badge/arxiv-2602.15984-red?logo=arxiv)](https://arxiv.org/abs/2602.15984)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Komod0D/flow-expansion/blob/main/tutorial.ipynb)

This repository contains the official implementation of the Flow Expansion algorithm, a method for verifier-constrained exploration for discovery beyond the data.

## Installation

```bash
pip install genexp
```

Or, to install from source in editable mode:

```bash
pip install -e .
```

## Overview

Flow Expansion is built on top of [diffusiongym](https://github.com/cristianpjensen/diffusiongym), a library for reward adaptation of pre-trained flow models across any data modality. To run Flow Expansion on your own model you need three things:

1. A **data type** (e.g. `DDTensor` for plain tensors, or a custom `DDMixin` subclass for structured data)
2. A **base model** (`BaseModel[D]`) wrapping your pre-trained network
3. A **reward** (`Reward[D]`) measuring sample quality

diffusiongym then handles environment construction, SDE simulation, and trajectory storage. `FlowExpansionTrainer` runs the optimization loop on top.

## Usage

### 1. Data type

For plain tensor data, use the built-in `DDTensor`:

```python
from diffusiongym import DDTensor

x = DDTensor(torch.randn(batch_size, dim))
```

For structured data (graphs, molecules, images with conditioning), subclass `DDMixin` and implement `apply`, `combine`, `aggregate`, `collate`, `__len__`, and `__getitem__`. See [diffusiongym's types documentation](https://cristianpjensen.github.io/diffusiongym/) for details.

### 2. Base model

Subclass `BaseModel[D]` and set `output_type` to one of `"velocity"`, `"score"`, `"epsilon"`, or `"endpoint"` depending on what your network predicts:

```python
import torch
import torch.nn as nn
from typing import Any
from diffusiongym import BaseModel, DDTensor, OptimalTransportScheduler
from diffusiongym.schedulers import Scheduler

class MyFlowModel(BaseModel[DDTensor]):
    output_type = "velocity"   # or "score" | "epsilon" | "endpoint"

    def __init__(self, dim: int, device=None):
        super().__init__(device)
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256), nn.SiLU(), nn.Linear(256, dim)
        )
        self._scheduler = OptimalTransportScheduler()

    @property
    def scheduler(self) -> Scheduler[DDTensor]:
        return self._scheduler

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[DDTensor, dict]:
        return DDTensor(torch.randn(n, dim, device=self.device)), kwargs

    def forward(self, x: DDTensor, t: torch.Tensor, **kwargs: Any) -> DDTensor:
        t_in = t.unsqueeze(1) if t.ndim == 1 else t
        out = self.net(torch.cat([x.data, t_in], dim=1))
        return DDTensor(out)
```

The `scheduler` defines the interpolant `x_t = α_t x_1 + β_t x_0`. `OptimalTransportScheduler` uses the linear schedule `α_t = t`, `β_t = 1 − t`. `CosineScheduler` and `DiffusionScheduler` are also available.

All four `output_type` values are mathematically equivalent — `FlowExpansionTrainer` converts between them internally using the scheduler.

The base class provides `train_loss(x1)` automatically once `output_type` and `scheduler` are set, so you can train your model with:

```python
import diffusiongym
diffusiongym.train_base_model(model, optimizer, data, steps=10_000)
```

### 3. Reward

Subclass `Reward[D]` and implement `__call__`, which returns a `(rewards, valids)` pair — both `torch.Tensor` of shape `(n,)`:

```python
from diffusiongym import Reward, DDTensor

class MyReward(Reward[DDTensor]):
    def __call__(self, sample: DDTensor, latent: DDTensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = -sample.data.norm(dim=-1)           # example: penalise large norms
        valids  = torch.ones(len(sample))
        return rewards, valids
```

### 4. Environment

Create an environment that matches your model's `output_type`. The simplest way is `construct_env`, which picks the right environment class automatically:

```python
import diffusiongym

env = diffusiongym.construct_env(
    base_model=model,
    reward=MyReward(),
    discretization_steps=100,
    reward_scale=1.0,
)
```

Alternatively, instantiate the environment class directly:

```python
from diffusiongym import VelocityEnvironment   # or Score/Epsilon/EndpointEnvironment

env = VelocityEnvironment(model, MyReward(), discretization_steps=100)
```

If your model is registered in diffusiongym's registry, you can also use the `make()` factory:

```python
from diffusiongym import base_model_registry

@base_model_registry.register("mytask/mymodel")
class MyFlowModel(BaseModel[DDTensor]):
    ...

env = diffusiongym.make(
    base_model="mytask/mymodel",
    reward="mytask/myreward",
    discretization_steps=100,
    device=device,
)
```

### 5. Flow Expansion trainer

Once you have an environment, pass it to `FlowExpansionTrainer` along with fine and base model copies:

```python
import copy
from omegaconf import OmegaConf
from genexp import FlowExpansionTrainer

config = OmegaConf.create({
    "gamma": 1.0,       # score-weighting strength
    "eta": 1.0,         # projection step weight (set 0 if no constraint)
    "beta": 0.0,        # score subtraction coefficient
    "epsilon": 0.01,    # endpoint clipping
    "traj": True,       # use trajectory-level adjoint (recommended)
    "lmbda": "const",   # lambda schedule: "const" or "variance"
    "adjoint_matching": {
        "lr": 1e-4,
        "batch_size": 128,
        "clip_grad_norm": 1.0,
        "clip_loss": 1e5,
        "sampling": {"num_samples": 512},
    },
})

fine_model = copy.deepcopy(env.base_model)
base_model = copy.deepcopy(env.base_model)

trainer = FlowExpansionTrainer(config, env, fine_model, base_model, device=device)
```

Then run the expansion loop with `fit`. Each mirror-descent iteration consists of an **expand** step (move toward higher reward) followed by a **project** step (pull back toward the constraint set), each with one or more adjoint-matching fine-tuning rounds configured via `adjoint_matching.num_iterations` and `adjoint_matching.finetune_steps`:

```python
losses = trainer.fit(num_iterations=10)
```

`fit` returns a flat list of per-AM-round losses (expand rounds first, then project rounds, for each iteration), which you can use to monitor convergence.

`project()` requires `grad_constraint` to be set on the trainer (pass it as a keyword argument to `FlowExpansionTrainer`). It is the gradient of the constraint functional with respect to the sample.

## Quickstart

Check `tutorial.ipynb` for a complete worked example on a 1D trimodal GMM, using diffusiongym's built-in pre-trained model:

```python
import diffusiongym, copy
from omegaconf import OmegaConf
from genexp import FlowExpansionTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = diffusiongym.make(
    base_model="1d/trimodal_gmm",
    reward="1d/binary",
    discretization_steps=50,
    device=device,
)

fine_model = copy.deepcopy(env.base_model)
base_model = copy.deepcopy(env.base_model)
trainer = FlowExpansionTrainer(config, env, fine_model, base_model, device=device)

losses = trainer.fit(num_iterations=3)
```

## Citation

If you use this code in your research, please include the following citation in your work:


```
@inproceedings{de2026verifier,
	title={Verifier-Constrained Flow Expansion for Discovery Beyond the Data},
	author={De Santi*, Riccardo and Protopapas*, Kimon and Hsieh, Ya-Ping and Krause, Andreas},
	booktitle={International Conference on Learning Representations (ICLR)},
	year={2026},
	pdf={https://arxiv.org/pdf/2602.15984},
}
```

References for verifier-free flow expansion methods:

```
@inproceedings{de2025flow,
 	author = {De Santi, Riccardo and Vlastelica, Marin and Hsieh, Ya-Ping and Shen, Zebang and He, Niao and Krause, Andreas},
 	booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
 	pdf = {https://www.arxiv.org/abs/2511.22640},
 	title = {Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning},
 	year = {2025}
}

@inproceedings{de2025provable,
	author = {De Santi*, Riccardo and Vlastelica*, Marin and Hsieh, Ya-Ping and Shen, Zebang and He, Niao and Krause, Andreas},
	booktitle = {Proc. International Conference on Machine Learning (ICML)},
	pdf = {https://arxiv.org/pdf/2506.15385},
	title = {Provable Maximum Entropy Manifold Exploration via Diffusion Models},
	year = {2025}
}
```
