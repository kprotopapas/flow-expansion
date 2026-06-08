# Verifier-Constrained Flow Expansion for Discovery Beyond the Data


[![arXiv](http://img.shields.io/badge/arxiv-2602.15984-red?logo=arxiv)](https://arxiv.org/abs/2602.15984)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Komod0D/flow-expansion/blob/main/tutorial.ipynb)

This repository contains the official implementation of the Flow Expansion algorithm, a method for verifier-constrained exploration for discovery beyond the data.

## Installation

Either use pip:

```bash
pip install -e .
```

Or first install `uv` here: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

Then run:

```bash
uv sync
```

## Overview

Flow Expansion is built on top of [diffusiongym](https://github.com/cristianpjensen/diffusiongym), a library for reward adaptation of pre-trained flow models across any data modality. To run Flow Expansion on your own model you need three things:

1. A **data type** (e.g. `DDTensor` for plain tensors, or a custom `DDMixin` subclass for structured data)
2. A **base model** (`BaseModel[D]`) wrapping your pre-trained network
3. A **reward** (`Reward[D]`) measuring sample quality

diffusiongym then handles environment construction, SDE simulation, and trajectory storage. `FlowExpansionTrainer` runs the optimization loop on top.

## Quickstart

Check `tutorial.ipynb` for a complete worked example on a toy 1D trimodal GMM:

```python
import torch, diffusiongym
from omegaconf import OmegaConf
from genexp import FlowExpansionTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env.reward is a SigmoidalConstraint — automatically enables the project step
env = diffusiongym.make(
    base_model="1d/trimodal_gmm",
    reward="1d/sigmoidal",
    discretization_steps=50,
    device=device,
)

config = OmegaConf.create({
    "gamma": 1.0, "eta": 1.0, "beta": 0.0, "epsilon": 0.01,
    "traj": True, "lmbda": "const",
    "adjoint_matching": {
        "lr": 1e-4, "batch_size": 128, "num_iterations": 2,
        "finetune_steps": 50, "sampling": {"num_samples": 512},
    },
    "ddpo": {
        "lr": 1e-4, "batch_size": 128, "num_iterations": 2,
        "finetune_steps": 50, "sampling": {"num_samples": 512},
    },
})

trainer = FlowExpansionTrainer(config, env, device=device)
losses = trainer.fit(num_iterations=3)
```

## Usage

### 1. Data type

For plain tensor data, use the built-in `DDTensor`:

```python
from diffusiongym import DDTensor

x = DDTensor(torch.randn(batch_size, dim))
```

For structured data (graphs, molecules, images with conditioning), use one of the existing types or subclass `DDMixin` and implement `apply`, `combine`, `aggregate`, `collate`, `__len__`, and `__getitem__`. See [diffusiongym's types documentation](https://cristianpjensen.github.io/diffusiongym/) for details.

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

`FlowExpansionTrainer` converts all outputs to a score function internally using the scheduler. The `scheduler` defines the interpolant `x_t = α_t x_1 + β_t x_0`. `OptimalTransportScheduler` uses the linear schedule `α_t = t`, `β_t = 1 − t`. `CosineScheduler` and `DiffusionScheduler` are also available.

The base class provides `train_loss(x1)` automatically once `output_type` and `scheduler` are set, so you can train your model with:

```python
import diffusiongym
diffusiongym.train_base_model(model, optimizer, data, steps=10_000)
```

### 3. Constraint

`FlowExpansionTrainer` expects the environment's reward to be a `Constraint` — a subclass of `Reward[D]` that formalises the split between a **soft** (differentiable) and a **hard** (binary) form of the constraint:

| Return value | Meaning | Used by |
|---|---|---|
| `soft` | Differentiable approximation in [0, 1] | Adjoint-matching expand step |
| `hard` | Binary feasibility indicator {0, 1} | DDPO project step |

```python
from genexp import Constraint
from diffusiongym.types import DDTensor

class MyConstraint(Constraint[DDTensor]):
    def __call__(self, sample: DDTensor, latent: DDTensor, **kwargs):
        score = some_verifier(sample.data)        # differentiable score in [0, 1]
        soft  = torch.sigmoid(score)
        hard  = (score > 0.5).float()
        return soft, hard
```

The base class provides `grad_log_soft(x)` — the gradient of `log(soft(x))` w.r.t. `x` — for use by the expand step. If your reward is not a `Constraint` (e.g. a plain `Reward`), the project step is skipped.

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

If your model and constraints are registered in diffusiongym's registries you can also use the `make()` factory:

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

Alternatively, instantiate the environment class directly:

```python
from diffusiongym import VelocityEnvironment   # or Score/Epsilon/EndpointEnvironment

env = VelocityEnvironment(model, MyReward(), discretization_steps=100)
```

### 5. Flow Expansion trainer

Pass the environment directly to `FlowExpansionTrainer`. The trainer creates its own deep copies of `env.base_model` for the fine and reference models, and auto-detects the constraint from `env.reward`:

```python
from omegaconf import OmegaConf
from genexp import FlowExpansionTrainer

config = OmegaConf.create({
    "gamma": 1.0,       # score-weighting strength for the expand step
    "eta": 1.0,         # projection step weight (set 0 to skip projection)
    "beta": 0.0,        # KL subtraction coefficient
    "epsilon": 0.01,    # clipping for t → 1
    "traj": True,       # trajectory-level adjoint (recommended)
    "lmbda": "const",   # lambda schedule: "const" or "variance"
    "adjoint_matching": {
        "lr": 1e-4,
        "batch_size": 128,
        "num_iterations": 2,        # AM rounds per expand step
        "finetune_steps": 50,
        "sampling": {"num_samples": 512},
    },
    # Include "ddpo" to enable the constraint projection step.
    # Requires env.reward to be a Constraint subclass.
    "ddpo": {
        "lr": 1e-4,
        "batch_size": 128,
        "num_iterations": 2,        # DDPO rounds per project step
        "finetune_steps": 50,
        "sampling": {"num_samples": 512},
    },
})

trainer = FlowExpansionTrainer(config, env, device=device)
```

Then run the mirror-descent loop with `fit`. Each iteration consists of an **expand** step (adjoint matching toward higher reward) followed by a **project** step (DDPO toward constraint satisfaction), both sharing the same fine-tuned model:

```python
losses = trainer.fit(num_iterations=10)
```

`fit` returns a flat list of per-round losses (AM losses from the expand step, then DDPO losses from the project step, for each iteration).

The expand step uses the score function of the current base model as the reward signal. The project step uses DDPO with `env.reward`'s hard (binary) output as the reward, so it trains the model to satisfy the constraint. If `env.reward` is not a `Constraint`, or if the `ddpo` config block is absent, the project step is skipped.


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

Reference for the [diffusiongym](https://github.com/cristianpjensen/diffusiongym) library:

```
@inproceedings{jensen2026value,
  title={Value Matching: Scalable and Gradient-Free Reward-Guided Flow Adaptation},
  author={Cristian Perez Jensen and Luca Schaufelberger and Riccardo De Santi and Kjell Jorner and Andreas Krause},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
```
