"""
Tests for the main genexp workflow.
Part 1: legacy tensor models (DiffusionModel, VPSDE, EulerMaruyamaSampler).
Part 2: diffusiongym-based FlowExpansionTrainer.
All tests run on CPU with a tiny 2-D network for speed.
"""

import copy
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from typing import Any, Optional

from diffusiongym.base_models import BaseModel
from diffusiongym.environments import VelocityEnvironment
from diffusiongym.rewards import DummyReward
from diffusiongym.schedulers import OptimalTransportScheduler
from diffusiongym.types import DDTensor

from genexp.models import DiffusionModel, VPSDE
from genexp.sampling import EulerMaruyamaSampler
from genexp.trainers.genexp import FlowExpansionTrainer


DATA_DIM = 2
BATCH = 4
STEPS = 5  # discretization steps for env


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def make_network():
    return nn.Sequential(
        nn.Linear(DATA_DIM + 1, 16),
        nn.ReLU(),
        nn.Linear(16, DATA_DIM),
    )


def make_diffusion_model(device="cpu"):
    sde = VPSDE(0.1, 12, device=device)
    return DiffusionModel(make_network(), sde).to(device)


# ---------------------------------------------------------------------------
# Tiny velocity BaseModel[DDTensor] for diffusiongym tests
# ---------------------------------------------------------------------------

class TinyVelocityModel(BaseModel[DDTensor]):
    output_type = "velocity"

    def __init__(self, dim: int, device: Optional[torch.device] = None):
        super().__init__(device or torch.device("cpu"))
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 16), nn.ReLU(), nn.Linear(16, dim)
        )
        self._scheduler = OptimalTransportScheduler()

    @property
    def scheduler(self):
        return self._scheduler

    def sample_p0(self, n: int, **kwargs: Any):
        x = DDTensor(torch.randn(n, DATA_DIM, device=self.device))
        return x, kwargs

    def forward(self, x: DDTensor, t: torch.Tensor, **kwargs: Any) -> DDTensor:
        t_in = t.unsqueeze(1) if t.ndim == 1 else t
        out = self.net(torch.cat([x.data, t_in], dim=1))
        return DDTensor(out)


def make_velocity_model(device="cpu"):
    return TinyVelocityModel(DATA_DIM, torch.device(device))


def make_env(base_model, steps=STEPS):
    return VelocityEnvironment(base_model, DummyReward(), discretization_steps=steps)


def make_fe_config():
    return OmegaConf.create({
        "gamma": 0.1,
        "eta": 0.1,
        "epsilon": 0.005,
        "beta": 0.0,
        "traj": True,
        "lmbda": "const",
        "adjoint_matching": {
            "batch_size": BATCH,
            "clip_grad_norm": 0.4,
            "clip_loss": 1e5,
            "lr": 0.01,
            "sampling": {
                "num_samples": BATCH,
            },
        },
    })


# ---------------------------------------------------------------------------
# Part 1: legacy tensor models
# ---------------------------------------------------------------------------

def test_vpsde_alpha_sigma_shape():
    sde = VPSDE(0.1, 12, device="cpu")
    t = torch.linspace(0, 1, BATCH).unsqueeze(1)
    alpha, sigma = sde.get_alpha_sigma(t)
    assert alpha.shape == (BATCH, 1)
    assert sigma.shape == (BATCH, 1)


def test_vpsde_alpha_sigma_range():
    sde = VPSDE(0.1, 12, device="cpu")
    t = torch.linspace(0, 1, 10).unsqueeze(1)
    alpha, sigma = sde.get_alpha_sigma(t)
    assert (alpha >= 0).all() and (alpha <= 1).all()
    assert (sigma >= 0).all() and (sigma <= 1).all()


def test_diffusion_model_forward_shape():
    model = make_diffusion_model()
    x = torch.randn(BATCH, DATA_DIM)
    t = torch.rand(BATCH, 1)
    out = model(x, t)
    assert out.shape == (BATCH, DATA_DIM)


def test_diffusion_model_velocity_field_shape():
    model = make_diffusion_model()
    x = torch.randn(BATCH, DATA_DIM)
    t = torch.rand(BATCH, 1)
    v = model.velocity_field(x, t)
    assert v.shape == (BATCH, DATA_DIM)


def test_diffusion_model_score_func_shape():
    model = make_diffusion_model()
    x = torch.randn(BATCH, DATA_DIM)
    t = torch.rand(BATCH, 1).clamp(0.01, 0.99)
    s = model.score_func(x, t)
    assert s.shape == (BATCH, DATA_DIM)


def test_sampler_trajectory_length():
    model = make_diffusion_model()
    sampler = EulerMaruyamaSampler(model, data_shape=(DATA_DIM,), device="cpu")
    T = 6
    trajs, ts = sampler.sample_trajectories(N=BATCH, T=T)
    assert len(trajs) == T
    assert ts.shape == (T,)


def test_sampler_trajectory_shape():
    model = make_diffusion_model()
    sampler = EulerMaruyamaSampler(model, data_shape=(DATA_DIM,), device="cpu")
    trajs, _ = sampler.sample_trajectories(N=BATCH, T=6)
    assert trajs[0].full.shape == (BATCH, DATA_DIM)


def test_sampler_no_nans():
    model = make_diffusion_model()
    sampler = EulerMaruyamaSampler(model, data_shape=(DATA_DIM,), device="cpu")
    trajs, _ = sampler.sample_trajectories(N=BATCH, T=6)
    for s in trajs:
        assert not s.full.isnan().any()


# ---------------------------------------------------------------------------
# Part 2: diffusiongym-based FlowExpansionTrainer
# ---------------------------------------------------------------------------

@pytest.fixture
def fe_trainer():
    device = torch.device("cpu")
    base_model = make_velocity_model(device)
    fine_model = copy.deepcopy(base_model)
    env = make_env(base_model)
    config = make_fe_config()
    return FlowExpansionTrainer(config, env, fine_model, base_model, device=device)


def test_trainer_init(fe_trainer):
    assert fe_trainer.fine_model is not None
    assert fe_trainer.base_model is not None


def test_trainer_expand(fe_trainer):
    fe_trainer.expand()


def test_trainer_generate_dataset(fe_trainer):
    fe_trainer.expand()
    dataset = fe_trainer.generate_dataset()
    assert dataset is not None
    assert len(dataset) > 0


def test_trainer_finetune(fe_trainer):
    fe_trainer.expand()
    dataset = fe_trainer.generate_dataset()
    losses = fe_trainer.finetune(dataset, steps=2, debug=True)
    assert len(losses) > 0
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


def test_trainer_update_base_model(fe_trainer):
    original_params = {k: v.clone() for k, v in fe_trainer.base_model.named_parameters()}
    with torch.no_grad():
        for p in fe_trainer.fine_model.parameters():
            p.add_(0.1)
    fe_trainer.update_base_model()
    for k, v in fe_trainer.base_model.named_parameters():
        assert not torch.equal(v, original_params[k])


def test_full_tutorial_loop():
    device = torch.device("cpu")
    base_model = make_velocity_model(device)
    fine_model = copy.deepcopy(base_model)
    env = make_env(base_model)
    config = make_fe_config()

    trainer = FlowExpansionTrainer(config, env, fine_model, base_model, device=device)
    initial_params = {k: v.clone() for k, v in trainer.fine_model.named_parameters()}

    for _ in range(2):
        trainer.expand()
        dataset = trainer.generate_dataset()
        trainer.finetune(dataset, steps=2)
        trainer.update_base_model()

    assert any(
        not torch.equal(v, initial_params[k])
        for k, v in trainer.fine_model.named_parameters()
    ), "model weights unchanged after finetuning"