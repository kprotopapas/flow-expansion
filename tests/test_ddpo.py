"""Tests for DDPOTrainer — mirrors test_workflow.py but for DDPO."""

import copy
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from typing import Any, Optional

from diffusiongym.base_models import BaseModel
from diffusiongym.environments import VelocityEnvironment
from diffusiongym.rewards import DummyReward, Reward
from diffusiongym.schedulers import OptimalTransportScheduler
from diffusiongym.types import DDTensor

from genexp.trainers.ddpo import DDPOTrainer


class NormReward(Reward[DDTensor]):
    """Reward = L2 norm of sample — varies across batch so advantages are non-zero."""

    def __call__(self, sample: DDTensor, latent: DDTensor, **kwargs):
        r = sample.data.norm(dim=-1)
        return r, torch.ones_like(r)


DATA_DIM = 2
BATCH = 4
STEPS = 5


class TinyVelocityModel(BaseModel[DDTensor]):
    output_type = "velocity"

    def __init__(self, dim: int, device: Optional[torch.device] = None):
        super().__init__(device or torch.device("cpu"))
        self.net = nn.Sequential(nn.Linear(dim + 1, 16), nn.ReLU(), nn.Linear(16, dim))
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


def make_model(device="cpu"):
    return TinyVelocityModel(DATA_DIM, torch.device(device))


def make_env(base_model, steps=STEPS):
    return VelocityEnvironment(base_model, DummyReward(), discretization_steps=steps)


def make_ddpo_config():
    return OmegaConf.create(
        {
            "batch_size": BATCH,
            "lr": 1e-3,
            "clip_range": 0.2,
            "adv_clip_max": 10.0,
            "clip_grad_norm": 1.0,
            "num_inner_epochs": 1,
            "timestep_fraction": 1.0,
            "sampling": {
                "num_samples": BATCH,
            },
        }
    )


@pytest.fixture
def ddpo_trainer():
    device = "cpu"
    base_model = make_model(device)
    fine_model = copy.deepcopy(base_model)
    env = make_env(base_model)
    config = make_ddpo_config()
    return DDPOTrainer(config, env, fine_model, device=torch.device(device))


def test_trainer_init(ddpo_trainer):
    assert ddpo_trainer.fine_model is not None
    assert ddpo_trainer.env is not None


def test_sample_trajectories(ddpo_trainer):
    sample = ddpo_trainer.sample_trajectories()
    assert sample is not None
    assert len(sample.trajectory) == STEPS + 1
    assert len(sample.noises) == STEPS
    assert len(sample.diffusions) == STEPS
    assert sample.rewards.shape == (BATCH,)


def test_generate_dataset(ddpo_trainer):
    dataset = ddpo_trainer.generate_dataset()
    assert dataset is not None
    assert len(dataset) > 0


def test_train_step(ddpo_trainer):
    dataset = ddpo_trainer.generate_dataset()
    sample = dataset[0]
    ddpo_trainer.fine_model.train()
    loss = ddpo_trainer.train_step(sample)
    assert torch.isfinite(loss)


def test_finetune(ddpo_trainer):
    dataset = ddpo_trainer.generate_dataset()
    losses = ddpo_trainer.finetune(dataset, steps=2, debug=True)
    assert len(losses) > 0
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


def test_fit_updates_weights():
    device = torch.device("cpu")
    base_model = make_model("cpu")
    fine_model = copy.deepcopy(base_model)
    env = VelocityEnvironment(base_model, NormReward(), discretization_steps=STEPS)
    config = make_ddpo_config()
    config.sampling.num_samples = BATCH

    trainer = DDPOTrainer(config, env, fine_model, device=device)
    initial_params = {k: v.clone() for k, v in trainer.fine_model.named_parameters()}

    losses = trainer.fit(num_iterations=3)

    assert len(losses) == 3
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert any(
        not torch.equal(v, initial_params[k])
        for k, v in trainer.fine_model.named_parameters()
    ), "model weights unchanged after DDPO finetuning"


def test_env_policy_restored_after_sample(ddpo_trainer):
    """env._policy should be None (default) after sample_trajectories."""
    original = ddpo_trainer.env._policy
    ddpo_trainer.sample_trajectories()
    assert ddpo_trainer.env._policy is original
