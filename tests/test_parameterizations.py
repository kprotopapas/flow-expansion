"""
Closed-form tests for _score_func and _velocity across all output_type variants.

Ground truth: OT scheduler (α=t, β=1-t) + isotropic Gaussian p_data = N(0,I).
Marginal: p_t = N(0, σ²(t)·I)  where σ²(t) = t² + (1-t)².

The four analytic identities that must hold for any x_t and t ∈ (0,1):
  score*    = -x / σ²
  velocity* = x · (2t-1) / σ²
  endpoint* = E[x₁|x_t] = α·x / σ²
  epsilon*  = E[ε|x_t]  = β·x / σ²

Each OracleModel returns the ground-truth output for a given parameterization.
_score_func and _velocity must recover score* and velocity* regardless of which
parameterization is used.
"""

import pytest
import torch
from diffusiongym.schedulers import OptimalTransportScheduler
from diffusiongym.types import DDTensor

from genexp.trainers.adjoint_matching import _velocity
from genexp.trainers.genexp import _score_func

DATA_DIM = 4
BATCH = 8
TOL = 1e-5


class OracleModel:
    """Minimal model stub returning the analytic ground-truth for each output_type."""

    def __init__(self, output_type: str):
        self.output_type = output_type
        self.scheduler = OptimalTransportScheduler()

    def forward(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        sched = self.scheduler
        alpha = sched.alpha(x, t)      # DDTensor, shape (batch, 1), values = t
        beta = sched.beta(x, t)        # DDTensor, shape (batch, 1), values = 1-t
        var = alpha ** 2 + beta ** 2   # DDTensor, σ²(t) broadcast to (batch, 1)

        if self.output_type == "score":
            return -x / var

        if self.output_type == "velocity":
            return x * (alpha * 2 - 1) / var

        if self.output_type == "endpoint":
            # E[x₁ | x_t] = α·x / σ²
            return x * alpha / var

        if self.output_type == "epsilon":
            # E[ε | x_t] = β·x / σ²
            return x * beta / var

        raise ValueError(f"Unknown output_type {self.output_type!r}")


@pytest.fixture
def inputs():
    torch.manual_seed(42)
    # Avoid t ≈ 0 where κ = α̇/α = 1/t diverges.
    t = torch.rand(BATCH) * 0.6 + 0.2   # uniform on [0.2, 0.8]
    x = DDTensor(torch.randn(BATCH, DATA_DIM))
    return x, t


def _analytic_score(x: DDTensor, t: torch.Tensor) -> DDTensor:
    var = t ** 2 + (1 - t) ** 2                           # shape (batch,)
    var_bd = var.view(BATCH, *([1] * (x.data.ndim - 1)))  # (batch, 1, ...)
    return DDTensor(-x.data / var_bd)


def _analytic_velocity(x: DDTensor, t: torch.Tensor) -> DDTensor:
    var = t ** 2 + (1 - t) ** 2
    var_bd = var.view(BATCH, *([1] * (x.data.ndim - 1)))
    coeff = (2 * t - 1).view(BATCH, *([1] * (x.data.ndim - 1)))
    return DDTensor(x.data * coeff / var_bd)


OUTPUT_TYPES = ["score", "velocity", "endpoint", "epsilon"]


@pytest.mark.parametrize("output_type", OUTPUT_TYPES)
def test_score_func_closed_form(inputs, output_type):
    x, t = inputs
    model = OracleModel(output_type)
    s = _score_func(model, x, t)
    s_gt = _analytic_score(x, t)
    assert torch.allclose(s.data, s_gt.data, atol=TOL), (
        f"_score_func({output_type!r}): max err = {(s.data - s_gt.data).abs().max():.2e}"
    )


@pytest.mark.parametrize("output_type", OUTPUT_TYPES)
def test_velocity_closed_form(inputs, output_type):
    x, t = inputs
    model = OracleModel(output_type)
    v = _velocity(model, x, t)
    v_gt = _analytic_velocity(x, t)
    assert torch.allclose(v.data, v_gt.data, atol=TOL), (
        f"_velocity({output_type!r}): max err = {(v.data - v_gt.data).abs().max():.2e}"
    )
