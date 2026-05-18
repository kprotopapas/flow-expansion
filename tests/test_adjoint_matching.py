"""
Closed-form and reference-equivalence tests for LeanAdjointSolverFlow and adj_matching_loss.

Adjoint ODE (zero-velocity case, OT scheduler)
-----------------------------------------------
For v = 0, the adjoint Euler step simplifies to:

    a(t - dt) = a(t) · t / (t + dt)

because eps_pred = -x / (alpha + dt) = -x / (t + dt), so the Jacobian
w.r.t. x is -I / (t + dt), giving the update factor (1 - dt/(t + dt)) = t/(t + dt).

Telescoping from a(1) = -∇R(x₁) through all T-1 steps:

    a(t_k) = t_k · a(1) = -t_k · x₁   (exact, no discretization error)

Adjoint matching loss optimum
-----------------------------
The loss is: E[(2/σ · (v_fine - v_base) - σ · adj)²]

Setting the bracket to zero gives the optimal perturbation:
    v_fine* = v_base + (σ²/2) · adj   →   loss = 0

Reference-equivalence tests
----------------------------
The current implementation (DDTensor-based) is compared against a raw-tensor
reference that mirrors Adjoint_Matching_Mols/finetuning/flow_adjoint_solver.py.
The only structural differences are:
  - type: DDTensor vs raw torch.Tensor
  - g_term: .aggregate("sum").sum() vs .sum()  [identical for 2D tensors]
  - gradient: DDTensor.gradient() vs torch.autograd.grad()  [same computation]
  - result ordering: new reverses to forward-time; old stays backward-time
"""

import pytest
import torch
from diffusiongym.schedulers import OptimalTransportScheduler
from diffusiongym.types import DDTensor

from genexp.trainers.adjoint_matching import LeanAdjointSolverFlow, adj_matching_loss

BATCH = 4
DATA_DIM = 3
T = 6      # number of timesteps including t=0 and t=1
TOL = 1e-5


# ---------------------------------------------------------------------------
# Minimal oracle model
# ---------------------------------------------------------------------------

class ZeroVelocityModel:
    """Velocity-predicting model that always returns zero."""
    output_type = "velocity"

    def __init__(self):
        self.scheduler = OptimalTransportScheduler()

    def forward(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        return x.zeros_like()


@pytest.fixture
def zero_vel_solver():
    # grad_reward_fn = identity → ∇R(x) = x, i.e. R(x) = 0.5|x|²
    return LeanAdjointSolverFlow(
        ZeroVelocityModel(),
        grad_reward_fn=lambda x: x,
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Adjoint trajectory tests
# ---------------------------------------------------------------------------

def test_adjoint_trajectory_zero_velocity(zero_vel_solver):
    """For v=0 and R(x) = 0.5|x|², the adjoint is a(t) = -t·x₁ exactly."""
    torch.manual_seed(0)
    ts = torch.linspace(0, 1, T)
    x1 = DDTensor(torch.randn(BATCH, DATA_DIM))

    result = zero_vel_solver.solve([x1] * T, ts)
    t_out = result["t"]            # ts[:-1], shape (T-1,)
    traj_adj = result["traj_adj"]  # T-1 DDTensors in forward-time order

    assert len(traj_adj) == T - 1

    for k in range(T - 1):
        t_k = t_out[k].item()
        expected = -t_k * x1.data
        assert torch.allclose(traj_adj[k].data, expected, atol=TOL), (
            f"Adjoint wrong at t={t_k:.3f}: "
            f"max err = {(traj_adj[k].data - expected).abs().max():.2e}"
        )


def test_velocity_prediction_stored_correctly(zero_vel_solver):
    """traj_v_pred entries are zero for a zero-velocity model."""
    torch.manual_seed(1)
    ts = torch.linspace(0, 1, T)
    x1 = DDTensor(torch.randn(BATCH, DATA_DIM))

    result = zero_vel_solver.solve([x1] * T, ts)
    traj_v = result["traj_v_pred"]

    assert len(traj_v) == T - 1
    for k, v_k in enumerate(traj_v):
        assert torch.allclose(v_k.data, torch.zeros_like(v_k.data), atol=TOL), (
            f"v_pred nonzero at index {k}: max = {v_k.data.abs().max():.2e}"
        )


def test_adjoint_at_t0_is_zero(zero_vel_solver):
    """a(0) = 0 regardless of x₁, because the factor t/(t+dt) = 0 at t=0."""
    torch.manual_seed(2)
    ts = torch.linspace(0, 1, T)
    x1 = DDTensor(torch.randn(BATCH, DATA_DIM) * 10)  # large x₁

    result = zero_vel_solver.solve([x1] * T, ts)
    a0 = result["traj_adj"][0]  # t_out[0] = 0

    assert result["t"][0].item() == pytest.approx(0.0)
    assert torch.allclose(a0.data, torch.zeros_like(a0.data), atol=TOL)


def test_adjoint_scales_linearly_with_x1(zero_vel_solver):
    """Scaling x₁ by a constant scales the entire adjoint trajectory by the same constant."""
    torch.manual_seed(3)
    ts = torch.linspace(0, 1, T)
    x1 = DDTensor(torch.randn(BATCH, DATA_DIM))
    scale = 3.7

    r1 = zero_vel_solver.solve([x1] * T, ts)
    r2 = zero_vel_solver.solve([DDTensor(x1.data * scale)] * T, ts)

    for k in range(T - 1):
        assert torch.allclose(r2["traj_adj"][k].data, r1["traj_adj"][k].data * scale, atol=TOL), (
            f"Linearity failed at k={k}"
        )


def test_grad_fk_shifts_adjoint(zero_vel_solver):
    """With a constant grad_f_k = c, the adjoint at t=0 equals -dt·c (exact)."""
    torch.manual_seed(4)
    ts = torch.linspace(0, 1, T)
    x1 = DDTensor(torch.randn(BATCH, DATA_DIM))
    dt = ts[1] - ts[0]

    c_val = torch.ones(BATCH, DATA_DIM) * 2.0
    c = DDTensor(c_val)
    solver_fk = LeanAdjointSolverFlow(
        ZeroVelocityModel(),
        grad_reward_fn=lambda x: x,
        grad_f_k_fn=lambda x, t: c,
        device=torch.device("cpu"),
    )
    result = solver_fk.solve([x1] * T, ts)

    # At t=0 the multiplicative factor is exactly 0, so any adj from prior steps
    # is killed. Only the -dt*c subtraction at the last step survives.
    a0 = result["traj_adj"][0]
    expected = -dt.item() * c_val
    assert torch.allclose(a0.data, expected, atol=TOL), (
        f"a(0) with const grad_fk: max err = {(a0.data - expected).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# adj_matching_loss tests
# ---------------------------------------------------------------------------

def test_adj_matching_loss_zero_at_optimum():
    """Loss is 0 at the optimal perturbation v_fine = v_base + (σ²/2)·adj."""
    torch.manual_seed(5)
    adj    = DDTensor(torch.randn(BATCH, DATA_DIM))
    sigma  = DDTensor(torch.rand(BATCH, DATA_DIM) + 0.1)
    v_base = DDTensor(torch.randn(BATCH, DATA_DIM))
    v_fine = v_base + sigma ** 2 * adj * 0.5

    loss = adj_matching_loss(v_base, v_fine, adj, sigma)
    assert loss.abs() < TOL, f"Expected 0, got {loss:.2e}"


def test_adj_matching_loss_at_base_model():
    """When v_fine = v_base, loss = mean_batch sum_dim (σ·adj)²."""
    torch.manual_seed(6)
    adj    = DDTensor(torch.randn(BATCH, DATA_DIM))
    sigma  = DDTensor(torch.rand(BATCH, DATA_DIM) + 0.1)
    v_base = DDTensor(torch.randn(BATCH, DATA_DIM))

    loss = adj_matching_loss(v_base, v_base, adj, sigma)

    dims = tuple(range(1, adj.data.ndim))
    expected = ((sigma.data * adj.data) ** 2).sum(dim=dims).mean()
    assert torch.allclose(loss, expected, atol=TOL), (
        f"Loss at base model: got {loss:.6f}, expected {expected:.6f}"
    )


def test_adj_matching_loss_is_nonnegative():
    """Loss is always >= 0."""
    torch.manual_seed(7)
    for _ in range(10):
        adj    = DDTensor(torch.randn(BATCH, DATA_DIM))
        sigma  = DDTensor(torch.rand(BATCH, DATA_DIM) + 0.1)
        v_base = DDTensor(torch.randn(BATCH, DATA_DIM))
        v_fine = DDTensor(torch.randn(BATCH, DATA_DIM))
        assert adj_matching_loss(v_base, v_fine, adj, sigma) >= 0


# ---------------------------------------------------------------------------
# Reference-equivalence tests (vs Adjoint_Matching_Mols raw-tensor impl)
# ---------------------------------------------------------------------------

class LinearVelocityModel:
    """velocity model: v(x,t) = x @ A.T for a fixed matrix A."""
    output_type = "velocity"

    def __init__(self, A: torch.Tensor):
        self.A = A
        self.scheduler = OptimalTransportScheduler()

    def forward(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        return DDTensor(x.data @ self.A.T)


def _ref_step(adj_raw, x_raw, A, scheduler, t_batch, dt):
    """Raw-tensor adjoint step mirroring flow_adjoint_solver.py:

        eps_pred = 2*v_pred - alpha_dot/(alpha+dt) * x_t
        g_term   = (adj * eps_pred).sum()            # sum over ALL elements
        v        = autograd.grad(g_term, x_t)[0]
        adj_new  = adj + dt * v
    """
    x_grad = x_raw.detach().requires_grad_(True)
    x_dd = DDTensor(x_grad)

    alpha     = scheduler.alpha(x_dd, t_batch).data      # (batch, 1)
    alpha_dot = scheduler.alpha_dot(x_dd, t_batch).data  # (batch, 1)

    v_pred   = x_grad @ A.T                              # linear velocity
    eps_pred = 2 * v_pred - alpha_dot / (alpha + dt) * x_grad
    g_term   = (adj_raw.detach() * eps_pred).sum()       # scalar
    v        = torch.autograd.grad(g_term, x_grad)[0]

    return (adj_raw.detach() + dt * v).detach()


def test_single_step_matches_reference():
    """One adjoint step: DDTensor .gradient() == torch.autograd.grad()."""
    torch.manual_seed(20)
    A = torch.randn(DATA_DIM, DATA_DIM) * 0.1
    scheduler = OptimalTransportScheduler()

    adj = DDTensor(torch.randn(BATCH, DATA_DIM))
    x_t = DDTensor(torch.randn(BATCH, DATA_DIM))
    t_batch = torch.full((BATCH,), 0.6)
    dt = torch.tensor(0.2)

    model = LinearVelocityModel(A)
    solver = LeanAdjointSolverFlow(model, grad_reward_fn=lambda x: x, device=torch.device("cpu"))
    adj_new, _ = solver.step(adj, x_t, t_batch, dt)

    adj_ref = _ref_step(adj.data, x_t.data, A, scheduler, t_batch, dt)

    assert torch.allclose(adj_new.data, adj_ref, atol=TOL), (
        f"Step mismatch: max err = {(adj_new.data - adj_ref).abs().max():.2e}"
    )


def test_full_trajectory_matches_reference():
    """Full solve: new (DDTensor, forward-order) == reference (raw tensor, backward-order reversed)."""
    torch.manual_seed(21)
    T_local = 6
    A = torch.randn(DATA_DIM, DATA_DIM) * 0.1
    scheduler = OptimalTransportScheduler()

    ts = torch.linspace(0, 1, T_local)
    dt = ts[1] - ts[0]
    ts_rev = ts.flip(0)

    trajectories = [DDTensor(torch.randn(BATCH, DATA_DIM)) for _ in range(T_local)]

    model = LinearVelocityModel(A)
    solver = LeanAdjointSolverFlow(model, grad_reward_fn=lambda x: x, device=torch.device("cpu"))
    result = solver.solve(trajectories, ts)

    # --- Reference: raw-tensor backward pass ---
    adj_ref = -trajectories[-1].data.clone()  # a(1) = -x₁
    trajs_adj_ref = []
    for i in range(1, T_local):
        t_batch = ts_rev[i].unsqueeze(0).expand(BATCH)
        x_raw = trajectories[T_local - i - 1].data
        adj_ref = _ref_step(adj_ref, x_raw, A, scheduler, t_batch, dt)
        trajs_adj_ref.append(adj_ref)

    # Reference is in backward-time order; new code is in forward-time order.
    trajs_adj_ref_fwd = list(reversed(trajs_adj_ref))

    for k in range(T_local - 1):
        new_adj = result["traj_adj"][k].data
        ref_adj = trajs_adj_ref_fwd[k]
        assert torch.allclose(new_adj, ref_adj, atol=TOL), (
            f"Trajectory mismatch at k={k}: max err = {(new_adj - ref_adj).abs().max():.2e}"
        )


def test_loss_formula_matches_reference():
    """adj_matching_loss DDTensor impl == raw-tensor reference from flow_adjoint.py."""
    torch.manual_seed(22)

    v_base_raw = torch.randn(BATCH, DATA_DIM)
    v_fine_raw = torch.randn(BATCH, DATA_DIM)
    adj_raw    = torch.randn(BATCH, DATA_DIM)
    # σ must be same shape as x for DDTensor; old code used scalar-per-timestep
    # broadcast via sigma[:,None,None] — replicate with uniform per-element σ.
    sigma_val  = 0.5
    sigma_raw  = torch.full((BATCH, DATA_DIM), sigma_val)

    # Reference formula (mirrors flow_adjoint.py adj_matching_loss):
    diff         = v_fine_raw - v_base_raw
    term_diff    = (2 / sigma_raw) * diff
    term_adj     = sigma_raw * adj_raw
    loss_ref     = ((term_diff - term_adj) ** 2).sum(dim=tuple(range(1, adj_raw.ndim))).mean()

    # New DDTensor formula:
    loss_new = adj_matching_loss(
        DDTensor(v_base_raw), DDTensor(v_fine_raw),
        DDTensor(adj_raw),    DDTensor(sigma_raw),
    )

    assert torch.allclose(loss_new, loss_ref, atol=TOL), (
        f"Loss mismatch: new={loss_new:.6f}, ref={loss_ref:.6f}"
    )
