from diffusiongym.rewards import Reward
from diffusiongym.types import D
import torch


class Constraint(Reward[D]):
    """A reward with a differentiable soft form and a binary hard form.

    Subclasses return (soft, hard) from __call__, where:
    - soft : differentiable approximation in [0, 1] (used by the AM expand step)
    - hard : binary feasibility indicator {0, 1} (used by the DDPO project step)
    """

    def grad_log_soft(self, x: D) -> D:
        """Gradient of log(soft(x)) w.r.t. x, for use in adjoint matching."""
        x_req = x.requires_grad()
        soft, _ = self(x_req, x_req)
        return x_req.gradient(torch.log(soft.clamp(min=1e-8)))
