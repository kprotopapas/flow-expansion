import copy
from typing import Callable, Optional, Union
import torch

class AugmentedLagrangian:
    """
    Augmented Lagrangian manager without external config.

    Args:
        constraint_fn: function mapping samples -> constraint values (positive means violation).
        rho_max: maximum allowed penalty parameter (used as an upper cap when you choose to cap).
        eta: multiplicative factor for increasing rho (e.g., 1.5 or 2.0).
        rho_init: initial rho (default 0.5).
        lambda_min: lower bound for lambda (interpreted as a non-positive bound; value is negated & abs'ed).
        tau: contraction threshold in (0,1); controls whether to grow rho.
        baseline: if True, keep lambda fixed to base_lambda and rho=0.0.
        base_lambda: fixed lambda used in baseline mode (forced to be non-positive).
        device: torch device or string (e.g., "cuda", "cpu"). Defaults to CUDA if available.
    """

    def __init__(
        self,
        constraint_fn: Callable,
        rho_max: float,
        eta: float,
        rho_init: float = 0.5,
        lambda_min: float = -10.0,
        tau: float = 0.99,
        baseline: bool = False,
        base_lambda: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        # Hyperparameters
        self.rho_init = float(rho_init)
        self.rho_max = float(rho_max)
        self.lambda_min = -abs(float(lambda_min))
        self.tau = float(tau)
        self.eta = float(eta)

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        # Models / functions
        self.constraint_fn = constraint_fn

        # ALM state
        self.lambda_ = 0.0
        self.rho = self.rho_init
        self.contraction_value = None
        self.old_contraction_value = None

        # Stats (defined lazily)
        self.constraint_violations = None
        self.exp_constraint = None

        # Baseline mode
        self.baseline = bool(baseline)
        if self.baseline:
            self.base_lambda = -abs(float(base_lambda))
            self.lambda_ = self.base_lambda
            self.rho = 0.0

    def set_constraint_fn(self, constraint_fn: Callable):
        self.constraint_fn = constraint_fn

    def get_current_lambda_rho(self):
        return copy.deepcopy(self.lambda_), copy.deepcopy(self.rho)

    def expected_constraint(self, new_samples: torch.Tensor):
        constraint = self.constraint_fn(new_samples)
        self.constraint_violations = (torch.sum(constraint > 0).detach().cpu().item()
                                      / max(1, len(new_samples)))
        self.exp_constraint = torch.mean(constraint).detach().cpu().item()
        if not self.baseline:
            # contraction_value = min(-lambda/rho, E[g(x)])
            # (rho should be > 0 in non-baseline mode by construction)
            self.contraction_value = min(-self.lambda_ / self.rho, self.exp_constraint)
        return copy.deepcopy(self.exp_constraint)

    def update_lambda(self, new_samples: torch.Tensor):
        # lambda_{k+1} = min(0, lambda_k - rho_k * E[g(x)])
        ec = self.expected_constraint(new_samples)
        lambda_suggestion = self.lambda_ - self.rho * ec
        lambda_ = min(0.0, lambda_suggestion)
        lambda_ = max(lambda_, self.lambda_min)
        return lambda_

    def update_rho(self):
        # Grow rho only if contraction hasn't improved enough.
        # Original behavior kept prints; preserved here for parity.
        if self.old_contraction_value is None:
            rho = self.rho
            print("k = 1")
        elif self.old_contraction_value < self.tau * self.contraction_value:
            rho = self.rho
            print("k =/= 1 and old_contraction_value < tau * contraction_value")
        else:
            rho = self.eta * self.rho
            print("eta * rho")

        self.old_contraction_value = self.contraction_value

        # If you want a hard cap, uncomment the next line:
        # rho = min(rho, self.rho_max)
        return rho

    def update_lambda_rho(self, new_samples: torch.Tensor):
        self.lambda_ = self.update_lambda(new_samples)
        self.rho = self.update_rho()
        if self.baseline:
            self.lambda_ = self.base_lambda
            self.rho = 0.0

    def get_statistics(self):
        return {
            "lambda": copy.deepcopy(self.lambda_),
            "rho": copy.deepcopy(self.rho),
            "expected_constraint": None if self.exp_constraint is None else copy.deepcopy(self.exp_constraint),
            "constraint_violations": None if self.constraint_violations is None else copy.deepcopy(self.constraint_violations),
        }
