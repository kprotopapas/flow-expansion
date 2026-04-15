import torch

class AugmentedReward:
    def __init__(
        self,
        grad_reward_fn: callable,          # returns ∇r(x) with same shape as x
        grad_constraint_fn: callable,      # returns ∇c(x) with same shape as x
        constraint_fn: callable,           # returns c(x) with shape (B,) or (B,1) for logging/gating
        reward_lambda: float,
        device: torch.device = None,
    ):
        self.grad_reward_fn = grad_reward_fn
        self.grad_constraint_fn = grad_constraint_fn
        self.constraint_fn = constraint_fn
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_lambda = float(reward_lambda)

        # ALM multipliers
        self.lambda_ = 0.0   # usually ≤ 0
        self.rho_ = 1.0      # > 0

        # caches for stats
        self.tmp_reward = None          # unknown (no reward value)
        self.tmp_constraint = None
        self.tmp_total = None

    def set_lambda_rho(self, lambda_: float, rho_: float):
        self.lambda_ = float(lambda_)
        self.rho_ = float(rho_)

    @torch.no_grad()
    def get_reward_constraint(self):
        # reward value is unknown -> NaN
        ret_reward = float("nan")
        if self.tmp_constraint is None:
            return ret_reward, float("nan"), float("nan")
        ret_constraint = self.tmp_constraint.detach().mean().cpu().item()
        ret_constraint_violations = (self.tmp_constraint > 0).float().mean().cpu().item()
        return ret_reward, ret_constraint, ret_constraint_violations

    @torch.no_grad()
    def augmented_reward(self, x: torch.Tensor):
        """
        Penalty-only surrogate for logging:
        total = -(rho/2) * ReLU(c - lambda/rho)^2
        Then scaled by reward_lambda.
        """
        x = x.to(self.device)
        c = self.constraint_fn(x)  # (B,) or (B,1)
        self.tmp_constraint = c
        # store a dummy reward tensor so existing code doesn't break
        self.tmp_reward = torch.zeros_like(c)

        h = c - (self.lambda_ / self.rho_)
        relu_h = torch.relu(h)
        penalty = 0.5 * self.rho_ * (relu_h ** 2)   # (B,) or (B,1)
        total = (-penalty).mean()
        self.tmp_total = total.detach()
        return self.reward_lambda * total

    def grad_augmented_reward_fn(self, x: torch.Tensor):
        """
        Returns ∇ J(x) where:
        J(x) = reward_lambda * [ E r(x) - (rho/2) * ReLU(c(x) - lambda/rho)^2 ]
        Using only ∇r(x) and ∇c(x).
        """
        x = x.to(self.device)

        # gradients provided by user
        grad_r = self.grad_reward_fn(x)          # same shape as x
        grad_c = self.grad_constraint_fn(x)      # same shape as x

        # values for gating the penalty and for stats
        c = self.constraint_fn(x)                # (B,) or (B,1)
        self.tmp_constraint = c

        # compute rho * ReLU(c - lambda/rho) with broadcasting over x's trailing dims
        h = c - (self.lambda_ / self.rho_)       # (B,) or (B,1)
        relu_h = torch.relu(h)                   # (B,) or (B,1)
        # make relu_h broadcastable to x
        while relu_h.dim() < x.dim():
            relu_h = relu_h.unsqueeze(-1)

        penalty_grad = self.rho_ * relu_h * grad_c    # same shape as x
        total_grad = self.reward_lambda * (grad_r - penalty_grad)

        # cache a penalty-only total for logging (no reward value available)
        with torch.no_grad():
            pen_scalar = (0.5 * self.rho_ * torch.relu(h) ** 2).mean()
            self.tmp_total = (-pen_scalar).detach()
            # dummy reward tensor for compatibility
            self.tmp_reward = torch.zeros_like(c)

        return total_grad

    @torch.no_grad()
    def get_statistics(self):
        reward_val = float("nan")  # unknown
        if self.tmp_constraint is None or self.tmp_total is None:
            return {
                "reward": reward_val,
                "constraint": float("nan"),
                "total_reward": float("nan"),
                "constraint_violations": float("nan"),
            }
        constraint = self.tmp_constraint.detach()
        constraint_mean = constraint.mean().cpu().item()
        constraint_violations = (constraint > 0).float().mean().cpu().item()
        total_reward = self.tmp_total.detach().cpu().item()
        return {
            "reward": reward_val,
            "constraint": constraint_mean,
            "total_reward": total_reward,
            "constraint_violations": constraint_violations,
        }
