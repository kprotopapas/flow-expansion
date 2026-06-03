"""Training script for a single flow-expansion run.

Accepts all hyperparameters as CLI flags and writes results to --output_dir:
  config.json       — full config used
  losses.npy        — per-round training losses
  base_samples.npy  — 1-D samples from the pre-trained base model
  fine_samples.npy  — 1-D samples from the fine-tuned model
  model.pt          — fine-tuned model state dict
"""

import argparse
import json
import os

import numpy as np
import torch
from vendi_score import vendi

import diffusiongym
from omegaconf import OmegaConf

from genexp.trainers.genexp import FlowExpansionTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)

    # Mirror-descent / top-level
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--traj", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--lmbda", default="const", choices=["const", "variance"])
    p.add_argument("--num_md_iterations", type=int, default=3)

    # Adjoint matching (expand step)
    p.add_argument("--am_lr", type=float, default=1e-4)
    p.add_argument("--am_batch_size", type=int, default=128)
    p.add_argument("--am_num_iterations", type=int, default=2)
    p.add_argument("--am_finetune_steps", type=int, default=50)
    p.add_argument("--am_num_samples", type=int, default=512)

    # DDPO (project step)
    p.add_argument("--ddpo_lr", type=float, default=1e-4)
    p.add_argument("--ddpo_batch_size", type=int, default=128)
    p.add_argument("--ddpo_num_iterations", type=int, default=2)
    p.add_argument("--ddpo_finetune_steps", type=int, default=50)
    p.add_argument("--ddpo_num_samples", type=int, default=512)

    # Environment
    p.add_argument("--discretization_steps", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=5000)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = diffusiongym.make(
        base_model="1d/trimodal_gmm",
        reward="1d/sigmoidal",
        discretization_steps=args.discretization_steps,
        device=device,
    )

    config = OmegaConf.create(
        {
            "gamma": args.gamma,
            "eta": args.eta,
            "beta": args.beta,
            "epsilon": args.epsilon,
            "traj": args.traj,
            "lmbda": args.lmbda,
            "num_md_iterations": args.num_md_iterations,
            "adjoint_matching": {
                "lr": args.am_lr,
                "batch_size": args.am_batch_size,
                "num_iterations": args.am_num_iterations,
                "finetune_steps": args.am_finetune_steps,
                "sampling": {"num_samples": args.am_num_samples},
            },
            "ddpo": {
                "lr": args.ddpo_lr,
                "batch_size": args.ddpo_batch_size,
                "num_iterations": args.ddpo_num_iterations,
                "finetune_steps": args.ddpo_finetune_steps,
                "sampling": {"num_samples": args.ddpo_num_samples},
            },
        }
    )

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(config, resolve=True), f, indent=2)

    with torch.no_grad():
        base_samples = env.sample(args.n_samples, pbar=False).sample.data.cpu().squeeze()
    np.save(os.path.join(args.output_dir, "base_samples.npy"), base_samples.numpy())

    trainer = FlowExpansionTrainer(config, env, device=device)
    losses = trainer.fit(config.num_md_iterations, pbar=True)
    np.save(os.path.join(args.output_dir, "losses.npy"), np.array(losses))

    env.base_model = trainer.fine_model
    with torch.no_grad():
        fine_out = env.sample(args.n_samples, pbar=False)
    env.base_model = trainer.base_base_model

    fine_samples = fine_out.sample.data.cpu().squeeze().numpy()
    np.save(os.path.join(args.output_dir, "fine_samples.npy"), fine_samples)

    valid_mask = fine_out.valids.cpu().numpy() > 0.5
    valid_samples = fine_samples[valid_mask]

    if len(valid_samples) > 1:
        x = valid_samples.reshape(-1, 1)
        dists = np.abs(x - x.T)
        sigma = np.median(dists[dists > 0])
        K = np.exp(-(dists**2) / (2 * sigma**2))
        vs = float(vendi.score_K(K))
    else:
        vs = float("nan")

    metrics = {
        "vendi_score": vs,
        "n_valid": int(valid_mask.sum()),
        "validity_rate": float(valid_mask.mean()),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(trainer.fine_model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print(f"vendi={vs:.3f}  valid={valid_mask.mean():.1%}  → {args.output_dir}")


if __name__ == "__main__":
    main()
