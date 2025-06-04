#!/usr/bin/env python3
import argparse
import os
import random
import math
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

################################################################################
# Synthetic Quadratic Bilevel Problem (now with stochastic noise)              #
################################################################################

class SyntheticQuadraticProblem:
    """Noisy quadratic bilevel benchmark.

    Lower-level (sample‑averaged):    g(x, y) = E_{ζ}[ 0.5*y^T Q_ζ y + (P_ζ x)^T y ]
    Upper-level (deterministic):      f(x, y) = c^T y + 0.01‖x‖² + 0.01‖y‖²

    Each call to `sample_batch()` draws a fresh (Q_ζ, P_ζ) so gradients are noisy.
    Linear inequality constraints A x + B y ≤ b are handled via a soft ReLU penalty.
    """

    def __init__(self, x_dim: int, y_dim: int, m: int, noise_scale: float = 0.1, seed: int = 0):
        self.x_dim, self.y_dim = x_dim, y_dim
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)

        # Deterministic parts
        L_det = self.rng.standard_normal((y_dim, y_dim))
        self.Q_det = torch.tensor(L_det.T @ L_det + 0.5 * np.eye(y_dim), dtype=torch.float32)
        self.P_det = torch.tensor(self.rng.standard_normal((x_dim, y_dim)) * 0.5, dtype=torch.float32)
        self.A = torch.tensor(self.rng.standard_normal((m, x_dim)) * 0.5, dtype=torch.float32)
        self.B = torch.tensor(self.rng.standard_normal((m, y_dim)) * 0.5, dtype=torch.float32)
        self.b = torch.tensor(self.rng.random(m) * 0.1, dtype=torch.float32)
        c = self.rng.standard_normal(y_dim)
        self.c = torch.tensor(c / np.linalg.norm(c), dtype=torch.float32)
        self.penalty_coeff = 10.0

    # ----- stochastic sampling ------------------------------------------------
    def sample_batch(self):
        Z = self.rng.standard_normal((self.y_dim, self.y_dim))
        Z = 0.5 * (Z + Z.T) * self.noise_scale
        Q = self.Q_det + torch.tensor(Z, dtype=torch.float32)
        P = self.P_det + torch.tensor(self.rng.standard_normal((self.x_dim, self.y_dim)) * self.noise_scale,
                                      dtype=torch.float32)
        return Q, P

    # ----- objectives ---------------------------------------------------------
    def g(self, x: torch.Tensor, y: torch.Tensor, Q: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(y, Q @ y) + torch.dot(P @ x, y)

    def f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.dot(self.c, y) + 0.01 * (x.norm() ** 2 + y.norm() ** 2)

    def constraint_penalty(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.A @ x + self.B @ y - self.b
        return self.penalty_coeff * torch.relu(h).sum()

################################################################################
# Minimal First‑Order BOME Optimizer                                           #
################################################################################

class BOMEOptimizer:
    """First‑order bilevel optimizer (BOME‑style) with explicit gradient noise."""

    def __init__(self, inner_lr: float, outer_lr: float, inner_steps: int = 1, grad_noise_std: float = 0.02):
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.grad_noise_std = grad_noise_std

    def _add_noise(self, g: torch.Tensor) -> torch.Tensor:
        if self.grad_noise_std > 0.0:
            return g + self.grad_noise_std * torch.randn_like(g)
        return g

    def step(self, problem: SyntheticQuadraticProblem, x: torch.Tensor, y: torch.Tensor):
        # Draw a noisy sample of (Q, P) every call – stochastic gradients
        Q, P = problem.sample_batch()

        # ----- Inner loop: GD on noisy g --------------------------------------
        for _ in range(self.inner_steps):
            g_val = problem.g(x, y, Q, P) + problem.constraint_penalty(x, y)
            (grad_y,) = torch.autograd.grad(g_val, y, retain_graph=True)
            y.data -= self.inner_lr * self._add_noise(grad_y)

        # ----- Outer loop: GD on f (first‑order only) -------------------------
        f_val = problem.f(x, y)
        (grad_x,) = torch.autograd.grad(f_val, x)
        x.data -= self.outer_lr * self._add_noise(grad_x)

        return f_val.item(), g_val.item()

################################################################################
# Noisy‑MNIST Data Cleaning with First‑Order Hypergradients                    #
################################################################################

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.fc(self.conv(x))


def get_noisy_mnist(noise_ratio: float = 0.2, seed: int = 0):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    val_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    n = len(train_set.targets)
    idx = rng.choice(n, int(noise_ratio * n), replace=False)
    noisy_labels = rng.integers(0, 10, size=len(idx))
    train_set.targets[idx] = torch.tensor(noisy_labels)

    return train_set, val_set

################################################################################
# Utilities                                                                    #
################################################################################

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)

################################################################################
# BOME for Sample Reweighting (upper‑level weights)                             #
################################################################################

class DataCleaningBOME:
    def __init__(self, model: nn.Module, inner_lr: float, outer_lr: float, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr

    def fit(self, train_set, val_set, epochs: int = 5, batch_size: int = 128):
        n_train = len(train_set)
        w = torch.ones(n_train, requires_grad=True, device=self.device)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        ce = nn.CrossEntropyLoss(reduction="none")
        opt_theta = optim.SGD(self.model.parameters(), lr=self.inner_lr)
        opt_w = optim.SGD([w], lr=self.outer_lr)

        for epoch in range(epochs):
            self.model.train()
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                idxs = torch.arange(batch_idx * batch_size, batch_idx * batch_size + y.size(0), device=self.device)

                # Inner loop update
                logits = self.model(x)
                loss_vec = ce(logits, y)
                loss_inner = (w[idxs] * loss_vec).mean()
                opt_theta.zero_grad()
                loss_inner.backward(retain_graph=True)
                opt_theta.step()

                # Outer reweighting update (first‑order approx)
                grad_w, = torch.autograd.grad(loss_inner, w, retain_graph=False, allow_unused=True)
                opt_w.zero_grad()
                w.grad = grad_w
                opt_w.step()
                w.data.clamp_(0.0, 5.0)

            acc = evaluate(self.model, val_loader, self.device)
            print(f"Epoch {epoch+1}/{epochs}  •  Val Acc = {acc:.4f}")
        return w.detach().cpu()

################################################################################
# Entry points                                                                 #
################################################################################

def run_synthetic(seed: int = 0):
    problem = SyntheticQuadraticProblem(x_dim=5, y_dim=10, m=3, noise_scale=0.2, seed=seed)
    x = torch.randn(5, dtype=torch.float32, requires_grad=True)
    y = torch.randn(10, dtype=torch.float32, requires_grad=True)
    opt = BOMEOptimizer(inner_lr=1e-2, outer_lr=5e-3, inner_steps=1, grad_noise_std=0.05)

    for it in range(500):
        f_val, g_val = opt.step(problem, x, y)
        if it % 50 == 0:
            penalty = problem.constraint_penalty(x, y).item()
            print(f"Iter {it:03d}  f={f_val:.4f}  g={g_val:.4f}  constraint_penalty={penalty:.4f}")


def run_data_cleaning(seed: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set = get_noisy_mnist(noise_ratio=0.2, seed=seed)
    model = SimpleCNN()
    trainer = DataCleaningBOME(model, inner_lr=0.01, outer_lr=0.1, device=device)
    trainer.fit(train_set, val_set, epochs=3)
    acc_final = evaluate(model, DataLoader(val_set, batch_size=256), device)
    print(f"Final validation accuracy: {acc_final:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["synthetic", "noise_mnist"], default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.task == "synthetic":
        run_synthetic(args.seed)
    else:
        run_data_cleaning(args.seed)
