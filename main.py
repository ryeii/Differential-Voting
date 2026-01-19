#!/usr/bin/env python3
# Differential Voting 
# Usage:
#   !python main.py
#
# Outputs:
#   results_mid_raw/
#     exp1/
#       scenario_<name>.npz                (all raw arrays for Exp1 plots)
#       exp1_mid_<name>_copeland.png
#       exp1_mid_<name>_kemeny.png
#     exp2/
#       exp2_mid_raw.npz                   (all raw arrays for Exp2 plots)
#       exp2_mid_condorcet_hit.png
#       exp2_mid_kemeny_dist.png
#       exp2_mid_copeland_agree.png
#     exp3/
#       geo_<setup>.npz                    (raw delta/gradabs snapshots for geometry plots)
#       exp3_mid_grad_vs_margin_<setup>.png
#       exp3_mid_noise_raw.npz             (raw per-seed noise sweep results)
#       exp3_mid_noise_winner_acc.png
#       exp3_mid_noise_kemeny_kdist.png
#
# Dependencies: torch, numpy, matplotlib

import os
import time
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------
# Repro / utilities
# ----------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def rank_from_scores(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores)

def kendall_tau_distance(perm_a: np.ndarray, perm_b: np.ndarray) -> int:
    m = len(perm_a)
    inv_a = np.empty(m, dtype=int); inv_b = np.empty(m, dtype=int)
    inv_a[perm_a] = np.arange(m)
    inv_b[perm_b] = np.arange(m)
    dist = 0
    for i in range(m):
        for j in range(i + 1, m):
            dist += ((inv_a[i] < inv_a[j]) != (inv_b[i] < inv_b[j]))
    return dist

def majority_dir_from_eta(eta: np.ndarray) -> np.ndarray:
    m = eta.shape[0]
    maj = np.zeros((m, m), dtype=int)
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            if eta[a, b] > 0.5:
                maj[a, b] = 1
                maj[b, a] = 0
            elif eta[a, b] < 0.5:
                maj[a, b] = 0
                maj[b, a] = 1
            else:
                maj[a, b] = 0
                maj[b, a] = 0
    return maj

def copeland_scores_from_majority(majority_dir: np.ndarray) -> np.ndarray:
    wins = majority_dir.sum(axis=1)
    losses = majority_dir.sum(axis=0)
    return wins - losses

def condorcet_winner_from_majority(majority_dir: np.ndarray) -> Optional[int]:
    m = majority_dir.shape[0]
    wins = majority_dir.sum(axis=1)
    for a in range(m):
        if wins[a] == m - 1:
            return a
    return None

def kemeny_optimal_ranking_bruteforce(majority_dir: np.ndarray) -> np.ndarray:
    m = majority_dir.shape[0]
    assert m <= 8, "This script uses brute-force Kemeny; keep m<=8."
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m)]
    best_perm = None
    best_obj = 10**18
    candidates = list(range(m))
    for perm in itertools.permutations(candidates):
        inv = np.empty(m, dtype=int)
        inv[list(perm)] = np.arange(m)
        obj = 0
        for a, b in pairs:
            if majority_dir[a, b] == 1:
                obj += (inv[a] > inv[b])
            else:
                obj += (inv[b] > inv[a])
        if obj < best_obj:
            best_obj = obj
            best_perm = np.array(perm, dtype=int)
    return best_perm

def borda_like_scores_from_eta(eta: np.ndarray) -> np.ndarray:
    return eta.sum(axis=1) - np.diag(eta)


# ----------------------------
# Synthetic generators
# ----------------------------

def make_eta_transitive(m: int, kappa: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    u = rng.randn(m)
    eta = np.zeros((m, m), dtype=float)
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            eta[a, b] = sigmoid_np((u[a] - u[b]) / kappa)
    return eta

def make_eta_cycle(m: int, noise: float = 0.12, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    eta = np.zeros((m, m), dtype=float)
    margin = 0.25
    cycle = [(0, 1), (1, 2), (2, 0)]
    for a, b in cycle:
        eta[a, b] = 0.5 + margin
        eta[b, a] = 0.5 - margin
    for a in range(m):
        for b in range(a + 1, m):
            if a < 3 and b < 3:
                continue
            base = 0.5 + rng.uniform(-noise, noise)
            base = float(np.clip(base, 0.05, 0.95))
            eta[a, b] = base
            eta[b, a] = 1 - base
    return eta

def make_eta_neartie(m: int, tight: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    eta = np.zeros((m, m), dtype=float)
    for a in range(m):
        for b in range(a + 1, m):
            base = 0.5 + rng.uniform(-tight, tight)
            base = float(np.clip(base, 0.05, 0.95))
            eta[a, b] = base
            eta[b, a] = 1 - base
    return eta

def sample_pairwise_dataset(
    eta: np.ndarray,
    n: int,
    seed: int = 0,
    noise_flip: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    m = eta.shape[0]
    a = rng.randint(0, m, size=n)
    b = rng.randint(0, m, size=n)
    mask = (a != b)
    a = a[mask]; b = b[mask]
    while len(a) < n:
        aa = rng.randint(0, m, size=n - len(a))
        bb = rng.randint(0, m, size=n - len(a))
        mm = (aa != bb)
        a = np.concatenate([a, aa[mm]])
        b = np.concatenate([b, bb[mm]])
    a = a[:n]; b = b[:n]

    p = eta[a, b]
    y = (rng.rand(n) < p).astype(int)
    y = 2 * y - 1
    if noise_flip > 0:
        flips = rng.rand(n) < noise_flip
        y[flips] *= -1
    return a, b, y


# ----------------------------
# Hidden-context data
# ----------------------------

def make_hidden_context_mixture_with_global_condorcet(
    m: int, K: int, kappa: float, seed: int, winner: int = 0, strength: float = 1.0
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    eta_ctx = np.zeros((K, m, m), dtype=float)
    for c in range(K):
        u = rng.randn(m)
        u[winner] += strength
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                eta_ctx[c, a, b] = sigmoid_np((u[a] - u[b]) / kappa)
    return eta_ctx

def sample_hidden_context_dataset(
    eta_ctx: np.ndarray,
    pi: np.ndarray,
    n: int,
    seed: int,
    noise_flip: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    K, m, _ = eta_ctx.shape
    c = rng.choice(K, size=n, p=pi)
    a = rng.randint(0, m, size=n)
    b = rng.randint(0, m, size=n)
    mask = (a != b)
    a = a[mask]; b = b[mask]; c = c[mask]
    while len(a) < n:
        cc = rng.choice(K, size=n - len(a), p=pi)
        aa = rng.randint(0, m, size=n - len(a))
        bb = rng.randint(0, m, size=n - len(a))
        mm = (aa != bb)
        a = np.concatenate([a, aa[mm]])
        b = np.concatenate([b, bb[mm]])
        c = np.concatenate([c, cc[mm]])
    a = a[:n]; b = b[:n]; c = c[:n]

    p = eta_ctx[c, a, b]
    y = (rng.rand(n) < p).astype(int)
    y = 2 * y - 1
    if noise_flip > 0:
        flips = rng.rand(n) < noise_flip
        y[flips] *= -1
    return a, b, y, c


# ----------------------------
# Losses + training
# ----------------------------

@dataclass
class TrainConfig:
    lr: float = 0.1
    steps: int = 4500
    batch_size: int = 768
    weight_decay: float = 0.0
    device: str = "cpu"
    log_every: int = 300

@dataclass
class LossParams:
    tau: float
    beta: float = 10.0
    lam: float = 1e-4

def loss_btl(delta: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.log1p(torch.exp(-y * delta / tau))

def loss_soft_copeland(delta: torch.Tensor, y: torch.Tensor, tau: float, beta: float, lam: float) -> torch.Tensor:
    p = torch.sigmoid(delta / tau)
    s = torch.tanh(beta * (p - 0.5))
    return -y * s + 0.5 * lam * (delta ** 2)

def loss_soft_kemeny(delta: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sigmoid(-y * delta / tau)

def dlddelta_btl(delta: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    z = -y * delta / tau
    sig = 1 / (1 + np.exp(-z))
    return np.abs(-(y / tau) * sig)

def dlddelta_soft_kemeny(delta: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    z = -y * delta / tau
    sig = 1 / (1 + np.exp(-z))
    return np.abs(-(y / tau) * sig * (1 - sig))

def dlddelta_soft_copeland(delta: np.ndarray, y: np.ndarray, tau: float, beta: float, lam: float) -> np.ndarray:
    p = 1 / (1 + np.exp(-delta / tau))
    sech2 = 1 / (np.cosh(beta * (p - 0.5)) ** 2)
    ds = (beta / tau) * p * (1 - p) * sech2
    return np.abs(-y * ds + lam * delta)

def train_scores(
    m: int,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    y: np.ndarray,
    loss_name: str,
    lp: LossParams,
    cfg: TrainConfig,
    seed: int = 0,
    return_logs: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    set_seed(seed)
    device = cfg.device
    r = torch.nn.Parameter(torch.zeros(m, device=device))
    opt = torch.optim.Adam([r], lr=cfg.lr, weight_decay=cfg.weight_decay)

    n = len(y)
    logs: Dict[str, Any] = {"loss": [], "step": [], "r_norm": []}
    snaps = []

    for step in range(1, cfg.steps + 1):
        idx = np.random.randint(0, n, size=cfg.batch_size)
        a = torch.tensor(a_idx[idx], device=device, dtype=torch.long)
        b = torch.tensor(b_idx[idx], device=device, dtype=torch.long)
        yy = torch.tensor(y[idx], device=device, dtype=torch.float32)

        delta = r[a] - r[b]

        if loss_name == "btl":
            loss = loss_btl(delta, yy, lp.tau).mean()
        elif loss_name == "copeland":
            loss = loss_soft_copeland(delta, yy, lp.tau, lp.beta, lp.lam).mean()
        elif loss_name == "kemeny":
            loss = loss_soft_kemeny(delta, yy, lp.tau).mean()
        else:
            raise ValueError("Unknown loss_name")

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step % cfg.log_every == 0) or (step == 1) or (step == cfg.steps):
            logs["loss"].append(float(loss.detach().cpu().item()))
            logs["step"].append(step)
            logs["r_norm"].append(float(torch.norm(r.detach()).cpu().item()))

            if return_logs:
                snap_n = min(12000, n)
                snap_idx = np.random.randint(0, n, size=snap_n)
                rr = r.detach().cpu().numpy()
                dd = rr[a_idx[snap_idx]] - rr[b_idx[snap_idx]]
                dy = y[snap_idx]
                if loss_name == "btl":
                    gg = dlddelta_btl(dd, dy, lp.tau)
                elif loss_name == "kemeny":
                    gg = dlddelta_soft_kemeny(dd, dy, lp.tau)
                else:
                    gg = dlddelta_soft_copeland(dd, dy, lp.tau, lp.beta, lp.lam)
                snaps.append({"delta": dd.astype(np.float32), "gradabs": gg.astype(np.float32), "step": step})

    rr = r.detach().cpu().numpy().astype(np.float32)
    if return_logs:
        logs["grad_margin_snapshots"] = snaps
    return rr, logs


# ----------------------------
# Plot helper
# ----------------------------

def binned_gradient_curve(delta: np.ndarray, gradabs: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    inds = np.digitize(delta, bins) - 1
    vals = np.zeros_like(centers, dtype=np.float64)
    counts = np.zeros_like(centers, dtype=np.int64)
    for i in range(len(delta)):
        k = inds[i]
        if 0 <= k < len(centers):
            vals[k] += float(gradabs[i])
            counts[k] += 1
    vals = vals / np.maximum(counts, 1)
    return centers, vals, counts

def plot_grad_vs_margin_from_raw(raw_npz_path: str, out_png_path: str, title: str):
    d = np.load(raw_npz_path)
    delta = d["delta"].astype(np.float64)
    gradabs = d["gradabs"].astype(np.float64)
    bins = d["bins"].astype(np.float64)
    centers, vals, _counts = binned_gradient_curve(delta, gradabs, bins)
    fig = plt.figure()
    plt.plot(centers, vals, marker="o")
    plt.xlabel("Margin Δ = r(a) - r(b)")
    plt.ylabel("Average |∂ℓ/∂Δ|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=160)
    plt.close(fig)


# ----------------------------
# Experiment 1 (RAW-SAVE)
# ----------------------------

def experiment_1(root_dir: str, base_seed: int = 0):
    exp_dir = os.path.join(root_dir, "exp1")
    ensure_dir(exp_dir)
    print("\n[Exp1] Recovery vs smoothing (RAW-SAVE)")

    m = 8
    N = 60_000
    seeds = [base_seed + i for i in range(5)]
    cfg = TrainConfig(lr=0.1, steps=5000, batch_size=768, log_every=350, device="cpu")

    scenarios = [
        ("transitive", lambda s: make_eta_transitive(m, kappa=1.0, seed=s)),
        ("cycle",      lambda s: make_eta_cycle(m, noise=0.12, seed=s)),
        ("neartie",    lambda s: make_eta_neartie(m, tight=0.05, seed=s)),
        ("transitive_sharp", lambda s: make_eta_transitive(m, kappa=0.6, seed=s)),
    ]

    taus = np.array([0.5, 0.2, 0.1, 0.05], dtype=np.float32)
    betas = np.array([2.0, 10.0], dtype=np.float32)
    lams = np.array([3e-4, 1e-4], dtype=np.float32)

    for sc_i, (sc_name, eta_fn) in enumerate(scenarios):
        print(f"  Scenario: {sc_name}")
        eta = eta_fn(base_seed + 1000 + 17 * sc_i).astype(np.float32)
        maj = majority_dir_from_eta(eta)
        cop_winner = int(np.argmax(copeland_scores_from_majority(maj)))
        kem_rank = kemeny_optimal_ranking_bruteforce(maj).astype(np.int64)
        borda_rank = rank_from_scores(borda_like_scores_from_eta(eta)).astype(np.int64)
        borda_kdist = kendall_tau_distance(borda_rank, kem_rank)

        # Per-seed baselines and sweeps
        btl_rankings = np.zeros((len(seeds), m), dtype=np.int64)
        btl_hits = np.zeros((len(seeds),), dtype=np.float32)
        btl_kd = np.zeros((len(seeds),), dtype=np.float32)

        kem_rankings = np.zeros((len(taus), len(seeds), m), dtype=np.int64)
        kem_kd = np.zeros((len(taus), len(seeds)), dtype=np.float32)

        cop_rankings = np.zeros((len(taus), len(betas), len(lams), len(seeds), m), dtype=np.int64)
        cop_hits = np.zeros((len(taus), len(betas), len(lams), len(seeds)), dtype=np.float32)

        # Also store final learned score vectors r for optional post-hoc analysis
        r_btl = np.zeros((len(seeds), m), dtype=np.float32)
        r_kem = np.zeros((len(taus), len(seeds), m), dtype=np.float32)
        r_cop = np.zeros((len(taus), len(betas), len(lams), len(seeds), m), dtype=np.float32)

        for si, sd in enumerate(seeds):
            a, b, y = sample_pairwise_dataset(eta, n=N, seed=sd + 20000 + sc_i)
            a = a.astype(np.int64); b = b.astype(np.int64); y = y.astype(np.int8)

            rb, logs_btl = train_scores(m, a, b, y, "btl", LossParams(tau=0.2), cfg, seed=sd + 30000 + sc_i)
            r_btl[si] = rb
            rank_b = rank_from_scores(rb).astype(np.int64)
            btl_rankings[si] = rank_b
            btl_hits[si] = float(rank_b[0] == cop_winner)
            btl_kd[si] = float(kendall_tau_distance(rank_b, kem_rank))

            for ti, tau in enumerate(taus):
                rk, logs_k = train_scores(m, a, b, y, "kemeny", LossParams(tau=float(tau)), cfg, seed=sd + 40000 + 10*ti + sc_i)
                r_kem[ti, si] = rk
                rank_k = rank_from_scores(rk).astype(np.int64)
                kem_rankings[ti, si] = rank_k
                kem_kd[ti, si] = float(kendall_tau_distance(rank_k, kem_rank))

                for bi, beta in enumerate(betas):
                    for li, lam in enumerate(lams):
                        rc, logs_c = train_scores(
                            m, a, b, y, "copeland",
                            LossParams(tau=float(tau), beta=float(beta), lam=float(lam)),
                            cfg,
                            seed=sd + 50000 + 100*ti + 10*bi + li + sc_i
                        )
                        r_cop[ti, bi, li, si] = rc
                        rank_c = rank_from_scores(rc).astype(np.int64)
                        cop_rankings[ti, bi, li, si] = rank_c
                        cop_hits[ti, bi, li, si] = float(rank_c[0] == cop_winner)

        # Aggregate arrays needed for plots
        best_cop_by_tau = cop_hits.mean(axis=3).reshape(len(taus), -1).max(axis=1).astype(np.float32)
        btl_hit_mean = float(btl_hits.mean())
        btl_kd_mean = float(btl_kd.mean())
        kem_kd_mean = kem_kd.mean(axis=1).astype(np.float32)
        kem_kd_std = kem_kd.std(axis=1).astype(np.float32)

        # Save raw data for this scenario (everything needed to replot)
        raw_path = os.path.join(exp_dir, f"scenario_{sc_name}.npz")
        np.savez(
            raw_path,
            scenario=sc_name,
            m=m,
            N=N,
            seeds=np.array(seeds, dtype=np.int64),
            taus=taus,
            betas=betas,
            lams=lams,
            eta=eta,
            majority=maj.astype(np.int8),
            cop_winner=np.int64(cop_winner),
            kem_rank=kem_rank,
            borda_rank=borda_rank,
            borda_kdist=np.int64(borda_kdist),
            # learned scores
            r_btl=r_btl,
            r_kem=r_kem,
            r_cop=r_cop,
            # rankings + metrics
            btl_rankings=btl_rankings,
            btl_hits=btl_hits,
            btl_kd=btl_kd,
            kem_rankings=kem_rankings,
            kem_kd=kem_kd,
            cop_rankings=cop_rankings,
            cop_hits=cop_hits,
            # plot-ready aggregates
            btl_hit_mean=np.float32(btl_hit_mean),
            btl_kd_mean=np.float32(btl_kd_mean),
            best_cop_by_tau=best_cop_by_tau,
            kem_kd_mean=kem_kd_mean,
            kem_kd_std=kem_kd_std,
        )

        # Plot: Copeland recovery (from raw)
        fig = plt.figure()
        plt.plot(taus, [btl_hit_mean]*len(taus), marker="o", label="BTL (tau=0.2)")
        plt.plot(taus, best_cop_by_tau, marker="o", label="Soft Copeland (best over beta,lam)")
        plt.gca().invert_xaxis()
        plt.xlabel("tau (smaller = sharper)")
        plt.ylabel("Winner matches Copeland? (mean over seeds)")
        plt.title(f"Exp1 MID — {sc_name}: Copeland recovery")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"exp1_mid_{sc_name}_copeland.png"), dpi=160)
        plt.close(fig)

        # Plot: Kemeny recovery (from raw)
        fig = plt.figure()
        plt.plot(taus, [btl_kd_mean]*len(taus), marker="o", label="BTL (tau=0.2)")
        plt.errorbar(taus, kem_kd_mean, yerr=kem_kd_std, marker="o", capsize=3, label="Soft Kemeny (mean±std)")
        plt.gca().invert_xaxis()
        plt.xlabel("tau (smaller = sharper)")
        plt.ylabel("Kendall distance to Kemeny optimum")
        plt.title(f"Exp1 MID — {sc_name}: Kemeny recovery")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"exp1_mid_{sc_name}_kemeny.png"), dpi=160)
        plt.close(fig)

    print(f"[Exp1] Saved scenario_*.npz + plots in {exp_dir}")


# ----------------------------
# Experiment 2 (RAW-SAVE)
# ----------------------------

def experiment_2(root_dir: str, base_seed: int = 0):
    exp_dir = os.path.join(root_dir, "exp2")
    ensure_dir(exp_dir)
    print("\n[Exp2] Hidden-context stress test (RAW-SAVE)")

    m = 8
    K = 4
    N = 70_000
    seeds = [base_seed + i for i in range(4)]
    cfg = TrainConfig(lr=0.1, steps=5000, batch_size=768, log_every=350, device="cpu")

    pi1_grid = np.linspace(0.1, 0.9, 9).astype(np.float32)
    strength = 1.0
    kappa = 0.8

    methods = {
        "BTL_tau0.2": ("btl", LossParams(tau=0.2)),
        "SoftCopeland_tau0.1_b10_l1e-4": ("copeland", LossParams(tau=0.1, beta=10.0, lam=1e-4)),
        "SoftKemeny_tau0.1": ("kemeny", LossParams(tau=0.1)),
    }
    method_names = list(methods.keys())

    # Fix eta_ctx per seed for stability across pi
    eta_ctx_by_seed = {
        sd: make_hidden_context_mixture_with_global_condorcet(m, K, kappa=kappa, seed=sd + 999, winner=0, strength=strength).astype(np.float32)
        for sd in seeds
    }

    # Raw per-seed storage
    cw_exists = np.zeros((len(pi1_grid), len(seeds)), dtype=np.float32)
    cw_idx = np.full((len(pi1_grid), len(seeds)), -1, dtype=np.int64)

    cop_winner_idx = np.zeros((len(pi1_grid), len(seeds)), dtype=np.int64)
    kemeny_rank = np.zeros((len(pi1_grid), len(seeds), m), dtype=np.int64)

    # Per-method metrics and winners
    cond_hit = np.full((len(method_names), len(pi1_grid), len(seeds)), np.nan, dtype=np.float32)
    kdist = np.zeros((len(method_names), len(pi1_grid), len(seeds)), dtype=np.float32)
    cop_agree = np.zeros((len(method_names), len(pi1_grid), len(seeds)), dtype=np.float32)

    chosen_winner = np.zeros((len(method_names), len(pi1_grid), len(seeds)), dtype=np.int64)
    chosen_rank = np.zeros((len(method_names), len(pi1_grid), len(seeds), m), dtype=np.int64)

    # Cache targets to avoid repeated Kemeny recomputation within same (seed, pi1)
    target_cache: Dict[Tuple[int, float], Dict[str, Any]] = {}

    for pi_i, pi1 in enumerate(pi1_grid):
        pi = np.full(K, (1.0 - float(pi1)) / (K - 1), dtype=np.float32)
        pi[0] = pi1

        for si, sd in enumerate(seeds):
            eta_ctx = eta_ctx_by_seed[sd]
            cache_key = (sd, float(pi1))

            if cache_key not in target_cache:
                eta_bar = (pi[:, None, None] * eta_ctx).sum(axis=0)
                maj = majority_dir_from_eta(eta_bar)
                cw = condorcet_winner_from_majority(maj)
                kem = kemeny_optimal_ranking_bruteforce(maj).astype(np.int64)
                copw = int(np.argmax(copeland_scores_from_majority(maj)))
                target_cache[cache_key] = {"cw": cw, "kem": kem, "copw": copw}

            cw = target_cache[cache_key]["cw"]
            kem = target_cache[cache_key]["kem"]
            copw = target_cache[cache_key]["copw"]

            if cw is not None:
                cw_exists[pi_i, si] = 1.0
                cw_idx[pi_i, si] = int(cw)
            cop_winner_idx[pi_i, si] = int(copw)
            kemeny_rank[pi_i, si] = kem

            a, b, y, _c = sample_hidden_context_dataset(eta_ctx, pi, n=N, seed=sd + 12345, noise_flip=0.0)
            a = a.astype(np.int64); b = b.astype(np.int64); y = y.astype(np.int8)

            for mi, mn in enumerate(method_names):
                lname, lp = methods[mn]
                r, _ = train_scores(m, a, b, y, lname, lp, cfg, seed=sd + 20000 + (hash(mn) % 10000))
                rk = rank_from_scores(r).astype(np.int64)
                chosen_rank[mi, pi_i, si] = rk
                chosen_winner[mi, pi_i, si] = int(rk[0])

                kdist[mi, pi_i, si] = float(kendall_tau_distance(rk, kem))
                cop_agree[mi, pi_i, si] = float(rk[0] == copw)
                if cw is not None:
                    cond_hit[mi, pi_i, si] = float(rk[0] == cw)

    # Plot-ready aggregates (means over seeds)
    cw_exists_rate = cw_exists.mean(axis=1).astype(np.float32)
    cond_hit_mean = np.nanmean(cond_hit, axis=2).astype(np.float32)  # nanmean over seeds (only when cw exists)
    kdist_mean = kdist.mean(axis=2).astype(np.float32)
    cop_agree_mean = cop_agree.mean(axis=2).astype(np.float32)

    # Save all raw arrays
    raw_path = os.path.join(exp_dir, "exp2_mid_raw.npz")
    np.savez(
        raw_path,
        m=m, K=K, N=N,
        seeds=np.array(seeds, dtype=np.int64),
        pi1_grid=pi1_grid,
        method_names=np.array(method_names, dtype=object),
        # targets
        cw_exists=cw_exists,
        cw_idx=cw_idx,
        cop_winner_idx=cop_winner_idx,
        kemeny_rank=kemeny_rank,
        # outcomes
        chosen_winner=chosen_winner,
        chosen_rank=chosen_rank,
        # metrics (per seed)
        cond_hit=cond_hit,
        kdist=kdist,
        cop_agree=cop_agree,
        # plot-ready aggregates
        cw_exists_rate=cw_exists_rate,
        cond_hit_mean=cond_hit_mean,
        kdist_mean=kdist_mean,
        cop_agree_mean=cop_agree_mean,
    )

    # Plots (from aggregates)
    fig = plt.figure()
    for mi, mn in enumerate(method_names):
        plt.plot(pi1_grid, cond_hit_mean[mi], marker="o", label=mn)
    plt.plot(pi1_grid, cw_exists_rate, marker="x", linewidth=2, label="Condorcet exists rate")
    plt.xlabel("Mixture weight pi1 (context 0)")
    plt.ylabel("Condorcet winner hit-rate (when defined)")
    plt.title("Exp2 MID: Condorcet criterion under hidden contexts")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "exp2_mid_condorcet_hit.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure()
    for mi, mn in enumerate(method_names):
        plt.plot(pi1_grid, kdist_mean[mi], marker="o", label=mn)
    plt.xlabel("Mixture weight pi1 (context 0)")
    plt.ylabel("Kendall distance to Kemeny optimum")
    plt.title("Exp2 MID: Closeness to Kemeny ranking")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "exp2_mid_kemeny_dist.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure()
    for mi, mn in enumerate(method_names):
        plt.plot(pi1_grid, cop_agree_mean[mi], marker="o", label=mn)
    plt.xlabel("Mixture weight pi1 (context 0)")
    plt.ylabel("Winner matches Copeland? (mean over seeds)")
    plt.title("Exp2 MID: Copeland-winner agreement")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "exp2_mid_copeland_agree.png"), dpi=160)
    plt.close(fig)

    print(f"[Exp2] Saved exp2_mid_raw.npz + plots in {exp_dir}")


# ----------------------------
# Experiment 3 (RAW-SAVE)
# ----------------------------

def experiment_3(root_dir: str, base_seed: int = 0):
    exp_dir = os.path.join(root_dir, "exp3")
    ensure_dir(exp_dir)
    print("\n[Exp3] Geometry + robustness (RAW-SAVE)")

    m = 8
    N = 80_000
    cfg = TrainConfig(lr=0.1, steps=5000, batch_size=768, log_every=350, device="cpu")

    eta = 0.6 * make_eta_cycle(m, noise=0.10, seed=base_seed + 7) + 0.4 * make_eta_neartie(m, tight=0.05, seed=base_seed + 8)
    eta = eta.astype(np.float32)
    for a in range(m):
        for b in range(a + 1, m):
            eta[b, a] = 1 - eta[a, b]

    maj = majority_dir_from_eta(eta)
    cop_winner = int(np.argmax(copeland_scores_from_majority(maj)))
    kem_rank = kemeny_optimal_ranking_bruteforce(maj).astype(np.int64)

    # Save the distribution itself (so you can rerun plotting/targets without regenerating)
    np.savez(os.path.join(exp_dir, "exp3_problem_definition.npz"),
             m=m, N=N, eta=eta, majority=maj.astype(np.int8),
             cop_winner=np.int64(cop_winner), kem_rank=kem_rank)

    a0, b0, y0 = sample_pairwise_dataset(eta, n=N, seed=base_seed + 111, noise_flip=0.0)
    a0 = a0.astype(np.int64); b0 = b0.astype(np.int64); y0 = y0.astype(np.int8)

    # Geometry setups
    geo_setups = [
        ("BTL_tau0.2", "btl", LossParams(tau=0.2)),
        ("Kem_tau0.1", "kemeny", LossParams(tau=0.1)),
        ("Cop_tau0.1_b2", "copeland", LossParams(tau=0.1, beta=2.0, lam=1e-4)),
        ("Cop_tau0.1_b10", "copeland", LossParams(tau=0.1, beta=10.0, lam=1e-4)),
    ]

    # bins for binned curve (save alongside raw)
    bins = np.linspace(-3.5, 3.5, 36).astype(np.float32)

    for setup_name, lname, lp in geo_setups:
        r, logs = train_scores(m, a0, b0, y0, lname, lp, cfg,
                               seed=base_seed + 2000 + (hash(setup_name) % 1000),
                               return_logs=True)
        snap = logs["grad_margin_snapshots"][-1]
        delta = snap["delta"].astype(np.float32)
        gradabs = snap["gradabs"].astype(np.float32)
        step = int(snap["step"])

        # Save raw for this geometry plot
        raw_path = os.path.join(exp_dir, f"geo_{setup_name}.npz")
        np.savez(
            raw_path,
            setup_name=setup_name,
            loss_name=lname,
            tau=np.float32(lp.tau),
            beta=np.float32(lp.beta),
            lam=np.float32(lp.lam),
            step=np.int64(step),
            bins=bins,
            delta=delta,
            gradabs=gradabs,
            r_final=r.astype(np.float32),
            train_steps=np.int64(cfg.steps),
            batch_size=np.int64(cfg.batch_size),
        )

        # Plot from raw (so you can regenerate later from .npz)
        out_png = os.path.join(exp_dir, f"exp3_mid_grad_vs_margin_{setup_name}.png")
        plot_grad_vs_margin_from_raw(raw_path, out_png, title=f"Exp3 MID: Gradient vs margin ({setup_name})")

    # Robustness sweep — save per seed, per eps, per method (full raw)
    noise_grid = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    seeds = [base_seed + i for i in range(5)]

    methods = {
        "BTL_tau0.2": ("btl", LossParams(tau=0.2)),
        "SoftCopeland_tau0.1_b10": ("copeland", LossParams(tau=0.1, beta=10.0, lam=1e-4)),
        "SoftKemeny_tau0.1": ("kemeny", LossParams(tau=0.1)),
    }
    method_names = list(methods.keys())

    winner_acc = np.zeros((len(method_names), len(noise_grid), len(seeds)), dtype=np.float32)
    kdist = np.zeros((len(method_names), len(noise_grid), len(seeds)), dtype=np.float32)
    chosen_winner = np.zeros((len(method_names), len(noise_grid), len(seeds)), dtype=np.int64)
    chosen_rank = np.zeros((len(method_names), len(noise_grid), len(seeds), m), dtype=np.int64)

    for ei, eps in enumerate(noise_grid):
        for si, sd in enumerate(seeds):
            a, b, y = sample_pairwise_dataset(eta, n=N, seed=sd + 5000, noise_flip=float(eps))
            a = a.astype(np.int64); b = b.astype(np.int64); y = y.astype(np.int8)

            for mi, mn in enumerate(method_names):
                lname, lp = methods[mn]
                r, _ = train_scores(m, a, b, y, lname, lp, cfg,
                                    seed=sd + 6000 + (hash(mn) % 1000),
                                    return_logs=False)
                rk = rank_from_scores(r).astype(np.int64)
                chosen_rank[mi, ei, si] = rk
                chosen_winner[mi, ei, si] = int(rk[0])
                winner_acc[mi, ei, si] = float(rk[0] == cop_winner)
                kdist[mi, ei, si] = float(kendall_tau_distance(rk, kem_rank))

    # Aggregates for plotting
    winner_acc_mean = winner_acc.mean(axis=2).astype(np.float32)
    kdist_mean = kdist.mean(axis=2).astype(np.float32)

    # Save raw for noise sweep
    noise_raw_path = os.path.join(exp_dir, "exp3_mid_noise_raw.npz")
    np.savez(
        noise_raw_path,
        m=m, N=N,
        seeds=np.array(seeds, dtype=np.int64),
        noise_grid=noise_grid,
        method_names=np.array(method_names, dtype=object),
        cop_winner=np.int64(cop_winner),
        kem_rank=kem_rank,
        # per-seed outcomes/metrics
        chosen_winner=chosen_winner,
        chosen_rank=chosen_rank,
        winner_acc=winner_acc,
        kdist=kdist,
        # plot-ready aggregates
        winner_acc_mean=winner_acc_mean,
        kdist_mean=kdist_mean,
    )

    # Plot from aggregates
    fig = plt.figure()
    for mi, mn in enumerate(method_names):
        plt.plot(noise_grid, winner_acc_mean[mi], marker="o", label=mn)
    plt.xlabel("Label flip probability ε")
    plt.ylabel("Winner matches Copeland? (mean over seeds)")
    plt.title("Exp3 MID: Winner robustness to noise")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "exp3_mid_noise_winner_acc.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure()
    for mi, mn in enumerate(method_names):
        plt.plot(noise_grid, kdist_mean[mi], marker="o", label=mn)
    plt.xlabel("Label flip probability ε")
    plt.ylabel("Kendall distance to Kemeny optimum")
    plt.title("Exp3 MID: Ranking robustness to noise")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "exp3_mid_noise_kemeny_kdist.png"), dpi=160)
    plt.close(fig)

    print(f"[Exp3] Saved geo_*.npz + exp3_mid_noise_raw.npz + plots in {exp_dir}")


# ----------------------------
# Main
# ----------------------------

def main():
    t0 = time.time()
    root_dir = "results"
    ensure_dir(root_dir)

    experiment_1(root_dir, base_seed=0)
    experiment_2(root_dir, base_seed=100)
    experiment_3(root_dir, base_seed=200)

    dt = time.time() - t0
    print(f"\nDone. Outputs in: {os.path.abspath(root_dir)}")
    print(f"Total runtime: {dt:.1f}s (Colab CPU dependent)")

if __name__ == "__main__":
    main()
