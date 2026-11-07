"""Sanity checks before the sweep.

1. Reproduction: our vanilla arm must be the authors' procedure (pbc_examples/main_pbc.py).
   a) Step-0 equivalence: same initial weights, bitwise-identical loss and gradients equal to
      float32 rounding (~1e-7). Only the summation order inside backward differs (the authors also
      build an unused u_xx term), and L-BFGS amplifies that rounding over hundreds of iterations.
   b) For information, single-seed final errors vs the authors' script in this environment
      (torch 2.15.0+cpu, N_f = 100, seed 0): beta=1 -> 1.219548e-02, beta=30 -> 1.135607e+00.
      These agree to ~1-2% but not bitwise, for the reason in (a); compare over seeds instead
      (src/run_authors_script.py).
2. Hard constraints: HardPINN satisfies u(x, 0) = sin x and u(0, t) = u(2pi, t) to float32 precision.
3. Smoke test + timing: every arm runs end-to-end on a small budget with one thread.

Usage:  python src/check_setup.py            (run from the repo root)
"""
import itertools
import time

import numpy as np
import torch

import problem
from models import HardPINN
from train import RunConfig, run_one

AUTHORS = {1.0: 1.219548e-02, 30.0: 1.135607e+00}


def check_step0(beta, n_f=1000):
    """Authors' PhysicsInformedNN_pbc vs our Trainer at initialisation."""
    from net_pbc import PhysicsInformedNN_pbc  # authors' code
    from utils import set_seed                 # authors' code
    from train import Trainer

    d = problem.training_data(n_f)
    set_seed(0)
    authors = PhysicsInformedNN_pbc('convection', d['X_ic'], d['u_ic'], d['X_f'], d['bc_lb'],
                                    d['bc_ub'], [2, 50, 50, 50, 50, 1], np.zeros(n_f), 0.0, beta,
                                    0.0, 'LBFGS', 1.0, 'DNN', 1.0, 'tanh', 'mean')
    ours = Trainer(RunConfig(beta=beta, seed=0, n_f=n_f))
    same_w = all(torch.equal(a, b) for a, b in zip(authors.dnn.parameters(), ours.model.parameters()))
    la = authors.loss_pinn(verbose=False)
    ga = torch.cat([p.grad.flatten() for p in authors.dnn.parameters()])
    lo, _ = ours.losses(beta)
    lo.backward()
    go = torch.cat([p.grad.flatten() for p in ours.model.parameters()])
    grad_rel = ((ga - go).abs().max() / ga.abs().max()).item()
    ok = same_w and la.item() == lo.item() and grad_rel < 1e-6
    print(f"  beta={beta:>4}: same init {same_w}, loss {la.item():.9e} vs {lo.item():.9e}, "
          f"max grad diff {grad_rel:.1e} (relative) -> {'OK' if ok else 'FAIL'}")
    return ok


def check_reproduction():
    print("== 1a. step-0 equivalence with the authors' PhysicsInformedNN_pbc (N_f=1000) ==")
    ok = all(check_step0(beta) for beta in (1.0, 30.0))
    print("== 1b. single-seed final errors vs the authors' script (N_f=100, seed 0; info only) ==")
    for beta, ref in AUTHORS.items():
        row = run_one(RunConfig(beta=beta, seed=0, n_f=100, budget=125000, tag="check"))
        print(f"  beta={beta:>4}: ours {row['rel_l2']:.6e}  authors {ref:.6e}  "
              f"({row['loss_evals']} evals, {row['train_s']:.1f} s)")
    return ok


def check_hard_constraints():
    print("== 2. hard constraints ==")
    torch.manual_seed(0)
    m = HardPINN()
    x = torch.linspace(0, 2 * np.pi, 257).reshape(-1, 1)
    t = torch.linspace(0, 1, 101).reshape(-1, 1)
    with torch.no_grad():
        ic_err = (m(x, torch.zeros_like(x)) - torch.sin(x)).abs().max().item()
        bc_err = (m(torch.zeros_like(t), t) - m(torch.full_like(t, 2 * np.pi), t)).abs().max().item()
    print(f"  max |u(x,0) - sin x| = {ic_err:.2e}   max |u(0,t) - u(2pi,t)| = {bc_err:.2e}")
    ok = ic_err < 1e-6 and bc_err < 1e-5
    print("  ->", "OK" if ok else "FAIL")
    return ok


def smoke_and_time(budget=600):
    print(f"== 3. smoke test of all 8 arms, beta=30, N_f=1000, budget={budget}, 1 thread ==")
    torch.set_num_threads(1)
    for c, w, h in itertools.product([0, 1], repeat=3):
        cfg = RunConfig(beta=30.0, seed=0, curriculum=bool(c), causal=bool(w), hard=bool(h),
                        budget=budget, tag="check")
        row = run_one(cfg)
        ms = 1000 * row["train_s"] / max(row["loss_evals"], 1)
        print(f"  {row['arm']}: rel_l2 {row['rel_l2']:.3e}  evals {row['loss_evals']:>5}  "
              f"iters {row['lbfgs_iters']:>4}  stop={row['stop_reason']:<16} {ms:5.1f} ms/eval")


if __name__ == "__main__":
    t0 = time.time()
    a = check_reproduction()
    b = check_hard_constraints()
    smoke_and_time()
    print(f"done in {time.time() - t0:.0f} s; reproduction {'OK' if a else 'FAILED'}, "
          f"hard constraints {'OK' if b else 'FAILED'}")
