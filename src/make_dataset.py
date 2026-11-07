"""Export the benchmark dataset to data/convection_benchmark.npz.

The PDE has a closed-form solution, so the "dataset" consists of the fixed evaluation grid, the
reference solutions on it, and the exact training points every run uses. Anyone can evaluate a new
method against our numbers without running this code.

Contents (float32 unless noted):
  x (256,), t (100,)             evaluation grid; x in [0, 2pi) without endpoint, t in [0, 1]
  betas (11,) float64            convection speeds in the benchmark
  u_exact (11, 100, 256)         sin(x - beta t) on the grid, indexed [beta, t, x]
  X_f_1000 (1000, 2), X_f_100    collocation points (x, t) for N_f = 1000 (main) and 100 (authors' default)
  X_ic (256, 2), u_ic (256, 1)   initial-condition points and values
  bc_lb (100, 2), bc_ub (100, 2) periodic boundary point pairs (x = 0 and x = 2pi)

Usage:  python src/make_dataset.py
"""
from pathlib import Path

import numpy as np

import problem

REPO = Path(__file__).resolve().parents[1]
BETAS = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=np.float64)


def main():
    x, t, X_star = problem.grid()
    u = np.stack([problem.exact_u(b, X_star).reshape(problem.NT, problem.XGRID) for b in BETAS])
    d1000 = problem.training_data(1000)
    d100 = problem.training_data(100)
    out = REPO / "data" / "convection_benchmark.npz"
    out.parent.mkdir(exist_ok=True)
    f32 = np.float32
    np.savez_compressed(out, x=x.ravel().astype(f32), t=t.ravel().astype(f32), betas=BETAS,
                        u_exact=u.astype(f32), X_f_1000=d1000["X_f"].astype(f32),
                        X_f_100=d100["X_f"].astype(f32), X_ic=d1000["X_ic"].astype(f32),
                        u_ic=d1000["u_ic"].astype(f32), bc_lb=d1000["bc_lb"].astype(f32),
                        bc_ub=d1000["bc_ub"].astype(f32))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
