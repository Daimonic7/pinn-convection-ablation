"""Problem setup for 1D periodic convection.

    u_t + beta * u_x = 0,   (x, t) in [0, 2*pi) x [0, 1],   u(x, 0) = sin(x),   u(0, t) = u(2*pi, t)
    exact solution: u(x, t) = sin(x - beta * t)

The evaluation grid, initial/boundary points and collocation sampling replicate
pbc_examples/main_pbc.py from Krishnapriyan et al., "Characterizing possible failure modes in
physics-informed neural networks", NeurIPS 2021 (code: MIT license, (c) 2021 The Regents of the
University of California / LBNL). We import the authors' helpers instead of copying them, so the
"vanilla" arm of this study trains on exactly the same points as the authors' script.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORS_DIR = REPO_ROOT / "pbc_examples"
if str(AUTHORS_DIR) not in sys.path:
    sys.path.insert(0, str(AUTHORS_DIR))

from utils import sample_random  # noqa: E402  (authors' code: fixed seed-0 sampling)
from systems_pbc import convection_diffusion  # noqa: E402  (authors' Fourier reference solver)

XGRID = 256  # number of x points, x in [0, 2pi) (endpoint excluded)
NT = 100     # number of t points, t in [0, 1] (both ends included)


def grid(xgrid=XGRID, nt=NT):
    """Fixed evaluation grid, same ordering as the authors' X_star (t-major rows)."""
    x = np.linspace(0, 2 * np.pi, xgrid, endpoint=False).reshape(-1, 1)
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    return x, t, X_star


def exact_u(beta, X):
    """Analytic solution sin(x - beta t) at points X (n, 2) -> (n, 1)."""
    return np.sin(X[:, 0:1] - beta * X[:, 1:2])


def training_data(n_f, xgrid=XGRID, nt=NT):
    """Initial, boundary and collocation points exactly as in pbc_examples/main_pbc.py.

    Collocation points are drawn by the authors' sample_random, which always uses seed 0, so
    they are identical for every run; the run seed only changes the network initialisation.
    """
    x, t, _ = grid(xgrid, nt)
    X, T = np.meshgrid(x, t)

    # interior points: drop t = 0 and the x = 0 boundary column
    X_nb, T_ni = np.meshgrid(x[1:], t[1:])
    X_interior = np.hstack((X_nb.flatten()[:, None], T_ni.flatten()[:, None]))
    X_f = sample_random(X_interior, n_f)

    # initial condition: the authors take it from their FFT solution at t = 0 (= sin x)
    u_vals = convection_diffusion('sin(x)', 0.0, 1.0, 0, xgrid, nt)
    exact_grid = u_vals.reshape(nt, xgrid)
    X_ic = np.hstack((X[0:1, :].T, T[0:1, :].T))
    u_ic = exact_grid[0:1, :].T

    # periodic boundary pairs: x = 0 and x = 2pi at every grid time
    bc_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
    bc_ub = np.hstack((np.full_like(t, 2 * np.pi), t))

    return {"X_f": X_f, "X_ic": X_ic, "u_ic": u_ic, "bc_lb": bc_lb, "bc_ub": bc_ub}
