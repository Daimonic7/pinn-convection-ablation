"""Classical baseline for u_t + beta u_x = 0 with periodic BCs: Fourier pseudo-spectral solvers.

Two methods on the same 256 x 100 grid the PINNs are evaluated on:
  * "spectral_rk4": method of lines. u_x is computed spectrally (FFT, multiply by ik, inverse FFT;
    Nyquist mode zeroed) and time is advanced with classical RK4. The step size keeps
    beta * k_max * dt <= 2.5, inside RK4's stability limit on the imaginary axis (about 2.83).
  * "fourier_exact": the authors' reference solver (pbc_examples/systems_pbc.convection_diffusion),
    which advances each Fourier mode exactly by exp(-i beta k t).
Errors are relative L2 against the analytic sin(x - beta t); times are the best of 5 repeats.

Usage:  python src/spectral_baseline.py   ->  results/spectral_baseline.csv
"""
import csv
import math
import time
from pathlib import Path

import numpy as np

import problem
from systems_pbc import convection_diffusion  # authors' code

REPO = Path(__file__).resolve().parents[1]
BETAS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def spectral_rk4(beta, xgrid=problem.XGRID, nt=problem.NT, c_max=2.5):
    n = xgrid
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ik = 1j * np.fft.fftfreq(n, d=1.0 / n)
    ik[n // 2] = 0.0                                   # zero the Nyquist derivative
    dt_out = 1.0 / (nt - 1)
    m = max(1, math.ceil(beta * (n / 2) * dt_out / c_max))  # RK4 sub-steps per output interval
    dt = dt_out / m

    def rhs(u):
        return -beta * np.real(np.fft.ifft(ik * np.fft.fft(u)))

    u = np.sin(x)
    out = np.empty((nt, n))
    out[0] = u
    for j in range(1, nt):
        for _ in range(m):
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * dt * k1)
            k3 = rhs(u + 0.5 * dt * k2)
            k4 = rhs(u + dt * k3)
            u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[j] = u
    return out, m * (nt - 1)


def fourier_exact(beta, xgrid=problem.XGRID, nt=problem.NT):
    return convection_diffusion('sin(x)', 0.0, beta, 0, xgrid, nt).reshape(nt, xgrid), 0


def best_time(fn, beta, repeats=5):
    best, res = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn(beta)
        best = min(best, time.perf_counter() - t0)
    return best, res


def main():
    _, _, X_star = problem.grid()
    rows = []
    for beta in BETAS:
        u = problem.exact_u(beta, X_star).reshape(problem.NT, problem.XGRID)
        for name, fn in (("spectral_rk4", spectral_rk4), ("fourier_exact", fourier_exact)):
            wall, (sol, steps) = best_time(fn, beta)
            rel = float(np.linalg.norm(sol - u) / np.linalg.norm(u))
            rows.append({"beta": beta, "method": name, "rel_l2": rel, "time_steps": steps,
                         "wall_s": wall})
            print(f"beta={beta:>3} {name:<14} rel_l2={rel:.2e}  steps={steps:>5}  {1000*wall:7.2f} ms")
    out = REPO / "results" / "spectral_baseline.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
