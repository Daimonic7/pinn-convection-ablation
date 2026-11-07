"""Network variants for the ablation.

SoftPINN  - the authors' model: u = DNN([x, t]) with the authors' DNN class
            (pbc_examples/net_pbc.py, Krishnapriyan et al. 2021, MIT license). IC and periodic BC
            are enforced softly through loss terms.
HardPINN  - hard periodic-BC/IC model:
                u(x, t) = sin(x) + t * DNN([sin x, cos x, t])
            * feeding x only through (sin x, cos x) makes u exactly 2*pi-periodic in x
              (periodic embedding, e.g. Dong & Ni, J. Comput. Phys. 2021);
            * the factor t vanishes at t = 0, so u(x, 0) = sin(x) exactly
              (trial-solution form u = g + D * NN, Lagaris et al. 1998; CEE 292 PINN slides, slide 18).
            With both constraints built in, only the PDE residual is left in the loss.

            time_factor="exp" (arm H2, added after the first sweep results) uses
                u(x, t) = sin(x) + (1 - exp(-beta t)) * DNN([sin x, cos x, t]).
            Why: with D = t, the PDE at t = 0 forces DNN(x, 0) = -beta cos x, an output of size beta.
            With D = 1 - exp(-beta t), D'(0) = beta, so it only needs DNN(x, 0) = -cos x, of size 1
            for every beta, while the IC stays exact. beta is the current (curriculum-stage) beta.
"""
import torch

import problem  # noqa: F401  (puts the authors' pbc_examples/ on sys.path)
from net_pbc import DNN  # noqa: E402  (authors' code)

HIDDEN = (50, 50, 50, 50)


class SoftPINN(torch.nn.Module):
    def __init__(self, hidden=HIDDEN, activation="tanh"):
        super().__init__()
        self.dnn = DNN([2, *hidden, 1], activation)

    def forward(self, x, t):
        return self.dnn(torch.cat([x, t], dim=1))


class HardPINN(torch.nn.Module):
    def __init__(self, hidden=HIDDEN, activation="tanh", time_factor="t", beta=1.0):
        super().__init__()
        assert time_factor in ("t", "exp")
        self.dnn = DNN([3, *hidden, 1], activation)
        self.time_factor = time_factor
        self.beta = beta  # only used by time_factor="exp"; the trainer sets it per stage

    def forward(self, x, t):
        sx = torch.sin(x)
        z = torch.cat([sx, torch.cos(x), t], dim=1)
        d = t if self.time_factor == "t" else 1.0 - torch.exp(-self.beta * t)
        return sx + d * self.dnn(z)
