"""Train one PINN configuration of the 2x2x2 ablation on 1D convection and return its metrics.

Factors (each on/off):
  curriculum  C - curriculum warm-start (Krishnapriyan et al. 2021, Sec. 5.1): train at an easier
                  beta, then warm-start the next, harder beta from those weights, up to the target.
  causal      W - causal residual weighting (Wang, Sankaran & Perdikaris, "Respecting causality for
                  training physics-informed neural networks", CMAME 2024 / arXiv:2203.07404):
                  the residual of time chunk i gets weight w_i = exp(-eps * sum_{k<i} L_k), so late
                  times only count once earlier times are fitted.
  hard        H - hard periodic-BC/IC network (see models.HardPINN).

Compute matching: every run gets the same budget of loss evaluations (closure calls, i.e.
forward + backward passes; this is the unit the authors' "epoch" counter counts). Causal-weight
updates also cost one evaluation each. Runs can stop earlier when L-BFGS meets its tolerances.

The vanilla arm (C0 W0 H0) is the authors' training procedure: same network and initialisation,
same points, same loss, one L-BFGS call with the authors' settings (lr 1.0, history 50, tolerances
1e-7, strong Wolfe line search), capped at `budget` evaluations.
"""
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

import problem
from models import HardPINN, SoftPINN
from utils import set_seed  # authors' seeding helper (random, numpy, torch)


@dataclass
class RunConfig:
    beta: float
    seed: int
    curriculum: bool = False
    causal: bool = False
    hard: bool = False
    n_f: int = 1000
    budget: int = 10000              # max loss evaluations for the whole run
    lr: float = 1.0
    history_size: int = 50
    tol: float = 1e-7
    curriculum_dbeta: float = 5.0    # beta step between curriculum stages
    causal_chunks: int = 20          # number of time chunks for causal weights
    causal_eps: tuple = (1e-2, 1e-1, 1.0, 10.0, 100.0)
    causal_delta: float = 0.99       # advance eps once min weight exceeds this
    causal_round: int = 500          # evaluations per L-BFGS round with frozen weights
    tag: str = "main"
    hard_ansatz: str = "t"           # "t": u = sin x + t N (arm H1); "exp": (1 - e^{-beta t}) N (H2)
    causal_norm: str = "none"        # "beta2": weights from chunk residuals / beta^2 (arm W2)

    @property
    def arm(self):
        h = 0 if not self.hard else (1 if self.hard_ansatz == "t" else 2)
        w = 0 if not self.causal else (1 if self.causal_norm == "none" else 2)
        return f"C{int(self.curriculum)}W{w}H{h}"

    @property
    def run_id(self):
        d = asdict(self)
        d["causal_eps"] = list(d["causal_eps"])
        for k, default in (("hard_ansatz", "t"), ("causal_norm", "none")):
            if d[k] == default:       # fields added later: keep earlier run_ids unchanged
                d.pop(k)
        key = json.dumps(d, sort_keys=True)
        return hashlib.sha1(key.encode()).hexdigest()[:12]


def curriculum_betas(beta_target, dbeta):
    """Evenly spaced stages ending at the target, steps of at most dbeta (e.g. 30 -> 5,10,...,30)."""
    k = max(1, math.ceil(beta_target / dbeta - 1e-9))
    return [beta_target * i / k for i in range(1, k + 1)]


def _col(a, requires_grad=False):
    return torch.tensor(a, dtype=torch.float32, requires_grad=requires_grad)


class Trainer:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        data = problem.training_data(cfg.n_f)       # authors' sampling (fixed seed 0) ...
        set_seed(cfg.seed)                          # ... then the run seed, as in main_pbc.py
        self.model = (HardPINN(time_factor=cfg.hard_ansatz, beta=cfg.beta) if cfg.hard
                      else SoftPINN())

        X_f = data["X_f"]
        self.x_f = _col(X_f[:, 0:1], True)
        self.t_f = _col(X_f[:, 1:2], True)
        self.x_ic = _col(data["X_ic"][:, 0:1])
        self.t_ic = _col(data["X_ic"][:, 1:2])
        self.u_ic = _col(data["u_ic"])
        self.x_lb = _col(data["bc_lb"][:, 0:1])
        self.t_lb = _col(data["bc_lb"][:, 1:2])
        self.x_ub = _col(data["bc_ub"][:, 0:1])
        self.t_ub = _col(data["bc_ub"][:, 1:2])

        # time-chunk index of every collocation point (chunks of equal length on (0, 1])
        m = cfg.causal_chunks
        idx = np.minimum((X_f[:, 1] * m).astype(int), m - 1)
        self.chunk = torch.tensor(idx, dtype=torch.long)
        self.chunk_count = torch.bincount(self.chunk, minlength=m).clamp(min=1).float()

        self.evals = 0
        self.lbfgs_iters = 0
        self.n_rounds = 0
        self.final_eps = float("nan")
        self.stop_reason = "budget"
        self.last = {}

        # live progress for src/watch.py: one small JSON file per worker process
        prog_dir = os.environ.get("PINN_PROGRESS_DIR")
        self._hb_path = os.path.join(prog_dir, f"worker_{os.getpid()}.json") if prog_dir else None
        self._hb_last = 0.0
        self._t_start = time.time()
        self.stage_k, self.n_stages, self.stage_beta = 0, 1, cfg.beta

    def heartbeat(self, status="running", force=False, extra=None):
        """Write this run's progress at most every 2 s (atomic replace; cheap)."""
        if self._hb_path is None:
            return
        now = time.time()
        if not force and now - self._hb_last < 2.0:
            return
        self._hb_last = now
        loss = self.last.get("loss")
        d = {"pid": os.getpid(), "status": status, "run_id": self.cfg.run_id, "tag": self.cfg.tag,
             "arm": self.cfg.arm, "beta": self.cfg.beta, "seed": self.cfg.seed,
             "stage": self.stage_k + 1, "n_stages": self.n_stages, "stage_beta": self.stage_beta,
             "evals": self.evals, "budget": self.cfg.budget,
             "loss": None if loss is None else float(loss),
             "eps": None if math.isnan(self.final_eps) else self.final_eps,
             "started": self._t_start, "updated": now}
        if extra:
            d.update(extra)
        tmp = self._hb_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(d, f)
            os.replace(tmp, self._hb_path)
        except OSError:
            pass  # progress reporting must never break a run

    # ----- loss pieces -------------------------------------------------------------------------
    def residual(self, beta):
        u = self.model(self.x_f, self.t_f)
        ones = torch.ones_like(u)
        u_t = torch.autograd.grad(u, self.t_f, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, self.x_f, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        return u_t + beta * u_x

    def chunk_losses(self, f):
        """Mean squared residual per time chunk, shape (chunks,)."""
        sq = (f ** 2).squeeze(1)
        sums = torch.zeros(self.cfg.causal_chunks, dtype=sq.dtype).index_add(0, self.chunk, sq)
        return sums / self.chunk_count

    def causal_weights(self, beta, eps):
        f = self.residual(beta)
        L = self.chunk_losses(f).detach()
        if self.cfg.causal_norm == "beta2":
            # residual in units of beta (nondimensional time beta*t), so eps means the same at
            # every beta and for soft/hard models; only the weights change, not the loss
            L = L / beta ** 2
        prefix = torch.cat([torch.zeros(1), torch.cumsum(L, 0)[:-1]])
        self.evals += 1  # counts against the budget
        return torch.exp(-eps * prefix)

    def losses(self, beta, weights=None):
        f = self.residual(beta)
        if self.cfg.hard:
            loss_ic = torch.zeros(())
            loss_bc = torch.zeros(())
        else:
            loss_ic = torch.mean((self.u_ic - self.model(self.x_ic, self.t_ic)) ** 2)
            loss_bc = torch.mean((self.model(self.x_lb, self.t_lb) - self.model(self.x_ub, self.t_ub)) ** 2)
        if weights is None:
            loss_f = torch.mean(f ** 2)
        else:
            loss_f = torch.mean(weights * self.chunk_losses(f))
        return loss_ic + loss_bc + loss_f, (loss_ic, loss_bc, loss_f)

    # ----- optimisation ------------------------------------------------------------------------
    def run_lbfgs(self, beta, max_evals, weights=None):
        """One fresh L-BFGS call on a fixed objective; returns evaluations used."""
        cfg = self.cfg
        params = list(self.model.parameters())
        opt = torch.optim.LBFGS(params, lr=cfg.lr, max_iter=max_evals, max_eval=max_evals,
                                history_size=cfg.history_size, tolerance_grad=cfg.tol,
                                tolerance_change=cfg.tol, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss, parts = self.losses(beta, weights)
            loss.backward()
            self.evals += 1
            self.last = {"loss": loss.detach(), "ic": parts[0].detach(),
                         "bc": parts[1].detach(), "f": parts[2].detach()}
            self.heartbeat()
            return loss

        start = self.evals
        opt.step(closure)
        self.lbfgs_iters += opt.state[params[0]].get("n_iter", 0)
        self.n_rounds += 1
        return self.evals - start

    def train_stage(self, beta, budget):
        cfg = self.cfg
        if not cfg.causal:
            used = self.run_lbfgs(beta, budget)
            if used < budget:
                self.stop_reason = "tolerance"
            return used

        start = self.evals
        eps_i = 0
        while self.evals - start < budget:
            w = self.causal_weights(beta, cfg.causal_eps[eps_i])
            while w.min() > cfg.causal_delta and eps_i < len(cfg.causal_eps) - 1:
                eps_i += 1
                w = self.causal_weights(beta, cfg.causal_eps[eps_i])
            self.final_eps = cfg.causal_eps[eps_i]
            if eps_i == len(cfg.causal_eps) - 1 and w.min() > cfg.causal_delta:
                self.stop_reason = "causal_converged"
                break
            left = budget - (self.evals - start)
            if left <= 0:
                break
            round_budget = min(cfg.causal_round, left)
            used = self.run_lbfgs(beta, round_budget, weights=w)
            if used < round_budget and used <= 5:
                # L-BFGS made no progress on this objective and eps did not advance: converged
                self.stop_reason = "tolerance"
                break
        return self.evals - start

    def train(self):
        cfg = self.cfg
        stages = curriculum_betas(cfg.beta, cfg.curriculum_dbeta) if cfg.curriculum else [cfg.beta]
        self.n_stages = len(stages)
        for k, beta_k in enumerate(stages):
            self.stage_k, self.stage_beta = k, beta_k
            if hasattr(self.model, "beta"):
                self.model.beta = beta_k      # H2's time factor follows the stage's beta
            self.heartbeat(force=True)
            remaining = cfg.budget - self.evals
            if remaining <= 0:
                self.stop_reason = "budget"
                break
            share = remaining // (len(stages) - k)   # unused budget rolls into later stages
            self.stop_reason = "budget"
            self.train_stage(beta_k, max(share, 1))
        if self.evals >= cfg.budget:
            self.stop_reason = "budget"
        if hasattr(self.model, "beta"):
            self.model.beta = cfg.beta        # evaluate at the target beta
        return stages

    # ----- evaluation ----------------------------------------------------------------------
    def predict(self, X):
        x = _col(X[:, 0:1])
        t = _col(X[:, 1:2])
        with torch.no_grad():
            return self.model(x, t).numpy().astype(np.float64)

    def final_unweighted_losses(self, beta):
        """Unweighted IC / BC / residual MSE at the end of training (comparable across arms)."""
        f = self.residual(beta).detach()
        with torch.no_grad():
            ic = torch.mean((self.u_ic - self.model(self.x_ic, self.t_ic)) ** 2)
            bc = torch.mean((self.model(self.x_lb, self.t_lb) - self.model(self.x_ub, self.t_ub)) ** 2)
        return float(ic), float(bc), float(torch.mean(f ** 2))


FIELDNAMES = [
    "run_id", "tag", "arm", "curriculum", "causal", "hard", "beta", "seed", "n_f", "budget",
    "rel_l2", "abs_err", "linf_rel", "loss_evals", "lbfgs_iters", "lbfgs_calls", "n_stages",
    "stop_reason", "final_eps", "final_loss_ic", "final_loss_bc", "final_loss_f",
    "train_s", "eval_s", "wall_s", "torch", "threads", "timestamp", "git_commit", "config_json",
]


def run_config_dict(cfg_dict, save_pred_path=None):
    """Picklable entry point for worker processes."""
    d = dict(cfg_dict)
    d["causal_eps"] = tuple(d["causal_eps"])
    return run_one(RunConfig(**d), save_pred_path)


def run_one(cfg: RunConfig, save_pred_path=None):
    """Train + evaluate one configuration; returns a flat dict for results/runs.csv."""
    t0 = time.perf_counter()
    tr = Trainer(cfg)
    stages = tr.train()
    t1 = time.perf_counter()

    _, _, X_star = problem.grid()
    u = problem.exact_u(cfg.beta, X_star)
    u_pred = tr.predict(X_star)
    rel_l2 = float(np.linalg.norm(u - u_pred, 2) / np.linalg.norm(u, 2))
    abs_err = float(np.mean(np.abs(u - u_pred)))
    linf_rel = float(np.linalg.norm(u - u_pred, np.inf) / np.linalg.norm(u, np.inf))
    ic, bc, fres = tr.final_unweighted_losses(cfg.beta)
    t2 = time.perf_counter()
    tr.heartbeat(status="done", force=True, extra={"rel_l2": rel_l2})

    if save_pred_path is not None:
        np.savez_compressed(save_pred_path,
                            u_pred=u_pred.reshape(problem.NT, problem.XGRID).astype(np.float32))

    cfg_d = asdict(cfg)
    cfg_d["causal_eps"] = list(cfg_d["causal_eps"])
    return {
        "run_id": cfg.run_id, "tag": cfg.tag, "arm": cfg.arm,
        "curriculum": int(cfg.curriculum), "causal": int(cfg.causal), "hard": int(cfg.hard),
        "beta": cfg.beta, "seed": cfg.seed, "n_f": cfg.n_f, "budget": cfg.budget,
        "rel_l2": rel_l2, "abs_err": abs_err, "linf_rel": linf_rel,
        "loss_evals": tr.evals, "lbfgs_iters": tr.lbfgs_iters, "lbfgs_calls": tr.n_rounds,
        "n_stages": len(stages), "stop_reason": tr.stop_reason, "final_eps": tr.final_eps,
        "final_loss_ic": ic, "final_loss_bc": bc, "final_loss_f": fres,
        "train_s": round(t1 - t0, 3), "eval_s": round(t2 - t1, 3), "wall_s": round(t2 - t0, 3),
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_json": json.dumps(cfg_d, sort_keys=True),
    }
