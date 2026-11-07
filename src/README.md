# PINN convection ablation — project code (CEE 292, Stanford)

New code for a reproduction + compute-matched ablation of PINN failure on 1D convection,
built on the authors' released code in [`../pbc_examples/`](../pbc_examples/) (left unmodified).

**Credit.**
- Krishnapriyan, Gholami, Zhe, Kirby, Mahoney, *Characterizing possible failure modes in
  physics-informed neural networks*, NeurIPS 2021. Code: MIT license, © 2021 The Regents of the
  University of California / LBNL (see [`../license.txt`](../license.txt), [`../Legal.txt`](../Legal.txt)).
  We import their network class, sampling, seeding and Fourier reference solver.
- Causal weighting: Wang, Sankaran, Perdikaris, *Respecting causality for training
  physics-informed neural networks*, CMAME 2024 (arXiv:2203.07404).
- Hard periodic BC via (sin x, cos x) embedding: Dong & Ni, J. Comput. Phys. 2021. Hard IC via
  the trial solution u = g + D·NN: Lagaris, Likas & Fotiadis 1998.

## Problem and protocol

u_t + β u_x = 0 on [0, 2π) × [0, 1], u(x, 0) = sin x, periodic in x. Exact solution sin(x − βt).

| | setting |
|---|---|
| network | 4 × 50 tanh MLP (authors' `DNN`), float32 |
| optimizer | L-BFGS, lr 1.0, history 50, tolerances 1e-7, strong Wolfe (authors' settings) |
| points | N_f = 1000 collocation (authors' fixed seed-0 sampling), 256 IC points, 100 periodic BC pairs |
| metric | relative L2 error vs sin(x − βt) on the fixed 256 × 100 grid; "solved" = ≤ 1e-2 |
| compute matching | every run capped at **10,000 loss evaluations** (forward + backward passes) |
| grid | β ∈ {1, 5, 10, …, 50}, seeds 0–4, 8 arms |

The 8 arms are all on/off combinations of three factors:

| factor | code | what it does |
|---|---|---|
| curriculum warm-start | `C1` | train at β = β_T·k/K for k = 1..K (K = ⌈β_T/5⌉), each stage starting from the previous weights; the budget is split evenly across stages, with unused budget rolling forward |
| causal residual weighting | `W1` | 20 time chunks; chunk i weighted by exp(−ε Σ_{k<i} L_k), with the weights frozen for 500-evaluation L-BFGS rounds; ε steps 1e-2 → 1e2 when all weights exceed 0.99 |
| hard periodic BC + IC | `H1` | u = sin x + t · N(sin x, cos x, t): exactly periodic, exact IC, so the loss is the residual only |

`C0W0H0` is the authors' vanilla PINN. Its loss and initial gradients match the authors'
`PhysicsInformedNN_pbc` to float32 rounding (`check_setup.py`).

**Post-hoc follow-up arms** (added after seed-0 results showed causal × hard blowing up; tag
`followup`, β ∈ {20, 30, 40}):
- `W2`: causal weights computed from chunk residuals divided by β², so ε is
  scale-free; the loss itself is unchanged.
- `H2`: hard time factor D(t) = 1 − e^{−βt} instead of t (needs a network output of size 1,
  not size β, at t = 0).

Run with `python src/run_sweep.py --arms C0W2H0,C1W2H0,C0W2H1,C1W2H1,C0W0H2,C1W0H2 --betas 20,30,40 --tag followup`.

## Environment

Python 3.12. On Windows-on-ARM, torch comes from PyTorch's index; elsewhere `pip install torch` works.

```
python -m venv venv
venv/Scripts/python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python -m pip install -r src/requirements.txt
```
`src/requirements-lock.txt` is the full `pip freeze` the results were produced with.

## Reproducing everything (run from the repo root)

| step | command | output |
|---|---|---|
| sanity checks | `python src/check_setup.py` | step-0 equivalence with the authors' code, hard-constraint check, per-arm timing |
| main ablation (440 runs) | `python src/run_sweep.py --workers 9` | `results/runs.csv`, `results/preds/*.npz` (seed 0) |
| paper-style curriculum (stages to convergence) | `python src/run_sweep.py --arms C1W0H0 --betas 20,30,40 --budget 125000 --tag repro_curriculum` | rows tagged `repro_curriculum` |
| authors' unmodified script, 5 seeds | `python src/run_authors_script.py` | `results/authors_script.csv` |
| classical baseline | `python src/spectral_baseline.py` | `results/spectral_baseline.csv` |
| dataset export | `python src/make_dataset.py` | `data/convection_benchmark.npz` |
| tables + figures | `python src/analyze.py` | `results/summary.md`, `results/summary_by_arm_beta.csv`, `results/figures/` |

The sweeps are **resumable**: each configuration has a `run_id` (hash of its full config), and
any `run_id` already in `results/runs.csv` is skipped. Each worker uses one thread.

## Files

| file | purpose |
|---|---|
| `problem.py` | grid, exact solution, training points (replicates `main_pbc.py`) |
| `models.py` | `SoftPINN` (authors' model) and `HardPINN` |
| `train.py` | `RunConfig`, the trainer (curriculum / causal / budget logic), `run_one` |
| `run_sweep.py` | parallel, resumable sweep → `results/runs.csv` |
| `run_authors_script.py` | runs `pbc_examples/main_pbc.py` over seeds |
| `spectral_baseline.py` | pseudo-spectral RK4 + exact Fourier solvers |
| `make_dataset.py` | exports the benchmark dataset |
| `analyze.py` | summary tables, factorial effects, figures |
| `check_setup.py` | pre-sweep checks |
| `watch.py` | live terminal dashboard for running sweeps (`python src/watch.py`) |

## `results/runs.csv` columns

`run_id, tag, arm, curriculum, causal, hard, beta, seed, n_f, budget` identify the run.
`rel_l2, abs_err, linf_rel` are errors on the evaluation grid. `loss_evals` (budget unit),
`lbfgs_iters`, `lbfgs_calls`, `n_stages`, `stop_reason` (`budget` / `tolerance` /
`causal_converged`) and `final_eps` describe the optimization. `final_loss_ic/bc/f` are the
unweighted final losses, and `train_s, eval_s, wall_s` are wall-clock times (all runs on one
Snapdragon X Elite CPU, one thread per run, with 11 runs in parallel). `torch, threads,
timestamp, git_commit, config_json` record provenance.
