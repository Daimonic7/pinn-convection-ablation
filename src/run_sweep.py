"""Run a grid of ablation configurations in parallel, appending one row per run to a CSV.

Resumable: any configuration whose run_id (a hash of its full config) is already in the CSV is
skipped, so the sweep can be stopped and restarted at any time. Each worker uses one torch thread.

Examples (from the repo root):
  python src/run_sweep.py                                   # main 2x2x2 sweep (defaults below)
  python src/run_sweep.py --arms C1W0H0 --betas 20,30,40 --budget 125000 --tag repro_curriculum
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALL_ARMS = [f"C{c}W{w}H{h}" for c, w, h in itertools.product([0, 1], repeat=3)]


def _init_worker():
    import torch
    torch.set_num_threads(1)


def git_commit():
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--betas", default="1,5,10,15,20,25,30,35,40,45,50")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--arms", default=",".join(ALL_ARMS))
    p.add_argument("--n-f", type=int, default=1000)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--tag", default="main")
    p.add_argument("--out", default=str(REPO / "results" / "runs.csv"))
    p.add_argument("--save-preds-seeds", default="0", help="seeds whose predictions are saved")
    p.add_argument("--curriculum-dbeta", type=float, default=5.0, help="beta step between stages")
    p.add_argument("--lr", type=float, default=1.0, help="L-BFGS learning rate")
    args = p.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from train import FIELDNAMES, RunConfig, run_config_dict

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_dir = out.parent / "preds"
    pred_dir.mkdir(exist_ok=True)
    prog_dir = out.parent / "progress"          # live per-worker heartbeats for src/watch.py
    prog_dir.mkdir(exist_ok=True)
    os.environ["PINN_PROGRESS_DIR"] = str(prog_dir)   # inherited by the worker processes
    (prog_dir / f"sweep_{args.tag}.json").write_text(json.dumps(
        {"tag": args.tag, "pid": os.getpid(), "started": time.time(), "out": str(out)}))
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {r["run_id"] for r in csv.DictReader(f)}

    betas = [float(b) for b in args.betas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    save_seeds = {int(s) for s in args.save_preds_seeds.split(",") if s != ""}

    cfgs = []
    planned = 0
    for seed in seeds:                       # seed-major: all arms at seed 0 finish first
        block = []
        for arm, beta in itertools.product(arms, betas):
            c, w, h = (arm[1] == "1", arm[3] != "0", arm[5] != "0")
            cfg = RunConfig(beta=beta, seed=seed, curriculum=c, causal=w, hard=h,
                            hard_ansatz="exp" if arm[5] == "2" else "t",
                            causal_norm="beta2" if arm[3] == "2" else "none",
                            curriculum_dbeta=args.curriculum_dbeta, lr=args.lr,
                            n_f=args.n_f, budget=args.budget, tag=args.tag)
            planned += 1
            if cfg.run_id not in done:
                block.append(cfg)
        block.sort(key=lambda k: (-(k.curriculum + k.causal), -k.beta))   # long jobs first
        cfgs += block

    commit = git_commit()
    n = len(cfgs)
    print(f"{n} runs to do ({len(done)} already in {out.name}); planned {planned}; "
          f"{args.workers} workers; commit {commit}", flush=True)
    (prog_dir / f"sweep_{args.tag}.json").write_text(json.dumps(
        {"tag": args.tag, "pid": os.getpid(), "started": time.time(), "out": str(out),
         "to_do": n, "planned": planned, "workers": args.workers}))
    if n == 0:
        return

    t0 = time.time()
    new_file = not out.exists()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = {}
        for cfg in cfgs:
            pred = str(pred_dir / f"{cfg.run_id}.npz") if cfg.seed in save_seeds else None
            d = asdict(cfg)
            futs[ex.submit(run_config_dict, d, pred)] = cfg
        for k, fut in enumerate(as_completed(futs), 1):
            cfg = futs[fut]
            try:
                row = fut.result()
            except Exception:
                with open(out.parent / "errors.log", "a") as f:
                    f.write(f"{time.ctime()} {cfg}\n{traceback.format_exc()}\n")
                print(f"[{k}/{n}] ERROR {cfg.arm} beta={cfg.beta} seed={cfg.seed} (see errors.log)",
                      flush=True)
                continue
            row["git_commit"] = commit
            with open(out, "a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=FIELDNAMES)
                if new_file:
                    wr.writeheader()
                    new_file = False
                wr.writerow(row)
            el = time.time() - t0
            print(f"[{k}/{n}] {row['arm']} beta={row['beta']:>4} seed={row['seed']} "
                  f"rel_l2={row['rel_l2']:.3e} evals={row['loss_evals']} {row['train_s']:.0f}s "
                  f"| elapsed {el/60:.1f} min, ETA {el/k*(n-k)/60:.0f} min", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
