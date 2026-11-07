"""Run the authors' unmodified script (pbc_examples/main_pbc.py) over seeds, for the reproduction table.

Each run is a separate process with one thread; results go to results/authors_script.csv
(resumable: (beta, seed, n_f) combinations already present are skipped).

Usage:  python src/run_authors_script.py --betas 1,10,20,30,40 --seeds 0,1,2,3,4 --n-f 1000
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIELDS = ["beta", "seed", "n_f", "rel_l2", "abs_err", "linf_rel", "last_logged_epoch", "wall_s",
          "timestamp"]


def run(beta, seed, n_f):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, "-u", "main_pbc.py", "--system", "convection",
                          "--beta", str(beta), "--seed", str(seed), "--N_f", str(n_f)],
                         cwd=REPO / "pbc_examples", env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    out = res.stdout

    def grab(name):
        m = re.search(rf"Error u {name}: ([0-9.eE+-]+)", out)
        return float(m.group(1)) if m else float("nan")

    epochs = re.findall(r"^epoch (\d+)", out, flags=re.M)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-2000:])
    return {"beta": beta, "seed": seed, "n_f": n_f, "rel_l2": grab("rel"), "abs_err": grab("abs"),
            "linf_rel": grab("linf"), "last_logged_epoch": int(epochs[-1]) if epochs else -1,
            "wall_s": round(wall, 2), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--betas", default="1,10,20,30,40")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-f", type=int, default=1000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default=str(REPO / "results" / "authors_script.csv"))
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(float(r["beta"]), int(r["seed"]), int(r["n_f"])) for r in csv.DictReader(f)}
    jobs = [(float(b), int(s), a.n_f) for s in a.seeds.split(",") for b in a.betas.split(",")
            if (float(b), int(s), a.n_f) not in done]
    print(f"{len(jobs)} authors'-script runs to do", flush=True)
    new_file = not out.exists()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run, *j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            with open(out, "a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    wr.writeheader()
                    new_file = False
                wr.writerow(row)
            print(f"[{k}/{len(jobs)}] beta={row['beta']} seed={row['seed']} rel_l2={row['rel_l2']:.3e} "
                  f"{row['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
