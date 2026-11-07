"""Live terminal dashboard for the running sweeps (Ctrl+C to quit; the sweeps keep running).

    python src/watch.py              # refreshes every 2 s
    python src/watch.py --once       # print one snapshot and exit

Shows: overall progress bars with an ETA per batch, what each worker is doing right now
(stage, loss evaluations used / budget, current loss), the most recent results, and a live
results grid (mean rel. L2 per arm x beta). Reads only files the sweeps write:
results/runs.csv, results/authors_script.csv, results/progress/*.json, results/logs/*.log.
"""
import argparse
import csv
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
PROG = RES / "progress"
LOGS = RES / "logs"
ARMS = ["C0W0H0", "C1W0H0", "C0W1H0", "C1W1H0", "C0W0H1", "C1W0H1", "C0W1H1", "C1W1H1"]
LABEL = {"C0W0": "vanilla", "C1W0": "curriculum", "C0W1": "causal", "C1W1": "curric+causal"}
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def label(arm):
    """C1W2H1 -> 'curric+causal/β² +hard'. W2 = beta^2-normalised causal, H2 = exp time factor."""
    c, w, h = arm[1] == "1", arm[3], arm[5]
    causal = "" if w == "0" else ("causal" if w == "1" else "causal/β²")
    if c and causal:
        base = "curric+" + causal
    else:
        base = "curriculum" if c else (causal or "vanilla")
    return base + {"0": "", "1": " +hard", "2": " +hard(exp)"}[h]


def read_rows(path):
    if not path.exists():
        return []
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    good = []
    for r in rows:                      # tolerate a half-written last line
        try:
            r["rel_l2"] = float(r["rel_l2"])
            r["beta"] = float(r["beta"])
            good.append(r)
        except (KeyError, TypeError, ValueError):
            pass
    return good


def planned_total(log_name, default, rows_done=0, meta=None):
    """Planned size of a batch: from the sweep's progress JSON, else its log's start line."""
    if meta and "planned" in meta:
        return meta["planned"]
    p = LOGS / log_name
    if not p.exists():
        return default
    txt = p.read_text(errors="ignore")
    m = re.findall(r"planned (\d+)", txt)
    if m:
        return int(m[-1])
    m = re.findall(r"(\d+) (?:authors'-script )?runs to do", txt)
    return max(int(m[0]), rows_done) if m else default


def heartbeats(max_age=90):
    out = []
    now = time.time()
    for p in PROG.glob("worker_*.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if d.get("status") == "running" and now - d.get("updated", 0) < max_age:
            out.append(d)
    return sorted(out, key=lambda d: d["started"])


def fmt_dur(s):
    if s is None:
        return "?"
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


def err_style(v):
    return "green" if v <= 1e-2 else ("yellow" if v <= 1e-1 else "red")


def eta_main(rows, hb, total, workers):
    """Remaining work = sum over not-yet-done configs of the latest observed train time for the
    same (arm, beta), minus progress of in-flight runs, divided by the number of workers."""
    if not rows:
        return None
    latest = {}
    for r in sorted(rows, key=lambda r: r["timestamp"]):
        latest[(r["arm"], r["beta"])] = float(r["train_s"])
    arms = sorted({r["arm"] for r in rows})
    betas = sorted({r["beta"] for r in rows})
    n_seeds = max(1, round(total / max(1, len(arms) * len(betas))))
    done = {(r["arm"], r["beta"], int(r["seed"])) for r in rows}
    running = {(h["arm"], h["beta"], h["seed"]): h for h in hb if h["tag"] == "main"}
    fallback = statistics.median(latest.values())
    work = 0.0
    for a in arms:
        for b in betas:
            for s in range(n_seeds):
                if (a, b, s) in done:
                    continue
                c = latest.get((a, b), fallback)
                h = running.get((a, b, s))
                if h:
                    c *= max(0.0, 1 - h["evals"] / max(1, h["budget"]))
                work += c
    return work / max(1, workers)


def build(console_width):
    now = time.time()
    all_rows = read_rows(RES / "runs.csv")
    main_rows = [r for r in all_rows if r.get("tag") == "main"]
    repro_rows = [r for r in all_rows if r.get("tag") == "repro_curriculum"]
    other_rows = [r for r in all_rows if r.get("tag") not in ("main", "repro_curriculum")]
    auth_rows = read_rows(RES / "authors_script.csv")
    hb = heartbeats()
    sweep_meta = {}
    for p in PROG.glob("sweep_*.json"):
        try:
            d = json.loads(p.read_text())
            sweep_meta[d["tag"]] = d
        except (OSError, ValueError, KeyError):
            pass

    # ---- header: time, power, CPU
    head = Text(f"PINN convection ablation — {datetime.now():%H:%M:%S}", style="bold")
    try:
        import psutil
        bat = psutil.sensors_battery()
        cpu = psutil.cpu_percent(interval=None if _LIVE else 0.5)
        if bat is not None:
            power = ("on AC" if bat.power_plugged else "ON BATTERY — runs are throttled, plug in!")
            head.append(f"   power: {power} ({bat.percent:.0f}%)",
                        style="green" if bat.power_plugged else "bold red")
        head.append(f"   CPU {cpu:.0f}%", style="cyan")
    except Exception:
        pass

    # ---- progress bars per batch
    bars = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    for col, kw in (("batch", {}), ("progress", {"ratio": 3}), ("done", {"justify": "right"}),
                    ("ETA", {"justify": "right"})):
        bars.add_column(col, **kw)
    main_total = planned_total("sweep_main.log", 440, len(main_rows), sweep_meta.get("main"))
    workers = sweep_meta.get("main", {}).get("workers", 9)
    eta = eta_main(main_rows, hb, main_total, workers) if len(main_rows) < main_total else 0
    batches = [("main ablation", len(main_rows), main_total, eta),
               ("authors' script", len(auth_rows),
                planned_total("authors_script.log", 25, len(auth_rows)), None),
               ("uncapped curriculum", len(repro_rows),
                planned_total("repro_curriculum.log", 15, len(repro_rows),
                              sweep_meta.get("repro_curriculum")), None)]
    for tag, meta in sorted(sweep_meta.items()):        # any further sweeps, e.g. "followup"
        if tag in ("main", "repro_curriculum"):
            continue
        n_done = sum(r["tag"] == tag for r in other_rows)
        batches.append((tag, n_done, meta.get("planned", n_done), None))
    for name, done, tot, eta_s in batches:
        frac = done / tot if tot else 0
        bars.add_row(name, ProgressBar(total=max(tot, 1), completed=done, width=None),
                     f"{done}/{tot} ({100 * frac:.0f}%)",
                     "done" if done >= tot else (fmt_dur(eta_s) if eta_s is not None else "-"))

    # ---- in-flight runs
    live = Table(title="running now", box=box.SIMPLE_HEAD, expand=True, title_justify="left")
    for col, kw in (("arm", {}), ("β", {"justify": "right"}), ("seed", {"justify": "right"}),
                    ("stage", {"justify": "right"}), ("loss evals", {"ratio": 2}),
                    ("", {"justify": "right"}), ("loss", {"justify": "right"}),
                    ("ε", {"justify": "right"}), ("elapsed", {"justify": "right"})):
        live.add_column(col, **kw)
    for h in hb:
        stage = f"{h['stage']}/{h['n_stages']} (β={h['stage_beta']:g})" if h["n_stages"] > 1 else "1/1"
        live.add_row(label(h["arm"]) if h["tag"] == "main" else f"{label(h['arm'])} [{h['tag']}]",
                     f"{h['beta']:g}", str(h["seed"]), stage,
                     ProgressBar(total=h["budget"], completed=min(h["evals"], h["budget"]), width=None),
                     f"{h['evals']}/{h['budget']}",
                     "-" if h["loss"] is None else f"{h['loss']:.2e}",
                     "-" if h.get("eps") is None else f"{h['eps']:g}",
                     fmt_dur(now - h["started"]))
    if not hb:
        live.add_row("(no heartbeats yet: runs started before the heartbeat code, or idle)",
                     "", "", "", "", "", "", "", "")

    # ---- recent results
    recent = Table(title="latest results", box=box.SIMPLE_HEAD, expand=True, title_justify="left")
    for col in ("time", "batch", "arm", "β", "seed", "rel. L2", "evals", "train"):
        recent.add_column(col, justify="right" if col in ("β", "seed", "rel. L2", "evals", "train") else "left")
    for r in sorted(all_rows, key=lambda r: r["timestamp"])[-8:][::-1]:
        recent.add_row(r["timestamp"][11:], r["tag"], label(r["arm"]), f"{r['beta']:g}", r["seed"],
                       Text(f"{r['rel_l2']:.2e}", style=err_style(r["rel_l2"])), r["loss_evals"],
                       fmt_dur(float(r["train_s"])))

    # ---- live results grid: mean rel. L2 per arm x beta
    betas = sorted({r["beta"] for r in main_rows})
    grid = Table(title="mean rel. L2 so far (main ablation; green ≤ 1e-2, yellow ≤ 1e-1, red > 1e-1; "
                       "superscript = seeds done)", box=box.SIMPLE_HEAD, title_justify="left")
    grid.add_column("arm")
    for b in betas:
        grid.add_column(f"β={b:g}", justify="right")
    for arm in ARMS:
        cells = []
        for b in betas:
            v = [r["rel_l2"] for r in main_rows if r["arm"] == arm and r["beta"] == b]
            if v:
                m = sum(v) / len(v)
                cells.append(Text(f"{m:.0e}", style=err_style(m)) +
                             Text(" " + str(len(v)).translate(SUPERSCRIPT), style="dim"))
            else:
                cells.append(Text("·", style="dim"))
        grid.add_row(label(arm), *cells)

    return Group(Panel(head, box=box.ROUNDED), bars, live, recent, grid)


_LIVE = False


def main():
    global _LIVE
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    console = Console()
    _LIVE = not a.once
    if a.once:
        console.print(build(console.width))
        return
    with Live(build(console.width), console=console, refresh_per_second=4, screen=True) as live:
        try:
            while True:
                time.sleep(a.interval)
                live.update(build(console.width))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
