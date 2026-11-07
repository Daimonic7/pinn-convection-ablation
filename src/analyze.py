"""Aggregate results/runs.csv into tables (results/summary.md, results/summary_by_arm_beta.csv)
and figures (results/figures/*.png). Works on partial results; cell counts are reported.

Usage:  python src/analyze.py
"""
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = RES / "figures"
TARGET = 1e-2

# Paper numbers (Krishnapriyan et al. 2021): vanilla from Fig. 3, curriculum from Table 1.
PAPER_VANILLA = {1.0: 7.84e-3, 10.0: 1.08e-2, 20.0: 7.50e-1, 30.0: 8.97e-1, 40.0: 9.61e-1}
PAPER_CURRIC = {20.0: 9.84e-3, 30.0: 2.02e-2, 40.0: 5.33e-2}

CW_LABEL = {(0, 0): "vanilla", (1, 0): "curriculum", (0, 1): "causal", (1, 1): "curriculum + causal"}
# Reference palette (dataviz skill, references/palette.md), slots 1-4 in fixed order.
CW_COLOR = {(0, 0): "#2a78d6", (1, 0): "#eb6834", (0, 1): "#1baf7a", (1, 1): "#eda100"}
CW_MARKER = {(0, 0): "o", (1, 0): "s", (0, 1): "^", (1, 1): "D"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])

ARMS = [f"C{c}W{w}H{h}" for h, c, w in itertools.product([0, 1], repeat=3)]


def arm_label(arm):
    """C1W2H1 -> 'curriculum + causal/β² + hard'. W2: β²-normalised causal; H2: exp time factor."""
    c, w, h = int(arm[1]), int(arm[3]), int(arm[5])
    base = CW_LABEL[(c, min(w, 1))]
    if w == 2:
        base = base.replace("causal", "causal/β²")
    return base + {0: "", 1: " + hard", 2: " + hard(exp)"}[h]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_runs(tag):
    rows = [r for r in read_csv(RES / "runs.csv") if r["tag"] == tag]
    for r in rows:
        for k in ("curriculum", "causal", "hard", "seed", "loss_evals", "lbfgs_iters", "n_stages"):
            r[k] = int(r[k])
        for k in ("beta", "rel_l2", "train_s"):
            r[k] = float(r[k])
    return rows


def by(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def stats(vals):
    v = np.array(vals, dtype=float)
    return {"n": len(v), "mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else 0.0,
            "median": float(np.median(v)), "min": v.min(), "max": v.max(),
            "n_solved": int((v <= TARGET).sum())}


def e(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2e}"


# ----------------------------------------------------------------------------------- tables
def summary_table(rows):
    cells = by(rows, "arm", "beta")
    out = []
    for (arm, beta), rs in sorted(cells.items(), key=lambda kv: (ARMS.index(kv[0][0]), kv[0][1])):
        s = stats([r["rel_l2"] for r in rs])
        s.update(arm=arm, label=arm_label(arm), beta=beta,
                 mean_evals=np.mean([r["loss_evals"] for r in rs]),
                 mean_train_s=np.mean([r["train_s"] for r in rs]))
        out.append(s)
    return out


def largest_solved(summ, target=TARGET):
    res = {}
    for arm in ARMS:
        cells = sorted((s for s in summ if s["arm"] == arm), key=lambda s: s["beta"])
        if not cells:
            continue
        mean_ok = [s["beta"] for s in cells if s["mean"] <= target]
        med_ok = [s["beta"] for s in cells if s["median"] <= target]
        contiguous = None
        for s in cells:
            if s["mean"] <= target:
                contiguous = s["beta"]
            else:
                break
        res[arm] = {"largest_mean": max(mean_ok) if mean_ok else None,
                    "largest_median": max(med_ok) if med_ok else None,
                    "contiguous_mean": contiguous}
    return res


EFFECTS = ["C", "W", "H", "CxW", "CxH", "WxH", "CxWxH"]


def factorial_effects(rows, beta_min=20.0, beta_max=float("inf")):
    """2^3 factorial effects on log10(rel L2): effect = mean(y | sign=+) - mean(y | sign=-),
    computed per (beta, seed) cell with all 8 arms, averaged over beta per seed, then
    mean +/- standard error over seeds."""
    cells = by([r for r in rows if beta_min <= r["beta"] <= beta_max], "beta", "seed")
    per_seed = defaultdict(lambda: defaultdict(list))
    for (beta, seed), rs in cells.items():
        y = {r["arm"]: math.log10(r["rel_l2"]) for r in rs}
        if len(y) < 8:
            continue
        for eff in EFFECTS:
            facs = eff.split("x")
            total = 0.0
            for arm, val in y.items():
                on = {"C": int(arm[1]), "W": int(arm[3]), "H": int(arm[5])}
                sign = np.prod([1 if on[f] else -1 for f in facs])
                total += sign * val
            per_seed[seed][eff].append(total / 4.0)
    res = {}
    for eff in EFFECTS:
        vals = [np.mean(per_seed[s][eff]) for s in per_seed if per_seed[s][eff]]
        if vals:
            res[eff] = (float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                        if len(vals) > 1 else 0.0, len(vals))
    return res


# ---------------------------------------------------------------------------------- figures
def fig_error_vs_beta(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, facecolor=SURFACE)
    for ax, h, title in zip(axes, (0, 1), ("Soft IC/BC (loss terms)", "Hard IC/BC (built into network)")):
        style(ax)
        for (c, w), lab in CW_LABEL.items():
            arm = f"C{c}W{w}H{h}"
            cells = by([r for r in rows if r["arm"] == arm], "beta")
            if not cells:
                continue
            betas = sorted(b for (b,) in cells)
            means = [np.mean([r["rel_l2"] for r in cells[(b,)]]) for b in betas]
            for b in betas:
                v = [r["rel_l2"] for r in cells[(b,)]]
                ax.scatter([b] * len(v), v, s=9, color=CW_COLOR[(c, w)], alpha=0.35, linewidths=0)
            ax.plot(betas, means, color=CW_COLOR[(c, w)], lw=2, marker=CW_MARKER[(c, w)], ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=lab)
        ax.axhline(TARGET, color=MUTED, lw=0.8)
        ax.text(50, TARGET * 0.85, "1e-2 target", color=MUTED, fontsize=7, va="top", ha="right")
        ax.set_yscale("log")
        ax.set_xlabel("convection speed β")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("relative L2 error (mean; dots = seeds)")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "error_vs_beta.png", dpi=200)
    plt.close(fig)


def fig_heatmaps(rows, beta=30.0, seed=0):
    from problem import NT, XGRID, exact_u, grid
    _, _, X_star = grid()
    panels = [("exact  sin(x − βt)", exact_u(beta, X_star).reshape(NT, XGRID), None)]
    for arm in ARMS:
        r = [r for r in rows if r["arm"] == arm and r["beta"] == beta and r["seed"] == seed]
        path = RES / "preds" / f"{r[0]['run_id']}.npz" if r else None
        if path and path.exists():
            panels.append((arm_label(arm), np.load(path)["u_pred"], r[0]["rel_l2"]))
    if len(panels) < 2:
        return
    ncol = 3
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 2.3 * nrow), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, (title, u, err) in zip(axes, panels):
        im = ax.imshow(u.T, origin="lower", aspect="auto", extent=[0, 1, 0, 2 * np.pi],
                       cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
        sub = "" if err is None else f"\nrel. L2 = {err:.2e}"
        ax.set_title(title + sub, fontsize=8, color=INK, loc="left")
        ax.tick_params(labelsize=7, colors=INK2)
        ax.set_xlabel("t", fontsize=7, color=INK2)
        ax.set_ylabel("x", fontsize=7, color=INK2)
        for s in ax.spines.values():
            s.set_color(AXIS)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.6, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.set_label("u(x, t)  (colors clipped at ±1)", fontsize=8, color=INK2)
    fig.savefig(FIG / f"heatmaps_beta{int(beta)}_seed{seed}.png", dpi=200)
    plt.close(fig)


def fig_effects(effects):
    if not effects:
        return
    names = [k for k in EFFECTS if k in effects]
    vals = [effects[k][0] for k in names]
    errs = [effects[k][1] for k in names]
    fig, ax = plt.subplots(figsize=(5.5, 3.0), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, vals, xerr=errs, height=0.6, color="#2a78d6",
            error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([n.replace("x", " × ") for n in names])
    for y, v, s, n in zip(ypos, vals, errs, names):
        if "x" not in n:  # label main effects only, just past the error bar
            off = s + 0.03
            ax.text(v + (off if v >= 0 else -off), y, f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.5, color=INK)
    lo = min(v - s for v, s in zip(vals, errs))
    hi = max(v + s for v, s in zip(vals, errs))
    ax.set_xlim(lo - 0.15, hi + 0.15)
    ax.set_xlabel("effect on log10(rel. L2), β ≥ 20  (negative = lower error)")
    fig.tight_layout()
    fig.savefig(FIG / "factorial_effects.png", dpi=200)
    plt.close(fig)


def fig_reproduction(main_rows, repro_rows, authors_rows, fine_rows=()):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True, facecolor=SURFACE)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    markers = ["o", "s", "^", "D"]

    def mean_by_beta(rows, key="rel_l2"):
        g = defaultdict(list)
        for r in rows:
            g[float(r["beta"])].append(float(r[key]))
        bs = sorted(g)
        return bs, [np.mean(g[b]) for b in bs]

    left = [("paper (Fig. 3, best lr, ≥10 seeds)", sorted(PAPER_VANILLA.items())),
            ("authors' script, 5 seeds", list(zip(*mean_by_beta(authors_rows)))),
            ("our vanilla arm, 5 seeds", list(zip(*mean_by_beta([r for r in main_rows if r["arm"] == "C0W0H0"]))))]
    right = [("paper (Table 1)", sorted(PAPER_CURRIC.items())),
             ("ours, Δβ=5, stages to convergence", list(zip(*mean_by_beta(repro_rows)))),
             ("ours, Δβ=5, compute-matched (10k evals)",
              list(zip(*mean_by_beta([r for r in main_rows if r["arm"] == "C1W0H0" and r["beta"] >= 20])))),
             ("ours, Δβ=1, stages to convergence", list(zip(*mean_by_beta(fine_rows))))]
    for ax, series, title in zip(axes, (left, right), ("Vanilla PINN", "Curriculum warm-start")):
        style(ax)
        for (lab, pts), col, mk in zip(series, colors, markers):
            if not pts:
                continue
            b, v = zip(*pts)
            ax.plot(b, v, color=col, lw=2, marker=mk, ms=5, markeredgecolor=SURFACE,
                    markeredgewidth=1, label=lab)
        ax.axhline(TARGET, color=MUTED, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(5e-3, 6)
        ax.set_xlabel("convection speed β")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=6.5, labelcolor=INK2, loc="upper left")
    axes[0].set_ylabel("relative L2 error (mean)")
    fig.tight_layout()
    fig.savefig(FIG / "reproduction.png", dpi=200)
    plt.close(fig)


def fig_effects_vs_beta(rows, betas):
    """Main effects (and W x H) at each beta: where does each fix matter?"""
    series = [("C", "curriculum (C)", "#2a78d6", "o"), ("W", "causal weighting (W)", "#eb6834", "s"),
              ("H", "hard IC/BC (H)", "#1baf7a", "^"), ("WxH", "W × H interaction", "#eda100", "D")]
    est = {b: factorial_effects(rows, b, b) for b in betas}
    bs = [b for b in betas if est[b]]
    if not bs:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.3), facecolor=SURFACE)
    style(ax)
    ax.axhline(0, color=AXIS, lw=0.8)
    for key, lab, col, mk in series:
        m = np.array([est[b][key][0] for b in bs])
        s = np.array([est[b][key][1] for b in bs])
        ax.fill_between(bs, m - s, m + s, color=col, alpha=0.15, linewidth=0)
        ax.plot(bs, m, color=col, lw=2, marker=mk, ms=5, markeredgecolor=SURFACE,
                markeredgewidth=1, label=lab)
    ax.set_xlabel("convection speed β")
    ax.set_ylabel("effect on log10(rel. L2)\n(negative = lower error)")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "effects_vs_beta.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------------- report
def main():
    import sys
    sys.path.insert(0, str(REPO / "src"))
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_runs("main")
    repro = load_runs("repro_curriculum")
    fine = load_runs("repro_curriculum_fine")
    follow = load_runs("followup")
    authors = read_csv(RES / "authors_script.csv")
    spectral = read_csv(RES / "spectral_baseline.csv")

    summ = summary_table(rows)
    with open(RES / "summary_by_arm_beta.csv", "w", newline="") as f:
        cols = ["arm", "label", "beta", "n", "mean", "std", "median", "min", "max", "n_solved",
                "mean_evals", "mean_train_s"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(summ)

    L = ["# Results summary", "",
         f"Main sweep: {len(rows)} runs (tag `main`). Metric: relative L2 error vs sin(x − βt) on the "
         "fixed 256 × 100 grid. 'Solved' = rel. L2 ≤ 1e-2. Budget: 10,000 loss evaluations per run.", ""]

    betas = sorted({s["beta"] for s in summ})
    L += ["## Mean relative L2 error (n seeds per cell in parentheses when < 5)", "",
          "| arm | " + " | ".join(f"β={b:g}" for b in betas) + " |",
          "|---|" + "---|" * len(betas)]
    for arm in ARMS:
        cells = {s["beta"]: s for s in summ if s["arm"] == arm}
        if not cells:
            continue
        txt = []
        for b in betas:
            s = cells.get(b)
            txt.append("-" if s is None else e(s["mean"]) + (f" ({s['n']})" if s["n"] < 5 else ""))
        L.append(f"| {arm_label(arm)} | " + " | ".join(txt) + " |")

    L += ["", "## Seeds solved (≤ 1e-2) out of n", "",
          "| arm | " + " | ".join(f"β={b:g}" for b in betas) + " |", "|---|" + "---|" * len(betas)]
    for arm in ARMS:
        cells = {s["beta"]: s for s in summ if s["arm"] == arm}
        if cells:
            L.append(f"| {arm_label(arm)} | " + " | ".join(
                "-" if b not in cells else f"{cells[b]['n_solved']}/{cells[b]['n']}" for b in betas) + " |")

    ls = largest_solved(summ)
    ls5 = largest_solved(summ, 5e-2)
    L += ["", "## Largest β solved", "",
          "Primary threshold 1e-2 (project outline). Soft-constraint arms sit right at it even at β = 1 "
          "(vanilla mean 1.06e-2), so a secondary 5e-2 threshold is shown too.", "",
          "| arm | largest β, mean ≤ 1e-2 | largest β, median ≤ 1e-2 | largest β, mean ≤ 5e-2 | "
          "all β up to (mean ≤ 5e-2) |", "|---|---|---|---|---|"]
    f_ = lambda v: "none" if v is None else f"{v:g}"  # noqa: E731
    for arm, d in ls.items():
        d5 = ls5[arm]
        L.append(f"| {arm_label(arm)} | {f_(d['largest_mean'])} | {f_(d['largest_median'])} | "
                 f"{f_(d5['largest_mean'])} | {f_(d5['contiguous_mean'])} |")

    eff = factorial_effects(rows)
    L += ["", "## 2×2×2 factorial effects on log10(rel. L2), β ≥ 20", "",
          "Effect = mean over arms with the factor on − mean with it off (other factors averaged). "
          "Computed per (β, seed), averaged over β, mean ± s.e. over seeds. −1.0 = 10× lower error.", "",
          "| effect | estimate | s.e. | seeds |", "|---|---|---|---|"]
    names = {"C": "curriculum (C)", "W": "causal weighting (W)", "H": "hard IC/BC (H)"}
    for k in EFFECTS:
        if k in eff:
            m, se, n = eff[k]
            L.append(f"| {names.get(k, k.replace('x', ' × '))} | {m:+.2f} | {se:.2f} | {n} |")

    L += ["", "### Main effects at each β (mean over 5 seeds; s.e. in parentheses)", "",
          "| β | curriculum (C) | causal (W) | hard (H) | W × H |", "|---|---|---|---|---|"]
    for b in betas:
        eb = factorial_effects(rows, b, b)
        if eb:
            L.append(f"| {b:g} | " + " | ".join(f"{eb[k][0]:+.2f} ({eb[k][1]:.2f})" for k in
                                                ("C", "W", "H", "WxH")) + " |")

    cost = by(rows, "arm")
    L += ["", "## Cost per run (mean over all β and seeds)", "",
          "| arm | loss evals | L-BFGS iters | train time (s) | stopped early (tolerance) |",
          "|---|---|---|---|---|"]
    for arm in ARMS:
        rs = cost.get((arm,), [])
        if rs:
            early = sum(r["stop_reason"] != "budget" for r in rs)
            L.append(f"| {arm_label(arm)} | {np.mean([r['loss_evals'] for r in rs]):.0f} | "
                     f"{np.mean([r['lbfgs_iters'] for r in rs]):.0f} | "
                     f"{np.mean([r['train_s'] for r in rs]):.0f} | {early}/{len(rs)} |")

    L += ["", "## Reproduction", "",
          "Cells: mean rel. L2 (n seeds); [min over seeds] for the curriculum variants, since the "
          "paper reports the lowest error per seed over an lr sweep.", "",
          "| β | paper vanilla | authors' script | our vanilla | paper curriculum | "
          "ours Δβ=5, to convergence | ours Δβ=5, compute-matched | ours Δβ=1, to convergence |",
          "|---|---|---|---|---|---|---|---|"]
    ag = by([dict(r, beta=float(r["beta"]), rel_l2=float(r["rel_l2"])) for r in authors], "beta")
    vg = by([r for r in rows if r["arm"] == "C0W0H0"], "beta")
    rg = by(repro, "beta")
    cg = by([r for r in rows if r["arm"] == "C1W0H0"], "beta")
    fg = by(fine, "beta")

    def mn(g, b, show_min=False):
        rs = g.get((b,), [])
        if not rs:
            return "-"
        v = [r["rel_l2"] for r in rs]
        return f"{np.mean(v):.2e} ({len(v)})" + (f" [{min(v):.1e}]" if show_min else "")
    for b in sorted(set(PAPER_VANILLA) | set(PAPER_CURRIC)):
        L.append(f"| {b:g} | {e(PAPER_VANILLA.get(b))} | {mn(ag, b)} | {mn(vg, b)} | "
                 f"{e(PAPER_CURRIC.get(b))} | {mn(rg, b, True)} | {mn(cg, b, True)} | {mn(fg, b, True)} |")

    lr_rows = load_runs("repro_curriculum_lr")
    if lr_rows:
        import json as _json
        L += ["", "## Curriculum learning-rate check (β = 30, Δβ = 5, stages to convergence)", "",
              "| L-BFGS lr | mean | median | min | solved | n |", "|---|---|---|---|---|---|"]
        groups = defaultdict(list)
        for r in lr_rows:
            groups[_json.loads(r["config_json"])["lr"]].append(r["rel_l2"])
        groups[1.0] = [r["rel_l2"] for r in repro if r["beta"] == 30.0]
        for lr in sorted(groups):
            v = np.array(groups[lr])
            L.append(f"| {lr:g} | {v.mean():.2e} | {np.median(v):.2e} | {v.min():.2e} | "
                     f"{int((v <= TARGET).sum())}/{len(v)} | {len(v)} |")
        L.append(f"| paper (best lr per seed) | {e(PAPER_CURRIC[30.0])} | | | | |")

    if follow:
        L += ["", "## Post-hoc follow-up (tag `followup`): β²-normalised causal weights (W2) and "
              "exp time factor (H2)", "",
              "Each follow-up arm next to the main-sweep arm it modifies. Mean rel. L2 (n seeds) [solved].", "",
              "| follow-up arm | vs. main arm | β | follow-up | main |", "|---|---|---|---|---|"]
        fgrp = by(follow, "arm", "beta")
        mgrp = by(rows, "arm", "beta")
        for (arm, b) in sorted(fgrp, key=lambda k: (k[0], k[1])):
            ref = arm[:3] + ("1" if arm[3] == "2" else arm[3]) + arm[4:5] + ("1" if arm[5] == "2" else arm[5])
            fv = [r["rel_l2"] for r in fgrp[(arm, b)]]
            mv = [r["rel_l2"] for r in mgrp.get((ref, b), [])]
            fs = f"{np.mean(fv):.2e} ({len(fv)}) [{sum(v <= TARGET for v in fv)}]"
            ms = f"{np.mean(mv):.2e} ({len(mv)}) [{sum(v <= TARGET for v in mv)}]" if mv else "-"
            L.append(f"| {arm_label(arm)} | {arm_label(ref)} | {b:g} | {fs} | {ms} |")

    if spectral:
        L += ["", "## Classical baseline (pseudo-spectral, same grid)", "",
              "| β | spectral RK4 rel. L2 | time (ms) | exact Fourier rel. L2 | time (ms) |",
              "|---|---|---|---|---|"]
        sp = by([dict(r, beta=float(r["beta"])) for r in spectral], "beta", "method")
        for b in sorted({float(r["beta"]) for r in spectral}):
            rk, fx = sp[(b, "spectral_rk4")][0], sp[(b, "fourier_exact")][0]
            L.append(f"| {b:g} | {float(rk['rel_l2']):.1e} | {1000*float(rk['wall_s']):.1f} | "
                     f"{float(fx['rel_l2']):.1e} | {1000*float(fx['wall_s']):.2f} |")

    (RES / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    if rows:
        fig_error_vs_beta(rows)
        fig_heatmaps(rows)
        fig_effects(eff)
        fig_effects_vs_beta(rows, betas)
    fig_reproduction(rows, repro, authors, fine)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
