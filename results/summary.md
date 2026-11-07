# Results summary

Main sweep: 440 runs (tag `main`). Metric: relative L2 error vs sin(x − βt) on the fixed 256 × 100 grid. 'Solved' = rel. L2 ≤ 1e-2. Budget: 10,000 loss evaluations per run.

## Mean relative L2 error (n seeds per cell in parentheses when < 5)

| arm | β=1 | β=5 | β=10 | β=15 | β=20 | β=25 | β=30 | β=35 | β=40 | β=45 | β=50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | 1.06e-02 | 1.12e-02 | 1.40e-02 | 9.42e-01 | 1.37e+00 | 1.38e+00 | 1.39e+00 | 1.21e+00 | 1.26e+00 | 1.47e+00 | 1.54e+00 |
| causal | 1.03e-02 | 1.53e-02 | 2.29e-02 | 7.49e-01 | 1.37e+00 | 1.22e+00 | 1.21e+00 | 1.13e+00 | 1.17e+00 | 1.46e+00 | 1.38e+00 |
| curriculum | 1.06e-02 | 1.12e-02 | 1.46e-02 | 3.69e-02 | 1.65e-01 | 2.04e-01 | 4.79e-01 | 6.92e-01 | 6.92e-01 | 8.10e-01 | 8.73e-01 |
| curriculum + causal | 1.03e-02 | 1.53e-02 | 2.16e-02 | 6.94e-02 | 1.74e-01 | 3.14e-01 | 5.69e-01 | 7.01e-01 | 7.25e-01 | 7.90e-01 | 8.34e-01 |
| vanilla + hard | 2.11e-03 | 4.67e-03 | 7.22e-03 | 1.07e+00 | 1.37e+00 | 1.41e+00 | 1.17e+00 | 1.21e+00 | 1.07e+00 | 1.06e+00 | 1.06e+00 |
| causal + hard | 2.49e-03 | 4.46e-03 | 9.13e-03 | 2.04e-02 | 7.72e-01 | 4.70e+00 | 5.92e+00 | 9.37e+00 | 6.73e+00 | 1.18e+01 | 1.31e+01 |
| curriculum + hard | 2.11e-03 | 4.67e-03 | 4.96e-03 | 1.29e-02 | 2.54e-02 | 2.42e-01 | 7.43e-01 | 9.56e-01 | 9.94e-01 | 9.89e-01 | 9.99e-01 |
| curriculum + causal + hard | 2.49e-03 | 4.46e-03 | 8.70e-03 | 2.23e-02 | 3.86e-02 | 1.67e-01 | 6.04e-01 | 8.58e-01 | 9.78e-01 | 9.84e-01 | 9.84e-01 |

## Seeds solved (≤ 1e-2) out of n

| arm | β=1 | β=5 | β=10 | β=15 | β=20 | β=25 | β=30 | β=35 | β=40 | β=45 | β=50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | 2/5 | 1/5 | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| causal | 3/5 | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| curriculum | 2/5 | 1/5 | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| curriculum + causal | 3/5 | 1/5 | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| vanilla + hard | 5/5 | 5/5 | 4/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| causal + hard | 5/5 | 5/5 | 3/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| curriculum + hard | 5/5 | 5/5 | 5/5 | 2/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| curriculum + causal + hard | 5/5 | 5/5 | 4/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |

## Largest β solved

Primary threshold 1e-2 (project outline). Soft-constraint arms sit right at it even at β = 1 (vanilla mean 1.06e-2), so a secondary 5e-2 threshold is shown too.

| arm | largest β, mean ≤ 1e-2 | largest β, median ≤ 1e-2 | largest β, mean ≤ 5e-2 | all β up to (mean ≤ 5e-2) |
|---|---|---|---|---|
| vanilla | none | none | 10 | 10 |
| causal | none | 1 | 10 | 10 |
| curriculum | none | none | 15 | 15 |
| curriculum + causal | none | 1 | 10 | 10 |
| vanilla + hard | 10 | 10 | 10 | 10 |
| causal + hard | 10 | 10 | 15 | 15 |
| curriculum + hard | 10 | 10 | 20 | 20 |
| curriculum + causal + hard | 10 | 10 | 20 | 20 |

## 2×2×2 factorial effects on log10(rel. L2), β ≥ 20

Effect = mean over arms with the factor on − mean with it off (other factors averaged). Computed per (β, seed), averaged over β, mean ± s.e. over seeds. −1.0 = 10× lower error.

| effect | estimate | s.e. | seeds |
|---|---|---|---|
| curriculum (C) | -0.61 | 0.08 | 5 |
| causal weighting (W) | +0.12 | 0.04 | 5 |
| hard IC/BC (H) | +0.14 | 0.06 | 5 |
| C × W | -0.09 | 0.03 | 5 |
| C × H | -0.04 | 0.07 | 5 |
| W × H | +0.10 | 0.03 | 5 |
| C × W × H | -0.15 | 0.05 | 5 |

### Main effects at each β (mean over 5 seeds; s.e. in parentheses)

| β | curriculum (C) | causal (W) | hard (H) | W × H |
|---|---|---|---|---|
| 1 | +0.00 (0.00) | +0.01 (0.04) | -0.66 (0.05) | +0.03 (0.05) |
| 5 | -0.00 (0.00) | +0.05 (0.07) | -0.45 (0.06) | -0.06 (0.04) |
| 10 | -0.07 (0.08) | +0.18 (0.03) | -0.38 (0.06) | +0.01 (0.04) |
| 15 | -0.98 (0.09) | -0.26 (0.11) | -0.56 (0.15) | -0.37 (0.11) |
| 20 | -1.28 (0.14) | -0.10 (0.11) | -0.37 (0.11) | -0.13 (0.10) |
| 25 | -1.08 (0.14) | +0.10 (0.03) | +0.24 (0.11) | +0.01 (0.08) |
| 30 | -0.54 (0.14) | +0.11 (0.12) | +0.25 (0.15) | +0.04 (0.07) |
| 35 | -0.38 (0.04) | +0.18 (0.05) | +0.27 (0.03) | +0.19 (0.04) |
| 40 | -0.26 (0.09) | +0.12 (0.10) | +0.17 (0.10) | +0.10 (0.04) |
| 45 | -0.37 (0.07) | +0.22 (0.06) | +0.20 (0.07) | +0.23 (0.05) |
| 50 | -0.35 (0.07) | +0.20 (0.06) | +0.18 (0.07) | +0.23 (0.07) |

## Cost per run (mean over all β and seeds)

| arm | loss evals | L-BFGS iters | train time (s) | stopped early (tolerance) |
|---|---|---|---|---|
| vanilla | 2380 | 2202 | 39 | 55/55 |
| causal | 2107 | 1761 | 36 | 53/55 |
| curriculum | 5192 | 4751 | 89 | 52/55 |
| curriculum + causal | 4706 | 4308 | 124 | 55/55 |
| vanilla + hard | 1832 | 1724 | 26 | 55/55 |
| causal + hard | 5068 | 4624 | 80 | 48/55 |
| curriculum + hard | 7001 | 6676 | 102 | 26/55 |
| curriculum + causal + hard | 6795 | 6423 | 140 | 26/55 |

## Reproduction

Cells: mean rel. L2 (n seeds); [min over seeds] for the curriculum variants, since the paper reports the lowest error per seed over an lr sweep.

| β | paper vanilla | authors' script | our vanilla | paper curriculum | ours Δβ=5, to convergence | ours Δβ=5, compute-matched | ours Δβ=1, to convergence |
|---|---|---|---|---|---|---|---|
| 1 | 7.84e-03 | 7.58e-03 (5) | 1.06e-02 (5) | - | - | 1.06e-02 (5) [7.9e-03] | - |
| 10 | 1.08e-02 | 2.65e-02 (5) | 1.40e-02 (5) | - | - | 1.46e-02 (5) [8.5e-03] | - |
| 20 | 7.50e-01 | 1.38e+00 (5) | 1.37e+00 (5) | 9.84e-03 | 1.65e-01 (5) [3.0e-02] | 1.65e-01 (5) [3.0e-02] | 4.42e-02 (5) [2.9e-02] |
| 30 | 8.97e-01 | 1.25e+00 (5) | 1.39e+00 (5) | 2.02e-02 | 4.26e-01 (5) [4.0e-02] | 4.79e-01 (5) [3.5e-02] | 4.87e-01 (5) [7.4e-02] |
| 40 | 9.61e-01 | 1.37e+00 (5) | 1.26e+00 (5) | 5.33e-02 | 6.37e-01 (5) [6.4e-02] | 6.92e-01 (5) [2.1e-01] | 7.06e-01 (5) [5.8e-01] |

## Curriculum learning-rate check (β = 30, Δβ = 5, stages to convergence)

| L-BFGS lr | mean | median | min | solved | n |
|---|---|---|---|---|---|
| 0.1 | 5.95e-01 | 6.70e-01 | 7.10e-02 | 0/5 | 5 |
| 0.5 | 4.19e-01 | 6.12e-01 | 2.33e-02 | 0/5 | 5 |
| 1 | 4.26e-01 | 5.91e-01 | 3.99e-02 | 0/5 | 5 |
| 2 | 6.15e-01 | 7.97e-01 | 3.66e-02 | 0/5 | 5 |
| paper (best lr per seed) | 2.02e-02 | | | | |

## Post-hoc follow-up (tag `followup`): β²-normalised causal weights (W2) and exp time factor (H2)

Each follow-up arm next to the main-sweep arm it modifies. Mean rel. L2 (n seeds) [solved].

| follow-up arm | vs. main arm | β | follow-up | main |
|---|---|---|---|---|
| vanilla + hard(exp) | vanilla + hard | 20 | 1.55e+00 (5) [0] | 1.37e+00 (5) [0] |
| vanilla + hard(exp) | vanilla + hard | 30 | 1.62e+00 (5) [0] | 1.17e+00 (5) [0] |
| vanilla + hard(exp) | vanilla + hard | 40 | 1.70e+00 (5) [0] | 1.07e+00 (5) [0] |
| causal/β² | causal | 20 | 1.31e+00 (5) [0] | 1.37e+00 (5) [0] |
| causal/β² | causal | 30 | 1.04e+00 (5) [0] | 1.21e+00 (5) [0] |
| causal/β² | causal | 40 | 9.94e-01 (5) [0] | 1.17e+00 (5) [0] |
| causal/β² + hard | causal + hard | 20 | 1.22e+00 (5) [0] | 7.72e-01 (5) [0] |
| causal/β² + hard | causal + hard | 30 | 1.07e+00 (5) [0] | 5.92e+00 (5) [0] |
| causal/β² + hard | causal + hard | 40 | 1.15e+00 (5) [0] | 6.73e+00 (5) [0] |
| curriculum + hard(exp) | curriculum + hard | 20 | 1.71e-02 (5) [2] | 2.54e-02 (5) [0] |
| curriculum + hard(exp) | curriculum + hard | 30 | 7.96e-01 (5) [0] | 7.43e-01 (5) [0] |
| curriculum + hard(exp) | curriculum + hard | 40 | 1.40e+00 (5) [0] | 9.94e-01 (5) [0] |
| curriculum + causal/β² | curriculum + causal | 20 | 3.33e-01 (5) [0] | 1.74e-01 (5) [0] |
| curriculum + causal/β² | curriculum + causal | 30 | 7.55e-01 (5) [0] | 5.69e-01 (5) [0] |
| curriculum + causal/β² | curriculum + causal | 40 | 8.31e-01 (5) [0] | 7.25e-01 (5) [0] |
| curriculum + causal/β² + hard | curriculum + causal + hard | 20 | 3.82e-02 (5) [0] | 3.86e-02 (5) [0] |
| curriculum + causal/β² + hard | curriculum + causal + hard | 30 | 6.29e-01 (5) [0] | 6.04e-01 (5) [0] |
| curriculum + causal/β² + hard | curriculum + causal + hard | 40 | 1.00e+00 (5) [0] | 9.78e-01 (5) [0] |

## Classical baseline (pseudo-spectral, same grid)

| β | spectral RK4 rel. L2 | time (ms) | exact Fourier rel. L2 | time (ms) |
|---|---|---|---|---|
| 1 | 5.0e-11 | 5.0 | 3.3e-16 | 0.79 |
| 5 | 1.9e-09 | 15.0 | 3.3e-16 | 0.75 |
| 10 | 3.9e-09 | 29.9 | 3.6e-16 | 0.75 |
| 15 | 9.3e-09 | 40.6 | 4.4e-16 | 0.75 |
| 20 | 1.1e-08 | 55.3 | 5.4e-16 | 0.76 |
| 25 | 1.7e-08 | 64.8 | 6.9e-16 | 0.77 |
| 30 | 1.9e-08 | 80.1 | 7.7e-16 | 0.76 |
| 35 | 2.0e-08 | 95.0 | 8.5e-16 | 0.78 |
| 40 | 2.6e-08 | 104.8 | 1.1e-15 | 0.78 |
| 45 | 2.8e-08 | 119.3 | 1.2e-15 | 0.78 |
| 50 | 3.4e-08 | 129.7 | 1.3e-15 | 0.79 |
