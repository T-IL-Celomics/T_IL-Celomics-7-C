#!/usr/bin/env python3
"""11  10-model curve fitting (AIC selection)

Fits up to 10 parametric models to a time-series feature per track and
selects the best by AIC.

Models: linear, quadratic, cubic, exponential_growth, exponential_decay,
        logistic, gompertz, power, log, plateau.

CLI
---
    python 11_curve_fitting_10_models.py \
        --input data.csv \
        --exp-col Experiment --parent-col Parent --time-col Time \
        --feature Area --outdir ./out
"""
import argparse, os, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")


# ── Model definitions ───────────────────────────────────────────
def _linear(t, a, b):          return a * t + b
def _quadratic(t, a, b, c):    return a * t**2 + b * t + c
def _cubic(t, a, b, c, d):     return a * t**3 + b * t**2 + c * t + d
def _exp_growth(t, a, b, c):   return a * np.exp(b * t) + c
def _exp_decay(t, a, b, c):    return a * np.exp(-b * t) + c
def _logistic(t, L, k, t0, b): return L / (1 + np.exp(-k * (t - t0))) + b
def _gompertz(t, a, b, c, d):  return a * np.exp(-b * np.exp(-c * t)) + d
def _power(t, a, b, c):        return a * (t + 0.01)**b + c
def _log_model(t, a, b):       return a * np.log(t + 1) + b
def _plateau(t, a, b, c):      return a * (1 - np.exp(-b * t)) + c

MODELS = {
    "linear":           (_linear,     2),
    "quadratic":        (_quadratic,  3),
    "cubic":            (_cubic,      4),
    "exp_growth":       (_exp_growth, 3),
    "exp_decay":        (_exp_decay,  3),
    "logistic":         (_logistic,   4),
    "gompertz":         (_gompertz,   4),
    "power":            (_power,      3),
    "log":              (_log_model,  2),
    "plateau":          (_plateau,    3),
}


def _aic(n, rss, k):
    if rss <= 0 or n <= k:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def fit_track(t, y):
    n = len(t)
    if n < 4:
        return {"best_model": "too_short", "best_aic": np.nan}
    results = {}
    for name, (func, n_params) in MODELS.items():
        if n <= n_params:
            continue
        try:
            popt, _ = curve_fit(func, t, y, maxfev=5000)
            yhat = func(t, *popt)
            rss = np.sum((y - yhat)**2)
            aic = _aic(n, rss, n_params)
            results[name] = aic
        except Exception:
            pass
    if not results:
        return {"best_model": "no_fit", "best_aic": np.nan}
    best = min(results, key=results.get)
    return {"best_model": best, "best_aic": results[best]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--exp-col", default="Experiment")
    ap.add_argument("--parent-col", default="Parent")
    ap.add_argument("--time-col", default="Time")
    ap.add_argument("--feature", required=True)
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")

    records = []
    for (e, p), grp in df.groupby([a.exp_col, a.parent_col]):
        grp = grp.sort_values(a.time_col)
        t = grp[a.time_col].values.astype(float)
        y = grp[a.feature].values.astype(float)
        mask = ~(np.isnan(t) | np.isnan(y))
        t, y = t[mask], y[mask]
        res = fit_track(t, y)
        res["Experiment"] = e
        res["Parent"] = p
        res["n_frames"] = len(t)
        records.append(res)

    result = pd.DataFrame(records)
    os.makedirs(a.outdir, exist_ok=True)
    out_csv = os.path.join(a.outdir, "curve_fitting_results.csv")
    result.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    print(f"\nBest model frequencies:\n{result['best_model'].value_counts()}")

    # bar chart
    counts = result["best_model"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(counts)), counts.values, tick_label=counts.index, edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title(f"Best-fit model distribution ({a.feature})")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out_png = os.path.join(a.outdir, "curve_fit_model_dist.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
