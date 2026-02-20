import numpy as np
import pandas as pd

# ====== model functions ======
def linear(x, a, b): return a * x + b
def exponential(x, a, b): return a * np.exp(b * x)
def logarithmic(x, a, b): return a * np.log(b * x + 1)
def logistic(x, a, b, c): return c / (1 + a * np.exp(b * -x))
def power(x, a, b): return a * np.power(x, b)
def inverse(x, a, b): return a / (x + b)
def quadratic(x, a, b, c): return a * x**2 + b * x + c
def sigmoid(x, a, b, c, d): return d + (a - d) / (1.0 + (x / c)**b)
def gompertz(x, a, b, c): return a * np.exp(-b * np.exp(-c * x))
def weibull(x, a, b, c): return a - b * np.exp(-c * x**2)
def poly3(x, a, b, c, d): return a * x**3 + b * x**2 + c * x + d
def poly4(x, a, b, c, d, e): return a * x**4 + b * x**3 + c * x**2 + d * x + e

MODEL_FUNCS = {
    "linear": linear,
    "exponential": exponential,
    "logarithmic": logarithmic,
    "logistic": logistic,
    "power": power,
    "inverse": inverse,
    "quadratic": quadratic,
    "sigmoid": sigmoid,
    "gompertz": gompertz,
    "weibull": weibull,
    "poly3": poly3,
    "poly4": poly4,
}

MODEL_PARAM_COUNTS = {
    "linear": 2,
    "exponential": 2,
    "logarithmic": 2,
    "logistic": 3,
    "power": 2,
    "inverse": 2,
    "quadratic": 3,
    "sigmoid": 4,
    "gompertz": 3,
    "weibull": 3,
    "poly3": 4,
    "poly4": 5,
}

def safe_predict(model_name: str, params: np.ndarray, x: np.ndarray) -> np.ndarray:
    func = MODEL_FUNCS.get(model_name)
    if func is None:
        raise ValueError(f"unknown model: {model_name}")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        y_pred = func(x, *params)
    return np.asarray(y_pred, dtype=float)

def fill_best_then_interp_inside_no_extrap(
    summary_path="cell_data/summary_table.xlsx",
    best_fit_csv="regression/fitting_best_with_nrmse.csv",
    out_csv_filled="cell_data/summary_table_filled_no_extrap.csv",
    out_csv_final="cell_data/summary_table_filled_no_extrap_FINAL_NO_NAN.csv",
    out_log_csv="cell_data/imputation_log_no_extrap.csv",
    time_col="TimeIndex",
    group_cols=("Experiment", "Parent"),
    # NOTE: we are NOT tracking ID at all
    exclude_cols=("dt",),
    # quality knobs:
    max_nan_frac_per_feature=0.02,
    min_cells_per_feature=200,
    drop_cells_with_any_nan=True,
    # dt handling:
    DEFAULT_DT=60,                # set your real dt (minutes) here
    DROP_DUP_TIMEINDEX=True,      # avoid duplicate index problems
):
    # load
    df = pd.read_excel(summary_path)
    best = pd.read_csv(best_fit_csv)

    # drop ID completely (avoids dtype issues + GEN_ hacks)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # normalize keys
    for c in group_cols:
        df[c] = df[c].astype(str)
        best[c] = best[c].astype(str)

    # numeric time index
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[time_col]).copy()
    df[time_col] = df[time_col].astype(int)

    # feature columns
    excluded = set(group_cols) | {time_col} | set(exclude_cols)
    cand_cols = [c for c in df.columns if c not in excluded and not str(c).startswith("Unnamed")]

    # convert candidates to numeric (features)
    for c in cand_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    features = [c for c in cand_cols if pd.api.types.is_numeric_dtype(df[c])]
    features = [c for c in features if not (c.endswith("__fitted") or c.endswith("__resid"))]
    print("[INFO] num features:", len(features))

    # best model lookup
    best_lookup = {}
    for _, r in best.iterrows():
        feat = r.get("Feature", None)
        model = r.get("Model", None)
        if feat is None or model is None or model not in MODEL_PARAM_COUNTS:
            continue
        k = MODEL_PARAM_COUNTS[model]
        params = []
        ok = True
        for i in range(k):
            v = r.get(f"param_{i}", np.nan)
            if not np.isfinite(v):
                ok = False
                break
            params.append(float(v))
        if not ok:
            continue
        key = (str(r[group_cols[0]]), str(r[group_cols[1]]), str(feat))
        best_lookup[key] = (model, np.array(params, dtype=float))

    logs = []
    out_groups = []

    for (exp, parent), g in df.groupby(list(group_cols), sort=False):
        g = g.sort_values(time_col).copy()

        if DROP_DUP_TIMEINDEX:
            g = g.drop_duplicates(subset=[time_col], keep="first").copy()

        # cell window: where ANY feature exists
        obs_mask = g[features].notna().any(axis=1)
        observed_times = g.loc[obs_mask, time_col].astype(int).values
        if len(observed_times) == 0:
            out_groups.append(g)
            continue

        t_min = int(observed_times.min())
        t_max = int(observed_times.max())

        # reindex ONLY inside [t_min..t_max]
        full_index = pd.RangeIndex(t_min, t_max + 1)
        g = (
            g.set_index(time_col)
             .reindex(full_index)
             .rename_axis(time_col)
             .reset_index()
        )

        # restore keys
        g[group_cols[0]] = exp
        g[group_cols[1]] = parent

        # ===== dt fix =====
        if "dt" in g.columns:
            g["dt"] = pd.to_numeric(g["dt"], errors="coerce")
            g["dt"] = g["dt"].ffill().bfill()
            if g["dt"].isna().all():
                g["dt"] = float(DEFAULT_DT)
            else:
                g["dt"] = g["dt"].fillna(float(DEFAULT_DT))
        else:
            g["dt"] = float(DEFAULT_DT)

        t_vals = g[time_col].astype(int).values
        x_all = t_vals.astype(float)

        for feat in features:
            y = pd.to_numeric(g[feat], errors="coerce").astype(float).values

            feat_obs = np.isfinite(y)
            if feat_obs.sum() < 2:
                logs.append({
                    "Experiment": exp, "Parent": parent, "Feature": feat,
                    "method": "skipped:<2_observations",
                    "n_model": 0, "n_interp": 0, "error": None,
                })
                continue

            f_min = int(t_vals[feat_obs].min())
            f_max = int(t_vals[feat_obs].max())

            miss = ~np.isfinite(y)
            allowed = (t_vals >= f_min) & (t_vals <= f_max)
            to_fill = miss & allowed
            if not to_fill.any():
                continue

            n_model = 0
            n_interp = 0
            err = None
            key = (str(exp), str(parent), str(feat))

            model_name = None
            if key in best_lookup:
                model_name, params = best_lookup[key]
                try:
                    y_pred = safe_predict(model_name, params, x_all)
                    fill_mask = to_fill & np.isfinite(y_pred)
                    if fill_mask.any():
                        y[fill_mask] = y_pred[fill_mask]
                        n_model = int(fill_mask.sum())
                except Exception as e:
                    err = str(e)[:200]

            # interpolate inside only
            s = pd.Series(y, index=t_vals).sort_index()
            s2 = s.interpolate(method="linear", limit_area="inside")
            y2 = s2.values

            still = ~np.isfinite(y)
            interp_mask = still & to_fill & np.isfinite(y2)
            if interp_mask.any():
                y[interp_mask] = y2[interp_mask]
                n_interp = int(interp_mask.sum())

            g[feat] = y
            logs.append({
                "Experiment": exp, "Parent": parent, "Feature": feat,
                "method": f"model+interp_inside:{model_name}" if model_name else "interp_inside:no_model",
                "n_model": n_model, "n_interp": n_interp, "error": err,
            })

        out_groups.append(g)

    out_df = pd.concat(out_groups, ignore_index=True)

    # sanity: dt should be non-null now
    if "dt" in out_df.columns:
        print("[CHECK] dt NaNs after fill:", int(out_df["dt"].isna().sum()))

    out_df.to_csv(out_csv_filled, index=False)
    pd.DataFrame(logs).to_csv(out_log_csv, index=False)
    print(f"[SAVE] filled (may still have NaNs) -> {out_csv_filled}")
    print(f"[SAVE] log -> {out_log_csv}")

    # ====== QUALITY FILTERING TO GET ZERO NaNs WITHOUT EXTRAPOLATION ======
    meta = set(group_cols) | {time_col} | set(exclude_cols)
    meta = {c for c in meta if c in out_df.columns}
    feat_cols = [c for c in out_df.columns if c not in meta and not str(c).startswith("Unnamed")]

    nan_frac = out_df[feat_cols].isna().mean()
    keep_feats = nan_frac[nan_frac <= max_nan_frac_per_feature].index.tolist()

    cell_has_feat = (
        out_df.groupby(list(group_cols))[keep_feats]
              .apply(lambda x: x.notna().any(axis=0))
    )
    cells_per_feat = cell_has_feat.sum(axis=0)
    keep_feats2 = cells_per_feat[cells_per_feat >= min_cells_per_feature].index.tolist()

    dropped = sorted(set(feat_cols) - set(keep_feats2))
    print(f"[INFO] dropping {len(dropped)} sparse/unreliable features")

    final_cols = list(meta) + keep_feats2
    final_df = out_df[final_cols].copy()

    if drop_cells_with_any_nan:
        bad_cells = (
            final_df.groupby(list(group_cols))[keep_feats2]
                    .apply(lambda x: x.isna().any().any())
        )
        bad_ids = bad_cells[bad_cells].index
        before = len(final_df)
        if len(bad_ids) > 0:
            mask = ~final_df.set_index(list(group_cols)).index.isin(bad_ids)
            final_df = final_df.loc[mask].copy()
        after = len(final_df)
        print(f"[INFO] dropped {before-after} rows from cells that still had NaNs")

    remaining_nan = int(final_df[keep_feats2].isna().any(axis=1).sum()) if keep_feats2 else 0
    print("[INFO] remaining rows with any NaN in kept features:", remaining_nan)

    # ensure dt is not NaN in final
    if "dt" in final_df.columns:
        final_df["dt"] = (
            pd.to_numeric(final_df["dt"], errors="coerce")
              .ffill().bfill()
              .fillna(float(DEFAULT_DT))
        )

    final_df.to_csv(out_csv_final, index=False)
    print(f"[SAVE] final NO-NAN table -> {out_csv_final} with shape {final_df.shape}")

if __name__ == "__main__":
    fill_best_then_interp_inside_no_extrap(
        summary_path="cell_data/summary_table.xlsx",
        best_fit_csv="regression/fitting_best_with_nrmse.csv",
        out_csv_filled="cell_data/summary_table_filled_no_extrap.csv",
        out_csv_final="cell_data/summary_table_filled_no_extrap_FINAL_NO_NAN.csv",
        out_log_csv="cell_data/imputation_log_no_extrap.csv",
        time_col="TimeIndex",
        group_cols=("Experiment", "Parent"),
        max_nan_frac_per_feature=0.02,
        min_cells_per_feature=200,
        drop_cells_with_any_nan=True,
        DEFAULT_DT=60,  # change if needed
    )
