"""
Two-Way ANOVA: Dynamic Clustering vs Static (Mean-Based) Clustering

Runs TWO separate comparisons for every measurement feature:

  1. Embedding clustering  vs  Static clustering
  2. Embedding+Fitting clustering  vs  Static clustering

Each comparison uses a two-way ANOVA (Method × Cluster) and reports:
  - P-value for the Method main effect
  - P-value for the Cluster main effect
  - P-value for the Method × Cluster interaction
  - FDR-adjusted (Benjamini-Hochberg) p-values for each

Inputs:
  Embedding       : clustering/Merged_Clusters_PCA_k{k}.csv
  Embedding+Fit   : fitting/embedding_fitting_Merged_Clusters_PCA_k{k}.csv
  Static          : cell_data/static_clustering_raw_data.csv  (or env override)

Outputs:
  results/two_way_anova_emb_vs_static_k{k}.xlsx / .csv
  results/two_way_anova_embfit_vs_static_k{k}.xlsx / .csv
"""

import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore", category=FutureWarning)

# Columns that are NOT measurement features
EXCLUDE_COLS = {
    'Unnamed: 0', 'Experiment', 'Parent', 'Cluster', 'PC1', 'PC2',
    'Treatment', 'Treatments', 'Groups', 'dt', 'TimeIndex', 'ID',
    'x_Pos', 'y_Pos',
}


def _get_features(df: pd.DataFrame) -> list[str]:
    """Return ordered list of numeric feature columns."""
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def _plot_zscore_heatmaps(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    features: list[str],
    label_a: str,
    label_b: str,
    output_png: str,
    k_val: int | None = None,
):
    """Generate separate Z-scored feature-mean heatmaps (one per method).

    For each method, compute the mean of each feature per cluster, then
    Z-score across clusters (subtract feature mean, divide by feature std).
    This allows comparison of relative trends regardless of original scale.
    """
    # Compute cluster means
    means_a = df_a.groupby('Cluster')[features].mean()
    means_b = df_b.groupby('Cluster')[features].mean()

    # Z-score: (value - feature_mean) / feature_std  across clusters
    def _zscore(means_df):
        mu = means_df.mean(axis=0)
        sigma = means_df.std(axis=0).replace(0, 1)  # avoid div-by-zero
        return (means_df - mu) / sigma

    z_a = _zscore(means_a)
    z_b = _zscore(means_b)

    # Rename cluster index for clarity
    z_a.index = [f'Cluster {c}' for c in z_a.index]
    z_b.index = [f'Cluster {c}' for c in z_b.index]

    # Shared symmetric color range
    vmin = min(z_a.min().min(), z_b.min().min())
    vmax = max(z_a.max().max(), z_b.max().max())
    vlim = max(abs(vmin), abs(vmax))

    k_str = f' (k={k_val})' if k_val else ''
    base, ext = os.path.splitext(output_png)
    os.makedirs(os.path.dirname(output_png) or '.', exist_ok=True)

    for z_df, label, suffix in [(z_a, label_a, '_A'), (z_b, label_b, '_B')]:
        n_feat = len(features)
        fig_w = max(14, n_feat * 0.35 + 3)
        n_clusters = len(z_df)
        fig_h = max(5, n_clusters * 1.2 + 3)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            z_df, ax=ax, cmap='RdBu_r', center=0,
            vmin=-vlim, vmax=vlim,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='white',
            cbar=True,
            cbar_kws={'label': 'Z-score', 'shrink': 0.8},
        )
        safe_label = label.replace('+', 'plus').replace(' ', '_')
        ax.set_title(f'{label} – Z-Scored Feature Means{k_str}',
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Feature', fontsize=10)
        ax.set_ylabel('Cluster', fontsize=10)
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=9)

        fig.tight_layout()
        out_path = f"{base}_{safe_label}{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Saved heatmap: '{out_path}'")


def _run_tukey_and_plot(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    features: list[str],
    label_a: str,
    label_b: str,
    output_png: str,
    k_val: int | None = None,
):
    """Run Tukey HSD per feature within each method and plot FDR-adjusted p-value heatmaps.

    For each method separately, Tukey's HSD compares every pair of clusters.
    P-values are then FDR-corrected (Benjamini-Hochberg) across all
    feature × pair tests.  The result is shown as a side-by-side heatmap:
    rows = cluster pairs, columns = features, cell = −log10(FDR p-value).
    """

    def _tukey_for_method(df, method_label):
        """Return DataFrame of Tukey p-values: rows=pairs, cols=features."""
        clusters = sorted(df['Cluster'].unique())
        if len(clusters) < 2:
            print(f"  ⚠️  {method_label}: fewer than 2 clusters, skipping Tukey")
            return None, None

        pair_labels = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pair_labels.append(f"{clusters[i]} vs {clusters[j]}")

        raw_pvals = {}   # feature -> list of p-values (one per pair)
        for feat in features:
            sub = df[['Cluster', feat]].dropna()
            if sub['Cluster'].nunique() < 2 or len(sub) < 6:
                raw_pvals[feat] = [np.nan] * len(pair_labels)
                continue
            try:
                tukey = pairwise_tukeyhsd(sub[feat], sub['Cluster'], alpha=0.05)
                # Build a lookup from the summary table rows (group1, group2 -> p-adj)
                summary = tukey.summary()
                pair_pval_map = {}
                for row in summary.data[1:]:  # skip header row
                    g1, g2, _, p_adj = row[0], row[1], row[2], row[3]
                    pair_key = (str(g1), str(g2))
                    pair_pval_map[pair_key] = float(p_adj)
                # Map our pair_labels to the extracted p-values
                feat_pvals = []
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        key = (str(clusters[i]), str(clusters[j]))
                        key_rev = (str(clusters[j]), str(clusters[i]))
                        p = pair_pval_map.get(key, pair_pval_map.get(key_rev, np.nan))
                        feat_pvals.append(p)
                raw_pvals[feat] = feat_pvals
            except Exception:
                raw_pvals[feat] = [np.nan] * len(pair_labels)

        pval_df = pd.DataFrame(raw_pvals, index=pair_labels)  # rows=pairs, cols=features

        # FDR correction across ALL tests (flatten, correct, reshape)
        flat = pval_df.values.flatten()
        mask = ~np.isnan(flat)
        if mask.sum() == 0:
            return pval_df, pval_df  # nothing to correct

        fdr_flat = np.full_like(flat, np.nan)
        _, fdr_vals, _, _ = multipletests(flat[mask], method='fdr_bh')
        fdr_flat[mask] = fdr_vals
        fdr_df = pd.DataFrame(
            fdr_flat.reshape(pval_df.shape),
            index=pval_df.index, columns=pval_df.columns,
        )
        return pval_df, fdr_df

    raw_a, fdr_a = _tukey_for_method(df_a, label_a)
    raw_b, fdr_b = _tukey_for_method(df_b, label_b)

    # Need at least one valid result
    if fdr_a is None and fdr_b is None:
        print("  ⚠️  Tukey HSD: no valid results for either method")
        return

    k_str = f' (k={k_val})' if k_val else ''
    base, ext = os.path.splitext(output_png)
    os.makedirs(os.path.dirname(output_png) or '.', exist_ok=True)

    for fdr_df, label, df_src in [
        (fdr_a, label_a, df_a), (fdr_b, label_b, df_b)
    ]:
        if fdr_df is None:
            continue

        # --- Aggregate per-cluster: min FDR p-value across all pairs
        #     involving that cluster, for each feature ---
        clusters = sorted(df_src['Cluster'].unique())
        cluster_pval = pd.DataFrame(index=features, columns=clusters, dtype=float)
        for feat in features:
            for cid in clusters:
                # Collect p-values from all pairs that include this cluster
                pvals_for_cluster = []
                for pair_label in fdr_df.index:
                    parts = pair_label.split(' vs ')
                    if str(cid) in parts:
                        v = fdr_df.loc[pair_label, feat]
                        if not np.isnan(v):
                            pvals_for_cluster.append(v)
                cluster_pval.loc[feat, cid] = (
                    min(pvals_for_cluster) if pvals_for_cluster else np.nan
                )

        cluster_pval = cluster_pval.astype(float)

        # --- Plot: features on Y-axis (standing up), clusters on X-axis ---
        n_feat = len(features)
        fig_h = max(8, n_feat * 0.22 + 2)
        fig_w = max(6, len(clusters) * 1.8 + 4)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        sns.heatmap(
            cluster_pval, ax=ax, cmap='RdBu_r', vmin=0, vmax=1,
            annot=cluster_pval.map(
                lambda v: f'{v:.2f}' if not np.isnan(v) else ''
            ),
            fmt='', annot_kws={'size': 7},
            linewidths=0.5, linecolor='white',
            cbar=True,
            cbar_kws={'label': 'Adjusted p-value', 'shrink': 0.6},
        )
        ax.set_title(
            f'{label} – Tukey HSD Adjusted p-values by Feature & Cluster{k_str}',
            fontsize=13, fontweight='bold', pad=10,
        )
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Feature', fontsize=11)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

        fig.tight_layout()
        safe_label = label.replace('+', 'plus').replace(' ', '_')
        out_path = f"{base}_{safe_label}{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Saved Tukey heatmap: '{out_path}'")


def run_two_way_anova(
    dynamic_csv: str,
    static_csv: str,
    output_xlsx: str,
    output_csv: str,
    k_val: int | None = None,
    dynamic_label: str = "Dynamic",
):
    """Run two-way ANOVA (Method × Cluster) for each feature and save results.

    Parameters
    ----------
    dynamic_csv    : path to the dynamic clustering CSV
    static_csv     : path to the static (mean-based) clustering CSV
    output_xlsx    : path for the Excel output
    output_csv     : path for the CSV output
    k_val          : cluster count (informational)
    dynamic_label  : label for the dynamic method (e.g. "Embedding" or "Emb+Fitting")
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"  Loading {dynamic_label} data: {dynamic_csv}")
    df_dyn = pd.read_csv(dynamic_csv, low_memory=False)
    print(f"    Rows: {len(df_dyn):,}  |  Clusters: {sorted(df_dyn['Cluster'].unique())}")

    print(f"  Loading static (mean-based) data:       {static_csv}")
    df_sta = pd.read_csv(static_csv, low_memory=False)
    print(f"    Rows: {len(df_sta):,}  |  Clusters: {sorted(df_sta['Cluster'].unique())}")

    # If the static data has more clusters than required k, filter or warn
    static_k = df_sta['Cluster'].nunique()
    dyn_k = df_dyn['Cluster'].nunique()
    if k_val is not None and static_k != k_val:
        print(f"  ⚠️  Static file has k={static_k} clusters, expected k={k_val}")
        if static_k < k_val:
            print(f"     Proceeding anyway (fewer static clusters)")
        # If static has MORE clusters, still proceed — the ANOVA handles it

    # ------------------------------------------------------------------
    # 2. Identify common numeric features
    # ------------------------------------------------------------------
    dyn_feats = set(_get_features(df_dyn))
    sta_feats = set(_get_features(df_sta))
    common = sorted(dyn_feats & sta_feats)
    dyn_only = dyn_feats - sta_feats
    sta_only = sta_feats - dyn_feats
    if dyn_only:
        print(f"  Features only in dynamic ({len(dyn_only)}): {sorted(dyn_only)[:5]}{'...' if len(dyn_only) > 5 else ''}")
    if sta_only:
        print(f"  Features only in static  ({len(sta_only)}): {sorted(sta_only)[:5]}{'...' if len(sta_only) > 5 else ''}")
    print(f"  Common features: {len(common)}")

    if not common:
        print("  ❌ No common features between dynamic and static files.")
        return

    features = common

    # ------------------------------------------------------------------
    # 3. Average per unique cell (Experiment × Parent) within each method
    # ------------------------------------------------------------------
    id_cols = ['Experiment', 'Parent', 'Cluster']

    # Ensure id columns exist in both
    for col in id_cols:
        if col not in df_dyn.columns:
            print(f"  ❌ Dynamic file missing required column: {col}")
            return
        if col not in df_sta.columns:
            print(f"  ❌ Static file missing required column: {col}")
            return

    df_dyn_u = df_dyn.groupby(id_cols)[features].mean().reset_index()
    df_dyn_u['Method'] = dynamic_label
    print(f"  {dynamic_label} unique cells: {len(df_dyn_u):,}")

    df_sta_u = df_sta.groupby(id_cols)[features].mean().reset_index()
    df_sta_u['Method'] = 'Static'
    print(f"  Static unique cells:  {len(df_sta_u):,}")

    # ------------------------------------------------------------------
    # 4. Stack and run two-way ANOVA per feature
    # ------------------------------------------------------------------
    combined = pd.concat([df_dyn_u, df_sta_u], ignore_index=True)
    combined['Cluster'] = combined['Cluster'].astype(str)
    combined['Method'] = combined['Method'].astype(str)

    records = []
    for feat in features:
        sub = combined[['Method', 'Cluster', feat]].dropna()
        if sub.shape[0] < 6:
            print(f"  ⚠️  Skipping '{feat}': too few observations ({sub.shape[0]})")
            continue
        if sub['Method'].nunique() < 2 or sub['Cluster'].nunique() < 2:
            print(f"  ⚠️  Skipping '{feat}': not enough factor levels")
            continue

        # Sanitise column name for statsmodels formula
        safe = feat.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        sub = sub.rename(columns={feat: safe})

        try:
            model = ols(f'{safe} ~ C(Method) * C(Cluster)', data=sub).fit()
            table = anova_lm(model, typ=2)

            records.append({
                'Feature': feat,
                'P-val\nMethod':      table.loc['C(Method)',              'PR(>F)'],
                'P-val\nCluster':     table.loc['C(Cluster)',             'PR(>F)'],
                'P-val\nInteraction': table.loc['C(Method):C(Cluster)',   'PR(>F)'],
            })
        except Exception as e:
            print(f"  ⚠️  Error on '{feat}': {e}")

    if not records:
        print("  ❌ No features produced valid ANOVA results.")
        return

    result = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 5. FDR correction (Benjamini-Hochberg)
    # ------------------------------------------------------------------
    for col, fdr_col in [
        ('P-val\nMethod',      'P-val\nMethod\nFDR'),
        ('P-val\nCluster',     'P-val\nCluster\nFDR'),
        ('P-val\nInteraction', 'P-val\nInteraction\nFDR'),
    ]:
        pvals = result[col].values
        _, fdr, _, _ = multipletests(pvals, method='fdr_bh')
        result[fdr_col] = fdr

    # Reorder columns
    result = result[[
        'Feature',
        'P-val\nMethod', 'P-val\nCluster', 'P-val\nInteraction',
        'P-val\nMethod\nFDR', 'P-val\nCluster\nFDR', 'P-val\nInteraction\nFDR',
    ]]

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    _pval_threshold = float(os.environ.get("PIPELINE_ANOVA_PVAL", "0.05"))

    def _highlight(val):
        try:
            if val < _pval_threshold:
                return 'color: blue;'
        except TypeError:
            pass
        return ''

    os.makedirs(os.path.dirname(output_xlsx) or '.', exist_ok=True)

    try:
        styled = result.style.map(_highlight, subset=[
            'P-val\nMethod', 'P-val\nCluster', 'P-val\nInteraction',
            'P-val\nMethod\nFDR', 'P-val\nCluster\nFDR', 'P-val\nInteraction\nFDR',
        ])
        styled.to_excel(output_xlsx, index=False, engine='openpyxl')
    except (AttributeError, ImportError):
        result.to_excel(output_xlsx, index=False, engine='openpyxl')
    result.to_csv(output_csv, index=False)

    # Summary
    n_sig_method = (result['P-val\nMethod\nFDR'] < _pval_threshold).sum()
    n_sig_cluster = (result['P-val\nCluster\nFDR'] < _pval_threshold).sum()
    n_sig_inter = (result['P-val\nInteraction\nFDR'] < _pval_threshold).sum()
    print(f"  ✅ Saved '{output_xlsx}'")
    print(f"  ✅ Saved '{output_csv}'")
    print(f"  📊 Significant features (FDR < {_pval_threshold}):")
    print(f"     Method ({dynamic_label} vs Static): {n_sig_method}/{len(result)}")
    print(f"     Cluster:                    {n_sig_cluster}/{len(result)}")
    print(f"     Interaction:                {n_sig_inter}/{len(result)}")

    # ------------------------------------------------------------------
    # 7. Z-scored feature-mean heatmaps
    # ------------------------------------------------------------------
    heatmap_png = output_xlsx.replace('.xlsx', '_zscore_heatmap.png')
    try:
        _plot_zscore_heatmaps(
            df_dyn_u, df_sta_u, features,
            label_a=dynamic_label, label_b='Static',
            output_png=heatmap_png, k_val=k_val,
        )
    except Exception as e:
        print(f"  ⚠️  Heatmap generation failed: {e}")

    # ------------------------------------------------------------------
    # 8. Tukey HSD FDR-adjusted p-value heatmaps
    # ------------------------------------------------------------------
    tukey_png = output_xlsx.replace('.xlsx', '_tukey_heatmap.png')
    try:
        _run_tukey_and_plot(
            df_dyn_u, df_sta_u, features,
            label_a=dynamic_label, label_b='Static',
            output_png=tukey_png, k_val=k_val,
        )
    except Exception as e:
        print(f"  ⚠️  Tukey heatmap generation failed: {e}")


# ================================================================
# Auto-discover k-files and pair with the static CSV
# ================================================================
_emb_dir    = os.environ.get("PIPELINE_EMB_CLUSTER_DIR", "clustering")
_fit_dir    = os.environ.get("PIPELINE_EMBFIT_CLUSTER_DIR", "fitting")
_static_csv = os.environ.get("PIPELINE_STATIC_CSV",
                             os.path.join("cell_data", "static_clustering_raw_data.csv"))
_out_dir    = os.environ.get("PIPELINE_TWOWAY_OUTPUT_DIR", "results")

if not os.path.exists(_static_csv):
    print(f"❌ Static clustering CSV not found: {_static_csv}")
    print("   Place the file in cell_data/ or set PIPELINE_STATIC_CSV env var.")
else:
    # Read cluster count from static file once
    _sta_preview = pd.read_csv(_static_csv, usecols=['Cluster'], nrows=1_000_000)
    _static_k = _sta_preview['Cluster'].nunique()
    print(f"Static file: {_static_csv}  (k={_static_k} clusters)")

    # ------------------------------------------------------------------
    # Comparison 1: Embedding vs Static
    # ------------------------------------------------------------------
    emb_jobs = []
    for emb_path in sorted(glob.glob(os.path.join(_emb_dir, "Merged_Clusters_PCA_k*.csv"))):
        basename = os.path.basename(emb_path)
        k_part = basename.replace("Merged_Clusters_PCA_k", "").replace(".csv", "")
        try:
            k_val = int(k_part)
        except ValueError:
            continue
        out_xlsx = os.path.join(_out_dir, f"two_way_anova_emb_vs_static_k{k_val}.xlsx")
        out_csv  = os.path.join(_out_dir, f"two_way_anova_emb_vs_static_k{k_val}.csv")
        emb_jobs.append((emb_path, k_val, out_xlsx, out_csv))

    if not emb_jobs:
        emb_default = os.environ.get("PIPELINE_CLUSTERS_CSV",
                                     os.path.join(_emb_dir, "Merged_Clusters_PCA.csv"))
        if os.path.exists(emb_default):
            emb_jobs.append((
                emb_default, None,
                os.path.join(_out_dir, "two_way_anova_emb_vs_static.xlsx"),
                os.path.join(_out_dir, "two_way_anova_emb_vs_static.csv"),
            ))

    if not emb_jobs:
        print(f"\n⚠️  No embedding clustering files found in {_emb_dir}/")
    else:
        for emb_path, k_val, out_xlsx, out_csv in emb_jobs:
            label = f"k={k_val}" if k_val else "default"
            print(f"\n{'='*60}")
            print(f"Two-Way ANOVA: Embedding vs Static — {label}")
            print(f"  Embedding: {emb_path}")
            print(f"  Static:    {_static_csv}")
            print(f"{'='*60}")
            run_two_way_anova(emb_path, _static_csv, out_xlsx, out_csv, k_val,
                              dynamic_label="Embedding")

    # ------------------------------------------------------------------
    # Comparison 2: Embedding+Fitting vs Static
    # ------------------------------------------------------------------
    fit_jobs = []
    for fit_path in sorted(glob.glob(os.path.join(_fit_dir, "embedding_fitting_Merged_Clusters_PCA_k*.csv"))):
        basename = os.path.basename(fit_path)
        k_part = basename.replace("embedding_fitting_Merged_Clusters_PCA_k", "").replace(".csv", "")
        try:
            k_val = int(k_part)
        except ValueError:
            continue
        out_xlsx = os.path.join(_out_dir, f"two_way_anova_embfit_vs_static_k{k_val}.xlsx")
        out_csv  = os.path.join(_out_dir, f"two_way_anova_embfit_vs_static_k{k_val}.csv")
        fit_jobs.append((fit_path, k_val, out_xlsx, out_csv))

    if not fit_jobs:
        fit_default = os.environ.get("PIPELINE_EMBFIT_CLUSTERS_CSV",
                                     os.path.join(_fit_dir, "embedding_fitting_Merged_Clusters_PCA.csv"))
        if os.path.exists(fit_default):
            fit_jobs.append((
                fit_default, None,
                os.path.join(_out_dir, "two_way_anova_embfit_vs_static.xlsx"),
                os.path.join(_out_dir, "two_way_anova_embfit_vs_static.csv"),
            ))

    if not fit_jobs:
        print(f"\n⚠️  No embedding+fitting clustering files found in {_fit_dir}/")
        print("   Run Step 6b (Emb+Fit Clustering) first, or set PIPELINE_EMBFIT_CLUSTER_DIR.")
    else:
        for fit_path, k_val, out_xlsx, out_csv in fit_jobs:
            label = f"k={k_val}" if k_val else "default"
            print(f"\n{'='*60}")
            print(f"Two-Way ANOVA: Emb+Fitting vs Static — {label}")
            print(f"  Emb+Fit: {fit_path}")
            print(f"  Static:  {_static_csv}")
            print(f"{'='*60}")
            run_two_way_anova(fit_path, _static_csv, out_xlsx, out_csv, k_val,
                              dynamic_label="Emb+Fitting")
