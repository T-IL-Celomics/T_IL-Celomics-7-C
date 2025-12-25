import os
import re
import pandas as pd
import numpy as np
import json
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import warnings

os.makedirs("regression", exist_ok=True)
os.makedirs("figures", exist_ok=True)

warnings.filterwarnings("ignore")

# ========== STEP 1: Define Models ==========
def linear(x, a, b): return a * x + b
def exponential(x, a, b): return a * np.exp(b * x)
def logarithmic(x, a, b): return a * np.log(b * x + 1)
def logistic(x, a, b, c): return c / (1 + a * np.exp(-b * x))
def power(x, a, b): return a * np.power(x, b)
def inverse(x, a, b): return a / (x + b)
def quadratic(x, a, b, c): return a * x**2 + b * x + c
def sigmoid(x, a, b, c, d): return d + (a - d) / (1.0 + (x / c)**b)
def gompertz(x, a, b, c): return a * np.exp(-b * np.exp(-c * x))
def weibull(x, a, b, c): return a - b * np.exp(-c * x**2)
def poly3(x, a, b, c, d): return a * x**3 + b * x**2 + c * x + d
def poly4(x, a, b, c, d, e): return a * x**4 + b * x**3 + c * x**2 + d * x + e

models = {
    'linear':      (linear,      [1, 1]),
    'exponential': (exponential, [1, 0.1]),
    'logarithmic': (logarithmic, [1, 1]),
    'logistic':    (logistic,    [1, 1, 1]),
    'power':       (power,       [1, 1]),
    'inverse':     (inverse,     [1, 1]),
    'quadratic':   (quadratic,   [1, 1, 1]),
    'sigmoid':     (sigmoid,     [1, 1, 1, 1]),
    'gompertz':    (gompertz,    [1, 1, 1]),
    'weibull':     (weibull,     [1, 1, 1]),
    'poly3':       (poly3,       [1, 1, 1, 1]),
    'poly4':       (poly4,       [1, 1, 1, 1, 1]),
}

def fit_model(model_func, xdata, ydata, p0):
    try:
        popt, _ = curve_fit(model_func, xdata, ydata, p0=p0, maxfev=10000)
        y_pred = model_func(xdata, *popt)
        r2   = r2_score(ydata, y_pred)
        rmse = np.sqrt(mean_squared_error(ydata, y_pred))
        X_design = np.column_stack([xdata**i for i in range(len(popt))])
        X_design = sm.add_constant(X_design)
        pval = sm.OLS(ydata, X_design).fit().f_pvalue
        return popt, r2, rmse, pval
    except:
        return None, None, None, None

# ========== STEP 2: Fit & Save CSVs ==========
csv_exist = (
    os.path.exists("regression/fitting_all_models.csv")
)

if not csv_exist:
    df = pd.read_csv("cell_data/raw_all_cells.csv")
    df["Experiment"] = df["Experiment"].astype(str)
    df["Parent"]     = df["Parent"].astype(str)

    exclude_cols = {"TimeIndex", "Parent", "Experiment", "unique_id", "ds"}
    all_features  = [c for c in df.columns if c not in exclude_cols]

    all_models = []
    total = df.groupby(["Experiment", "Parent"]).ngroups

    for i, ((exp, parent), group) in enumerate(df.groupby(["Experiment", "Parent"]), start=1):
        print(f"Processing cell {i} of {total}: {exp} / {parent}")
        t = group["TimeIndex"].astype(float).values
        x = t - t[0]

        for feat in all_features:
            y = group[feat].values
            if np.isnan(y).any(): 
                continue
            for name, (func, p0) in models.items():
                popt, r2, rmse, pval = fit_model(func, x, y, p0)
                row = {
                    "Experiment": exp, "Parent": parent,
                    "Feature": feat, "Model": name,
                    "R2": r2, "RMSE": rmse, "pval": pval
                }
                for idx_p in range(len(p0)):
                    row[f"param_{idx_p}"] = popt[idx_p] if popt is not None else np.nan
                all_models.append(row)

    df_all = pd.DataFrame(all_models)
    df_all.to_csv("regression/fitting_all_models.csv", index=False)
    print("regression/fitting_all_models.csv saved.")
else:
    print("Loading existing regression/fitting_all_models.csv")
    df_all = pd.read_csv("regression/fitting_all_models.csv")
# ========== STEP 3: Select Top Models & Compute NRMSE on df_top3 ==========
# 1) significant fits
df_sig = df_all[df_all["pval"] < 0.05].copy()

# === NEW: compute range for NRMSE on df_sig ===
raw = pd.read_csv("cell_data/raw_all_cells.csv")
raw["Experiment"] = raw["Experiment"].astype(str)
raw["Parent"]     = raw["Parent"].astype(str)
exclude = {"TimeIndex", "Parent", "Experiment", "unique_id", "ds"}
feat_cols = [c for c in raw.columns if c not in exclude]

# reshape to long format
raw_long = raw.melt(
    id_vars=["Experiment", "Parent"],
    value_vars=feat_cols,
    var_name="Feature",
    value_name="value"
)

# compute range per (cell, feature)
ranges = (
    raw_long
    .groupby(["Experiment", "Parent", "Feature"])["value"]
    .agg(lambda s: s.max() - s.min())
    .reset_index(name="range")
)

# Ensure consistent types before merge
df_sig["Experiment"] = df_sig["Experiment"].astype(str)
df_sig["Parent"]     = df_sig["Parent"].astype(str)

ranges["Experiment"] = ranges["Experiment"].astype(str)
ranges["Parent"]     = ranges["Parent"].astype(str)

# merge with df_sig
df_sig = df_sig.merge(
    ranges,
    on=["Experiment", "Parent", "Feature"],
    how="left"
)

# compute NRMSE
df_sig["NRMSE"] = df_sig["RMSE"] / df_sig["range"]

# save updated df_sig with NRMSE
df_sig.to_csv("regression/fitting_significant_models_with_nrmse.csv", index=False)

# 2) pick Top‐3 by NRMSE
df_top3 = (
    df_sig
    .sort_values(by=["Experiment", "Parent", "Feature", "NRMSE"])
    .groupby(["Experiment", "Parent", "Feature"])
    .head(3)
    .reset_index(drop=True)
)

# 3) pick Top‐1 by NRMSE
df_top1 = (
    df_top3
    .sort_values(by=["Experiment", "Parent", "Feature", "NRMSE"])
    .groupby(["Experiment", "Parent", "Feature"])
    .head(1)
    .reset_index(drop=True)
)

# 4) save outputs
df_top3.to_csv("regression/fitting_top3_with_nrmse.csv", index=False)
df_top1.to_csv("regression/fitting_best_with_nrmse.csv",  index=False)
print("Saved: regression/fitting_top3_with_nrmse.csv and regression/fitting_best_with_nrmse.csv (based on NRMSE)")

# ========== STEP 4: Visualizations ==========
# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

# 1) Parameter distributions for each dataset
numeric_cols = [c for c in df_all.columns if c.startswith("param_")] + ["R2","RMSE","pval"]

def plot_distributions(dfs, labels, title, fname, pct_low=1, pct_high=99):
    """
    Plot histograms for each numeric column (excluding 'pval'),
    clipping to a central percentile range and annotating outliers.
    
    dfs    : list of DataFrames
    labels : list of plot labels
    title  : suptitle for the row of plots
    fname  : base filename for saving
    pct_low/high : percentiles to clip range (default 1st–99th)
    """
    for col in numeric_cols:
        if col == "pval":
            continue  # skip pval entirely
        
        fig, axs = plt.subplots(1, len(dfs), figsize=(5*len(dfs),4), sharey=True)
        for ax, df, lbl in zip(axs, dfs, labels):
            # extract and clean data
            raw = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if raw.empty:
                ax.text(0.5, 0.5, "no data", ha='center')
                continue
            
            # determine clipping bounds
            low, high = np.nanpercentile(raw, [pct_low, pct_high])
            clipped = raw[(raw >= low) & (raw <= high)]
            n_total = len(raw)
            n_clipped = len(clipped)
            n_outliers = n_total - n_clipped
            
            # plot histogram of clipped data
            ax.hist(clipped, bins=40, edgecolor="black")
            ax.set_xlim(low, high)
            ax.set_title(f"{lbl}\n{col}")
            ax.set_yscale("log")
            
            # annotate outlier count
            ax.text(
                0.95, 0.95,
                f"outliers: {n_outliers}",
                ha='right', va='top', transform=ax.transAxes,
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
        
        plt.suptitle(title)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(f"figures/{fname}_{col}.png", dpi=300)
        plt.close()

# Example call:
plot_distributions(
    [df_all, df_top3, df_top1],
    ["All Models", "Top 3 Models", "Best Model"],
    "Parameter Distributions (1–99 percentile clipped)",
    "dist_clipped"
)

# 2) Stacked‐bar per feature → stacked_features_*
import matplotlib.pyplot as plt
import seaborn as sns

def plot_stacked_features(df, filename, title, palette_name='Set3'):
    """
    Plot a single stacked‐bar chart of Model proportions per Feature
    using a categorical seaborn palette.
      - df          : DataFrame with 'Feature','Model','pval'
      - filename    : base name for saving (no suffix)
      - title       : plot title
      - palette_name: any seaborn categorical palette (e.g. 'Set3','Paired','tab20')
    """
    # filter significant
    df_plot = df[df["pval"] < 0.05]
    # count & normalize
    counts = df_plot.groupby(["Feature","Model"]).size().unstack(fill_value=0)
    props  = counts.div(counts.sum(axis=1), axis=0)

    # determine colors in fixed model‐order
    models_list = props.columns.tolist()
    colors      = sns.color_palette(palette_name, n_colors=len(models_list))

    fig, ax = plt.subplots(figsize=(14,6))
    # plot with explicit colors
    props.plot(
        kind    ='bar',
        stacked =True,
        ax      =ax,
        color   =colors
    )

    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02,1))
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Proportion")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

    plt.tight_layout(rect=[0,0,0.85,1])
    out_path = f"figures/stacked_features_{filename}.png"
    fig.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


plot_stacked_features(df_sig,  "all_models_sig", "Significant Models per Feature (p<0.05)", palette_name='Paired_r')
plot_stacked_features(df_top3,"top3_models",    "Top 3 Models per Feature",               palette_name='Paired_r')
plot_stacked_features(df_top1,"best_model",     "Best Model per Feature",                 palette_name='Paired_r')


# 3) Boxplot RMSE by model
plt.figure(figsize=(12,6))
sns.boxplot(data=df_sig, x="Model", y="RMSE")
plt.title("RMSE Distribution by Model (p<0.05)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/boxplot_rmse_by_model.png", dpi=300)
plt.close()

# 4) Plot mean RMSE with error bars (± STD)
stats_model = df_sig.groupby("Model")[["RMSE","R2"]].agg(["mean","std"]).reset_index()
plt.figure(figsize=(10,5))
plt.errorbar(
    x=stats_model["Model"],
    y=stats_model[("RMSE","mean")],
    yerr=stats_model[("RMSE","std")],
    fmt="o", capsize=5
)
plt.xticks(rotation=45)
plt.ylabel("Mean RMSE ± STD")
plt.title("Average RMSE by Model (p<0.05)")
plt.tight_layout()
plt.savefig("figures/avg_rmse_by_model.png", dpi=300)
plt.close()

# 5) Plot mean R² with error bars (± STD)
plt.figure(figsize=(10,5))
plt.errorbar(
    x=stats_model["Model"],
    y=stats_model[("R2","mean")],
    yerr=stats_model[("R2","std")],
    fmt="o", capsize=5, color="C1"
)
plt.xticks(rotation=45)
plt.ylabel("Mean R² ± STD")
plt.title("Average R² by Model (p<0.05)")
plt.tight_layout()
plt.savefig("figures/avg_r2_by_model.png", dpi=300)
plt.close()

# 6) Boxplot of RMSE per feature
plt.figure(figsize=(14,6))
sns.boxplot(data=df_sig, x="Feature", y="RMSE")
plt.xticks(rotation=90)
plt.ylabel("RMSE")
plt.title("RMSE Distribution per Feature (p<0.05)")
plt.tight_layout()
plt.savefig("figures/boxplot_rmse_by_feature.png", dpi=300)
plt.close()

# ========== STEP 5: Treatment Analysis ==========
treatment_map = {
    'C02':'BRCACON1','D02':'BRCACON2','E02':'BRCACON3',
    'F02':'BRCACON4','G02':'BRCACON5','B02':'CON0',
}

def extract_location(exp_str):
    m = re.search(r'CHR\d+([A-Z]\d{2})', exp_str)
    return m.group(1) if m else None

def add_treatment_column(df):
    df = df.copy()
    df['Location']  = df['Experiment'].apply(extract_location)
    df['Treatment'] = df['Location'].map(treatment_map)
    return df

df_sig  = add_treatment_column(df_sig)
df_top3 = add_treatment_column(df_top3)
df_top1 = add_treatment_column(df_top1)

df_sig_all  = df_sig[df_sig['pval'] < 0.05]
df_sig_top3 = df_top3[df_top3['pval'] < 0.05]
df_sig_top1 = df_top1[df_top1['pval'] < 0.05]

param_cols = [c for c in df_sig.columns if c.startswith("param_")]

def plot_stacked_treatments(df, label):
    counts = df.groupby(['Treatment','Model']).size().unstack(fill_value=0)
    props  = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10,6))
    props.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.legend(title='Model', bbox_to_anchor=(1.02,1), loc='upper left')
    ax.set_title(f'Model Proportions by Treatment ({label})')
    ax.set_xlabel('Treatment')
    ax.set_ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0,0,0.85,1])
    fig.savefig(f"figures/stacked_treatments_{label.replace(' ','_')}.png", dpi=300)
    plt.close()

plot_stacked_treatments(df_sig_all,  "All_Models")
plot_stacked_treatments(df_sig_top3, "Top3_Models")
plot_stacked_treatments(df_sig_top1, "Best_Model")

def plot_rmse_by_treatment(df, label):
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x='Treatment', y='RMSE', ax=ax)
    ax.set_title(f'RMSE by Treatment ({label}, p<0.05)')
    ax.set_xlabel('Treatment'); ax.set_ylabel('RMSE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(f"figures/boxplot_rmse_by_treatment_{label.replace(' ','_')}.png", dpi=300)
    plt.close()

plot_rmse_by_treatment(df_sig_all,  "All_Models")
plot_rmse_by_treatment(df_sig_top3, "Top3_Models")
plot_rmse_by_treatment(df_sig_top1, "Best_Model")

# ========== STEP 5b: Best‐Model by Treatment Subplots (2×3 layout, consistent tab20 colors) ==========
def plot_best_model_by_treatment_grid(df_best,
                                      feature_col="Feature",
                                      model_col="Model",
                                      treatment_col="Treatment",
                                      palette_name="Set3"):
    # ensure treatment column exists
    if treatment_col not in df_best:
        df_best = add_treatment_column(df_best)

    # fix model order and generate colors
    full_models = sorted(models.keys())
    colors      = sns.color_palette(palette_name, n_colors=len(full_models))

    # Canonical order: קונטרול (CON0) ראשון
    canonical = ["CON0", "BRCACON1", "BRCACON2", "BRCACON3", "BRCACON4", "BRCACON5"]
    present   = set(df_best[treatment_col].dropna().unique())
    treatments = [t for t in canonical if t in present]
    assert len(treatments) <= 6, "Up to 6 treatments supported."

    # create 2×3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    axes = axes.flatten()

    for ax, tr in zip(axes, treatments):
        subset = df_best[df_best[treatment_col] == tr]
        counts = (
            subset
            .groupby([feature_col, model_col])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=full_models, fill_value=0)
        )
        props = counts.div(counts.sum(axis=1), axis=0)

        props.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=colors,
            legend=False
        )
        ax.set_title(tr)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.set_ylabel("Proportion")

    # turn off unused axes
    for ax in axes[len(treatments):]:
        ax.axis("off")

    # give legend some room above
    fig.subplots_adjust(top=0.85)
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(full_models))]
    fig.legend(handles, full_models,
               title="Model",
               loc="upper center",
               ncol=len(full_models),
               bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = "figures/stacked_best_model_by_treatment_grid.png"
    fig.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


plot_best_model_by_treatment_grid(df_top1, palette_name="Paired_r")


# ---- OPTION 1: Categorical Heatmap ----

# 1) Full list of your model names, sorted alphabetically
full_models = sorted(models.keys())

palette = sns.color_palette("pastel", n_colors=len(full_models))
cmap    = ListedColormap(palette)
norm    = BoundaryNorm(
    boundaries=np.arange(-0.5, len(full_models)+0.5),
    ncolors=len(full_models)
)

# 3) Pivot to pick the modal Best‐Model per (Treatment, Feature)
pivot = df_top1.pivot_table(
    index="Treatment",
    columns="Feature",
    values="Model",
    aggfunc=lambda x: x.mode().iloc[0] if len(x)>0 else np.nan,
    fill_value=np.nan
)

# 4) Map model names → integer codes
code_map   = {m:i for i,m in enumerate(full_models)}
data_codes = pivot.replace(code_map).astype(float)

# 5) Draw the heatmap
fig, ax = plt.subplots(figsize=(18,6))
sns.heatmap(
    data_codes,
    cmap=cmap,
    norm=norm,
    mask=data_codes.isna(),
    cbar_kws={
        "ticks": list(range(len(full_models))),
        "format": lambda x, pos: full_models[int(x)]
    },
    ax=ax
)
ax.set_title("Best Model by Treatment × Feature")
ax.set_xlabel("Feature")
ax.set_ylabel("Treatment")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("figures/heatmap_best_model_by_treatment_feature.png", dpi=300)
plt.close()


# ---- OPTION 2: High-res / Vector Sunburst ----

df_sb = df_top1[['Treatment','Model','Feature']].dropna()
df_sb['count'] = 1

# build a discrete color map so each model is unique
sunburst_models = sorted(df_sb['Model'].unique())
plotly_palette  = px.colors.qualitative.Plotly
color_map = {
    m: plotly_palette[i % len(plotly_palette)]
    for i, m in enumerate(sunburst_models)
}

fig = px.sunburst(
    df_sb,
    path=['Treatment','Model','Feature'],
    values='count',
    color='Model',
    color_discrete_map=color_map,
    title='Sunburst of Treatment → Model → Feature (Best Models)'
)
fig.update_traces(textinfo='label+percent entry')

# save a high-res PNG
fig.write_image(
    "figures/sunburst_treatment_model_feature.png",
    width=2000, height=2000, scale=1  # bump width/height for clarity
)
# or interactive HTML:
fig.write_html("figures/sunburst_treatment_model_feature.html")



# ========== STEP 6: JSON Creation ==========

# Derive how many true parameters each model has
model_param_counts = {name: len(init_params) for name, (_, init_params) in models.items()}

# Fixed list of all required features
all_features = [
    'Area', 'Acceleration', 'Acceleration_OLD', 'Acceleration_X', 'Acceleration_Y',
    'Coll', 'Coll_CUBE', 'Confinement_Ratio', 'Directional_Change', 'Overall_Displacement',
    'Displacement_From_Last_Id', 'Displacement2', 'Ellip_Ax_B_X', 'Ellip_Ax_B_Y',
    'Ellip_Ax_C_X', 'Ellip_Ax_C_Y', 'EllipsoidAxisLengthB', 'EllipsoidAxisLengthC',
    'Ellipticity_oblate', 'Ellipticity_prolate', 'Instantaneous_Angle', 'Instantaneous_Speed',
    'Instantaneous_Speed_OLD', 'Linearity_of_Forward_Progression', 'Mean_Curvilinear_Speed',
    'Mean_Straight_Line_Speed', 'Current_MSD_1', 'Final_MSD_1', 'MSD_Linearity_R2_Score',
    'MSD_Brownian_Motion_BIC_Score', 'MSD_Brownian_D', 'MSD_Directed_Motion_BIC_Score',
    'MSD_Directed_D', 'MSD_Directed_v2', 'Sphericity', 'Total_Track_Displacement',
    'Track_Displacement_X', 'Track_Displacement_Y', 'Velocity_X', 'Velocity_Y',
    'Eccentricity', 'Min_Distance', 'Velocity_Full_Width_Half_Maximum',
    'Velocity_Time_of_Maximum_Height', 'Velocity_Maximum_Height', 'Velocity_Ending_Value',
    'Velocity_Ending_Time', 'Velocity_Starting_Value', 'Velocity_Starting_Time'
]

def create_json(df, num_params=5, models_per_feature=3, transformer=None):
    """
    If transformer is given, applies it to the imputed X matrix
    before writing back into the JSON structure.
    """
    out = []
    flat_vectors = []
    subvec_len = num_params  # params

    # Build the raw cell×feature vectors
    grouped = {(e,p):g for (e,p),g in df.groupby(["Experiment","Parent"])}
    for (exp,parent), grp in grouped.items():
        cell = {"Experiment":str(exp), "Parent":str(parent), "fitting":{}}
        for feat in all_features:
            if feat in grp["Feature"].values:
                rows = grp[grp["Feature"]==feat]
                row_vecs = []
                for _,row in rows.iterrows():
                    name = row["Model"]
                    true_n = model_param_counts.get(name, num_params)
                    raw = []
                    for i in range(num_params):
                        if i < true_n:
                            v = row.get(f"param_{i}", np.nan)
                            raw.append(np.nan if (pd.isna(v) or np.isinf(v)) else float(v))
                        else:
                            raw.append(0.0)
                    row_vecs.append(raw)
                # pad/truncate
                zero = [np.nan]*subvec_len
                while len(row_vecs)<models_per_feature:
                    row_vecs.append(zero.copy())
                row_vecs = row_vecs[:models_per_feature]
            else:
                row_vecs = [[np.nan]*subvec_len for _ in range(models_per_feature)]

            flat = [v for sub in row_vecs for v in sub]
            cell["fitting"][feat] = flat
            flat_vectors.append(flat)
        out.append(cell)

    # stack, impute
    X = np.array(flat_vectors, dtype=float)
    mask = np.isnan(X)
    num_imputed = int(mask.sum())

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_filled = imputer.fit_transform(X)

    # apply optional transform
    if transformer is not None:
        X_filled = transformer.fit_transform(X_filled)

    # write back
    idx = 0
    for cell in out:
        for feat in all_features:
            cell["fitting"][feat] = [float(v) for v in X_filled[idx]]
            idx += 1

    return out, num_imputed

# ——— original exports ———
# output_path1 = "fitting_best_model.json"
# if not os.path.exists(output_path1):
#     json_top1, imp1 = create_json(df_top1, num_params=5, models_per_feature=1)
#     with open(output_path1,"w") as f: json.dump(json_top1, f, indent=2)
#     print(f"Saved {output_path1} (imputed {imp1} values).")
# else: print(f"{output_path1} exists, skipping.")
#
# output_path3 = "fitting_top3_models.json"
# if not os.path.exists(output_path3):
#     json_top3, imp3 = create_json(df_top3, num_params=5, models_per_feature=3)
#     with open(output_path3,"w") as f: json.dump(json_top3, f, indent=2)
#     print(f"Saved {output_path3} (imputed {imp3} values).")
# else: print(f"{output_path3} exists, skipping.")
# ---------------------------------------

# transformer pipelines for log‐only and log+robust
# apply sign(x) * log1p(|x|) instead of plain log1p
sign_log1p_pipe = FunctionTransformer(
    func=lambda X: np.sign(X) * np.log1p(np.abs(X)),
    validate=False
)

log_z_pipe = Pipeline([
    ("sign_log1p", sign_log1p_pipe),
    ("z-score",     StandardScaler())
])

# ——— new: log-only JSON ———
for name, df_obj, mf in [
    ("best_model_log", df_top1, 1),
    ("top3_models_log", df_top3, 3)
]:
    out_json, imp = create_json(df_obj, num_params=5,
                                models_per_feature=mf,
                                transformer=sign_log1p_pipe)
    path = f"fitting_{name}.json"
    with open(path,"w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved {path} (imputed+logged {imp} NaNs)")

# ——— new: log-then-robust JSON ———
for name, df_obj, mf in [
    ("best_model_log_scaled", df_top1, 1),
    ("top3_models_log_scaled", df_top3, 3)
]:
    out_json, imp = create_json(df_obj, num_params=5,
                                models_per_feature=mf,
                                transformer=log_z_pipe)
    path = f"fitting_{name}.json"
    with open(f"fitting_{name}.json","w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved fitting_{name}.json (imputed+logged+z-scaled, {imp} NaNs)")

# ========== STEP 7: Check for null entries in JSON ==========
with open("fitting_top3_models_log.json") as f:
    data = json.load(f)
cnt = sum(
    1
    for cell in data
    for vec in cell["fitting"].values()
    for v in vec
    if v is None
)
print("null entries in JSON:", cnt)
