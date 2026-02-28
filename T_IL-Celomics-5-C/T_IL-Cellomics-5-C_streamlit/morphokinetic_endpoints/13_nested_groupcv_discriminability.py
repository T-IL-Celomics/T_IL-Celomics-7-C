#!/usr/bin/env python3
"""13  Discriminability (nested GroupKFold blocked by Experiment)

Trains a logistic-regression classifier to predict a label column from
numeric features using nested GroupKFold CV (outer) with groups = Experiment.

CLI
---
    python 13_nested_groupcv_discriminability.py \
        --input ./out/table_with_clusters_and_pcs.csv \
        --label-col DoseCategory --exp-col Experiment \
        --outer-splits 5 --outdir ./out
"""
import argparse, os, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")

EXCLUDE = {"Experiment","Parent","Time","TimeIndex","Cluster","PC1","PC2",
           "Condition","DoseCategory","DoseCombo","Treatment","Treatments",
           "METR_Category","GABY_Category","METR_Norm","GABY_Norm",
           "Dose","Groups","Unnamed: 0","ID","x_Pos","y_Pos","dt","ds",
           "unique_id","n_frames","DoseLabel"}


def nested_cv(df, label_col, exp_col, outer_splits=5, exclude=None):
    exclude = exclude or EXCLUDE
    feats = [c for c in df.columns if c not in exclude and c != label_col
             and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Features: {len(feats)}")
    mask = df[label_col].notna()
    df = df[mask].copy()
    X = df[feats].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    groups = df[exp_col].values
    classes = le.classes_
    n_classes = len(classes)
    print(f"  Classes: {list(classes)}, n={len(y)}")

    n_groups = len(np.unique(groups))
    actual = min(outer_splits, n_groups)
    if actual < 2:
        print("  Not enough groups for CV")
        return None

    cv = GroupKFold(n_splits=actual)
    fold_results = []
    for fold_i, (tr, te) in enumerate(cv.split(X, y, groups)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2:
            continue
        clf = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                                  max_iter=5000, class_weight="balanced",
                                  random_state=42)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        ba = balanced_accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        fold_results.append({"fold": fold_i, "balanced_acc": ba, "macro_f1": f1,
                             "n_train": len(tr), "n_test": len(te)})
        print(f"  Fold {fold_i}: balanced_acc={ba:.3f}, macro_f1={f1:.3f}")

    if not fold_results:
        return None
    res = pd.DataFrame(fold_results)
    print(f"\n  Mean balanced_acc: {res['balanced_acc'].mean():.3f} ± {res['balanced_acc'].std():.3f}")
    print(f"  Mean macro_f1:    {res['macro_f1'].mean():.3f} ± {res['macro_f1'].std():.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--exp-col", default="Experiment")
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    print(f"Loaded {a.input}: {df.shape}")
    res = nested_cv(df, a.label_col, a.exp_col, a.outer_splits)
    if res is None:
        print("  No results produced.")
        return

    os.makedirs(a.outdir, exist_ok=True)
    out_csv = os.path.join(a.outdir, "discriminability_cv.csv")
    res.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(res["fold"], res["balanced_acc"], color="steelblue", edgecolor="black")
    axes[0].axhline(res["balanced_acc"].mean(), color="red", ls="--")
    axes[0].set_xlabel("Fold"); axes[0].set_ylabel("Balanced Accuracy"); axes[0].set_title("Balanced Accuracy per Fold")
    axes[1].bar(res["fold"], res["macro_f1"], color="darkorange", edgecolor="black")
    axes[1].axhline(res["macro_f1"].mean(), color="red", ls="--")
    axes[1].set_xlabel("Fold"); axes[1].set_ylabel("Macro F1"); axes[1].set_title("Macro F1 per Fold")
    fig.tight_layout()
    out_png = os.path.join(a.outdir, "discriminability_folds.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_png}")


if __name__ == "__main__":
    main()
