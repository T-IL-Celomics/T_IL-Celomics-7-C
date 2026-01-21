import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os

# ========== PARAMETERS ==========
EMBED_JSON = "Embedding008.json"
COMBINED_JSON = "embedding_fitting_combined_by_feature_scaled.json"
EMBED_LABELS_CSV = "Embedding - K=3/cluster_assignments_k3.csv"
COMBINED_LABELS_CSV = "Embedding - K=3/embedding_fitting_Merged_Clusters_PCA.csv"
TEST_SIZE = 0.15 # 15% for test set
RANDOM_STATE = 42

# ========== 1. LOAD + PROCESS EMBEDDING ONLY ==========
with open(EMBED_JSON) as f:
    embed_data = json.load(f)

embed_records = []
for cell in embed_data:
    vec = []
    for feat, val in cell.items():
        if feat in ["Experiment", "Parent"]:
            continue
        vec.extend(val if isinstance(val, list) else [val])
    if not any(np.isnan(vec)):
        embed_records.append({
            "Experiment": str(cell["Experiment"]),
            "Parent": str(cell["Parent"]),
            "Vector": vec
        })

df_embed = pd.DataFrame(embed_records)
df_embed["CellID"] = df_embed["Experiment"] + "_" + df_embed["Parent"]

X_embed = np.stack(df_embed["Vector"].values)
scaler = StandardScaler()
X_embed_scaled = scaler.fit_transform(X_embed)

# ========== 2. LOAD + PROCESS COMBINED (Embedding + Fitting) ==========
with open(COMBINED_JSON) as f:
    combined_data = json.load(f)

combined_records = []
for cell in combined_data:
    vec = []
    for feat, val in cell.items():
        if feat in ["Experiment", "Parent", "Treatment"]:
            continue
        vec.extend(val)
    if not any(np.isnan(vec)):
        combined_records.append({
            "Experiment": str(cell["Experiment"]),
            "Parent": str(cell["Parent"]),
            "Vector": vec
        })

df_combined = pd.DataFrame(combined_records)
df_combined["CellID"] = df_combined["Experiment"] + "_" + df_combined["Parent"]
X_combined = np.stack(df_combined["Vector"].values)

# ========== 3. LOAD LABELS ==========
labels_embed = pd.read_csv(EMBED_LABELS_CSV)
labels_embed["CellID"] = labels_embed["Experiment"].astype(str) + "_" + labels_embed["Parent"].astype(str)
df_embed = df_embed.merge(labels_embed[["CellID", "Cluster"]], on="CellID", how="inner")

labels_combined = pd.read_csv(COMBINED_LABELS_CSV)
labels_combined["CellID"] = labels_combined["Experiment"].astype(str) + "_" + labels_combined["Parent"].astype(str)
# Take unique cluster per cell (assumes cluster is constant for the cell)
labels_combined = labels_combined.drop_duplicates(subset="CellID")[["CellID", "Cluster"]]
df_combined = df_combined.merge(labels_combined[["CellID", "Cluster"]], on="CellID", how="inner")

# ========== 4. TRAIN/TEST SPLIT BY SHARED CELLS ==========
common_ids = set(df_embed["CellID"]).intersection(set(df_combined["CellID"]))
common_ids = list(common_ids)

# Create stable test split
cellid_cluster_df = df_embed[df_embed["CellID"].isin(common_ids)][["CellID", "Cluster"]].drop_duplicates()

train_ids, test_ids = train_test_split(
    cellid_cluster_df["CellID"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=cellid_cluster_df["Cluster"]
)

# Create masks
mask_embed = df_embed["CellID"].isin(test_ids)
mask_comb = df_combined["CellID"].isin(test_ids)

X_train_embed = X_embed_scaled[~mask_embed]
X_test_embed  = X_embed_scaled[mask_embed]
y_train_embed = df_embed.loc[~mask_embed, "Cluster"].values
y_test_embed  = df_embed.loc[mask_embed, "Cluster"].values

X_train_comb = X_combined[~mask_comb]
X_test_comb  = X_combined[mask_comb]
y_train_comb = df_combined.loc[~mask_comb, "Cluster"].values
y_test_comb  = df_combined.loc[mask_comb, "Cluster"].values

# ========== 5. FIT + EVALUATE RANDOM FOREST ==========

def train_evaluate(X_train, X_test, y_train, y_test, label, save_dir="Embedding - K=3"):
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Print to console
    print(f"\n=== Results for {label} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save classification report to Excel
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(save_dir, f"classification_report_{label.replace(' ', '_')}.xlsx")
    report_df.to_excel(report_path)
    print(f"Saved classification report to {report_path}")

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    plt.figure(figsize=(6,6))
    disp.plot(cmap="Blues", values_format=".0f")
    plt.title(f"Confusion Matrix - {label}")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, f"confusion_matrix_{label.replace(' ', '_')}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

train_evaluate(X_train_embed, X_test_embed, y_train_embed, y_test_embed, "Embedding Only")
train_evaluate(X_train_comb, X_test_comb, y_train_comb, y_test_comb, "Embedding + Fitting")
