import pandas as pd
import torch
import numpy as np
import json
import traceback
import umap.umap_ as umap
import time

from chronos import BaseChronosPipeline
from huggingface_hub import hf_hub_download

model_name_map = {
    "ChronosT5Tiny": "amazon/chronos-t5-tiny",
    "ChronosT5Mini": "amazon/chronos-t5-mini",
    "ChronosT5Small": "amazon/chronos-t5-small",
    "ChronosT5Base": "amazon/chronos-t5-base",
    "ChronosT5Large": "amazon/chronos-t5-large",
    "ChronosBoltTiny": "amazon/chronos-bolt-tiny",
    "ChronosBoltMini": "amazon/chronos-bolt-mini",
    "ChronosBoltSmall": "amazon/chronos-bolt-small",
    "ChronosBoltBase": "amazon/chronos-bolt-base",
    "MoiraiSmall": "Salesforce/moirai-1.1-R-small",
    "MoiraiBase": "Salesforce/moirai-1.1-R-base",
    "MoiraiLarge": "Salesforce/moirai-1.1-R-large",
    "MoiraiMoESmall": "Salesforce/moirai-1.1-MoE-small",
    "MoiraiMoEBase": "Salesforce/moirai-1.1-MoE-base",
    "TimesFM_1_0_200m": "google/timesfm-1.0-200m",
    "TimesFM_2_0_500m": "google/timesfm-2.0-500m"
}

def infer_source(model_name: str) -> str:
    if model_name.startswith("Chronos"):
        return "chronos"
    elif model_name.startswith("Moirai"):
        return "moirai"
    elif model_name.startswith("TimesFM"):
        return "timesfm"
    else:
        raise ValueError(f"Cannot infer source from model name: {model_name}")

def load_model(model_name: str, source: str):
    hf_model_id = model_name_map[model_name]
    if source == "chronos":
        return BaseChronosPipeline.from_pretrained(hf_model_id, device_map="auto", torch_dtype=torch.float32)
    else:
        raise ValueError("Only 'chronos' models are supported in this version.")

def embed_feature(model, tensor):
    with torch.no_grad():
        embed_tensor, _ = model.embed(tensor)
        selected = embed_tensor[:, 1, :]
        pooled = selected.mean(dim=0).detach().cpu().numpy()
    return pooled

def run_embedding_pipeline(df, model_dict_path, output_path, dim=3):
    start_time = time.time()
    df = df.copy()
    df["Parent"] = df["Parent"].astype(str)
    df["Experiment"] = df["Experiment"].astype(str)

    with open(model_dict_path, "r") as f:
        feature_model_dict = json.load(f)

    model_cache = {}
    excluded_features = ["Experiment", "Parent", "TimeIndex", "dt", 'ID']
    all_features = [f for f in df.columns if f not in excluded_features]
    feature_embeddings = {feature: {} for feature in all_features}

    cell_groups = list(df.groupby(["Experiment", "Parent"]))
    first_exp, first_parent = cell_groups[0][0]

    for feature in all_features:
        if feature not in feature_model_dict:
            print(f"Skipping embedding for {feature}, using mean instead.")
            continue

        model_name = feature_model_dict[feature]
        print(f"\nProcessing feature '{feature}' with model '{model_name}'")
        source = infer_source(model_name)
        key = (model_name, source)
        if key not in model_cache:
            model_cache[key] = load_model(model_name, source)
        model = model_cache[key]

        pooled_vectors = []
        valid_cells = []

        for (exp, parent), group in cell_groups:
            group_sorted = group.sort_values("TimeIndex")
            ts_values = group_sorted[feature].values.reshape(-1, 1)
            try:
                context_tensor = torch.tensor(ts_values.flatten(), dtype=torch.float32)
                pooled = embed_feature(model, context_tensor)
                pooled_vectors.append(pooled)
                valid_cells.append((exp, parent))
            except Exception:
                traceback.print_exc()

        print(f"UMAP input shape for {feature}: {np.array(pooled_vectors).shape}")
        reducer = umap.UMAP(n_components=dim, random_state=42)
        reduced = reducer.fit_transform(pooled_vectors)
        print(f"UMAP output shape for {feature}: {reduced.shape}")

        for (exp, parent), vec in zip(valid_cells, reduced):
            feature_embeddings[feature][(exp, parent)] = vec

    results = []
    for (exp, parent), group in cell_groups:
        group_sorted = group.sort_values("TimeIndex")
        result_dict = {
            "Experiment": exp,
            "Parent": parent
        }

        print(f"\n--- Final embedding for cell {exp}-{parent} ---") if (exp, parent) == (first_exp, first_parent) else None

        for feature in all_features:
            if feature in feature_model_dict:
                vec = feature_embeddings[feature].get((exp, parent), [float("nan")] * dim)
                result_dict[feature] = [float(v) for v in vec]
                if (exp, parent) == (first_exp, first_parent):
                    print(f"{feature} (UMAP): {vec}")
            else:
                try:
                    mean_val = group_sorted[feature].astype(float).mean()
                    result_dict[feature] = [mean_val]
                    if (exp, parent) == (first_exp, first_parent):
                        print(f"{feature} (mean): {mean_val}")
                except:
                    result_dict[feature] = [float("nan")]
                    if (exp, parent) == (first_exp, first_parent):
                        print(f"{feature} (mean): NaN")

        results.append(result_dict)
        embedding_lengths = sum(len(v) for k,v in result_dict.items() if k not in ['Experiment','Parent'])
        print(f"Processed {exp}-{parent} → Feature count: {len(result_dict) - 2}, Embedding length: {embedding_lengths}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    duration = time.time() - start_time
    print(f"\nSaved all embeddings to {output_path}")
    print(f"Total runtime: {duration / 60:.2f} minutes")

# Example usage:
df = pd.read_csv("MergedAndFilteredExperiment008.csv")
run_embedding_pipeline(
    df=df,
    model_dict_path="best_chronos_model_per_feature.json",
    output_path="Embedding008.json")

