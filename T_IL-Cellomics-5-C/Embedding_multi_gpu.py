import os
import json
import time
import traceback
import argparse
import numpy as np
import pandas as pd
import torch
import umap.umap_ as umap
import multiprocessing as mp

from chronos import BaseChronosPipeline

BOLT_TO_T5_MAP = {
    "ChronosBoltTiny":  "ChronosT5Tiny",
    "ChronosBoltMini":  "ChronosT5Small",
    "ChronosBoltSmall": "ChronosT5Small",
    "ChronosBoltBase":  "ChronosT5Base"
}


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

def normalize_model_name(model_name: str) -> str:
    # if someone gave a Bolt model, convert it to T5
    if model_name.startswith("ChronosBolt"):
        return BOLT_TO_T5_MAP[model_name]
    return model_name

def infer_source(model_name: str) -> str:
    if model_name.startswith("Chronos"):
        return "chronos"
    elif model_name.startswith("Moirai"):
        return "moirai"
    elif model_name.startswith("TimesFM"):
        return "timesfm"
    else:
        raise ValueError(f"cannot infer source from model name: {model_name}")

def load_model(model_name: str, source: str):
    model_name = normalize_model_name(model_name)
    hf_model_id = model_name_map[model_name]

    return BaseChronosPipeline.from_pretrained(
        hf_model_id,
        device_map="auto",
        torch_dtype=torch.float32
    )


def get_model_device(model) -> torch.device:
    # chronos pipeline usually has .model (hf model) under the hood
    if hasattr(model, "model") and hasattr(model.model, "device"):
        return model.model.device
    # fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embed_feature(model, context_1d: torch.Tensor) -> np.ndarray:
    device = get_model_device(model)
    context_1d = context_1d.to(device)

    with torch.no_grad():
        hf_model = model.model  # ChronosBoltModelForForecasting

        # ChronosBolt expects [B, T]
        x = context_1d.unsqueeze(0)

        outputs = hf_model(
            x,
            output_hidden_states=True,
            return_dict=True
        )

        # last hidden state: [B, T, H]
        h = outputs.hidden_states[-1]

        # pool over time → [H]
        pooled = h.mean(dim=1).squeeze(0).detach().cpu().numpy()

    return pooled



def cell_key(exp: str, parent: str) -> str:
    return f"{exp}||{parent}"

def split_list(lst, n):
    n = max(1, int(n))
    return [lst[i::n] for i in range(n)]

def compute_embeddings_for_features(
    df: pd.DataFrame,
    feature_model_dict: dict,
    features_subset: list,
    dim: int,
    out_partial_path: str,
    verbose: bool = True
):
    df = df.copy()
    df["Parent"] = df["Parent"].astype(str)
    df["Experiment"] = df["Experiment"].astype(str)

    cell_groups = list(df.groupby(["Experiment", "Parent"]))
    model_cache = {}

    partial = {}  # feature -> { "exp||parent": [dim] }

    for feature in features_subset:
        model_name = feature_model_dict[feature]
        source = infer_source(model_name)
        key = (model_name, source)

        if verbose:
            print(f"\n[gpu worker] feature '{feature}' with model '{model_name}'")

        if key not in model_cache:
            model_cache[key] = load_model(model_name, source)
        model = model_cache[key]

        pooled_vectors = []
        valid_keys = []

        for (exp, parent), group in cell_groups:
            group_sorted = group.sort_values("TimeIndex")
            ts_values = group_sorted[feature].values.astype(np.float32)

            # skip empty / all-nan
            if ts_values.size == 0 or np.all(np.isnan(ts_values)):
                continue

            try:
                # 1d tensor (same as you had)
                context_tensor = torch.tensor(ts_values.flatten(), dtype=torch.float32)
                pooled = embed_feature(model, context_tensor)
                pooled_vectors.append(pooled)
                valid_keys.append(cell_key(exp, parent))
            except Exception:
                traceback.print_exc()

        if len(pooled_vectors) == 0:
            partial[feature] = {}
            continue

        reducer = umap.UMAP(n_components=dim, random_state=42)
        reduced = reducer.fit_transform(np.array(pooled_vectors))

        feat_map = {}
        for k, vec in zip(valid_keys, reduced):
            feat_map[k] = [float(v) for v in vec]
        partial[feature] = feat_map

        if verbose:
            print(f"[gpu worker] {feature}: pooled={np.array(pooled_vectors).shape} umap={reduced.shape}")

    os.makedirs(os.path.dirname(out_partial_path), exist_ok=True)
    with open(out_partial_path, "w") as f:
        json.dump(partial, f)
    if verbose:
        print(f"\n[gpu worker] wrote partial embeddings to {out_partial_path}")

def merge_partials_and_write_final(
    df: pd.DataFrame,
    feature_model_dict: dict,
    all_features: list,
    partial_paths: list,
    out_final_path: str,
    dim: int
):
    df = df.copy()
    df["Parent"] = df["Parent"].astype(str)
    df["Experiment"] = df["Experiment"].astype(str)

    # load partials: feature -> {cellkey: vec}
    merged_feature_maps = {feat: {} for feat in all_features}
    for p in partial_paths:
        with open(p, "r") as f:
            part = json.load(f)
        for feat, cellmap in part.items():
            merged_feature_maps[feat].update(cellmap)

    cell_groups = list(df.groupby(["Experiment", "Parent"]))

    results = []
    for (exp, parent), group in cell_groups:
        group_sorted = group.sort_values("TimeIndex")
        ck = cell_key(exp, parent)

        row = {"Experiment": exp, "Parent": parent}

        for feature in all_features:
            if feature in feature_model_dict:
                vec = merged_feature_maps.get(feature, {}).get(ck, [float("nan")] * dim)
                row[feature] = vec
            else:
                # mean fallback (your original behavior)
                try:
                    mean_val = float(pd.to_numeric(group_sorted[feature], errors="coerce").mean())
                    row[feature] = [mean_val]
                except Exception:
                    row[feature] = [float("nan")]

        results.append(row)

    os.makedirs(os.path.dirname(out_final_path), exist_ok=True)
    with open(out_final_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nmerged + saved final embeddings to {out_final_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_xlsx", default="cell_data/summary_table.xlsx")
    ap.add_argument("--model_dict", default="best_chronos_model_per_feature_jeries_4GPU.json")
    ap.add_argument("--out_final", default="embeddings/summary_table_Embedding.json")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    start = time.time()

    df = pd.read_excel(args.input_xlsx)

    with open(args.model_dict, "r") as f:
        feature_model_dict = json.load(f)

    excluded_features = ["Experiment", "Parent", "TimeIndex", "dt", "ID"]
    all_features = [c for c in df.columns if c not in excluded_features]

    # only features that have a model go to gpu workers
    modeled_features = [f for f in all_features if f in feature_model_dict]
    if len(modeled_features) == 0:
        raise RuntimeError("no features matched in model_dict json")

    n = max(1, int(args.num_gpus))
    feature_chunks = split_list(modeled_features, n)

    partial_paths = []
    procs = []

    # IMPORTANT: each worker must see only one gpu
    # simplest: set CUDA_VISIBLE_DEVICES per process
    if n == 1:
        partial_path = "embeddings/_partial_gpu0.json"
        compute_embeddings_for_features(
            df=df,
            feature_model_dict=feature_model_dict,
            features_subset=feature_chunks[0],
            dim=args.dim,
            out_partial_path=partial_path,
            verbose=args.verbose
        )
        partial_paths.append(partial_path)
    else:
        mp.set_start_method("spawn", force=True)

        for gpu_id in range(n):
            partial_path = f"embeddings/_partial_gpu{gpu_id}.json"
            partial_paths.append(partial_path)

            def _worker(gid, feats, outp, dim, verbose):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
                # now "cuda:0" inside this process == the selected gpu
                compute_embeddings_for_features(
                    df=df,
                    feature_model_dict=feature_model_dict,
                    features_subset=feats,
                    dim=dim,
                    out_partial_path=outp,
                    verbose=verbose
                )

            p = mp.Process(target=_worker, args=(gpu_id, feature_chunks[gpu_id], partial_path, args.dim, args.verbose))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    merge_partials_and_write_final(
        df=df,
        feature_model_dict=feature_model_dict,
        all_features=all_features,
        partial_paths=partial_paths,
        out_final_path=args.out_final,
        dim=args.dim
    )

    print(f"total runtime: {(time.time() - start)/60:.2f} minutes")

if __name__ == "__main__":
    main()
