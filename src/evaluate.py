"""Ranking metrics for the "Buy it again" row."""
import numpy as np
import pandas as pd

def evaluate(df, rank_col, ks=(5,10)):
    """Average recall@k and precision@k over users with at least one reordered product.
    df: one row per (user, product) with columns user_id, label and rank_col
    rank_col: name of the rank column (1=shown first)
    ks: row sizes to evaluate
    """
    data = df[["user_id","label", rank_col]].copy()
    for k in ks:
        data[f"hit_at_{k}"] = data["label"]*(data[rank_col]<=k)

    per_user = data.groupby("user_id").agg(
        positives = ("label", "sum"),
        candidates = ("label","size"),
        **{f"hits_at_{k}" : (f"hit_at_{k}", "sum") for k in ks}
    )

    per_user = per_user[per_user["positives"]>0]

    results = {"Users": len(per_user)}
    for k in ks:
        results[f"recall_at_{k}"] = (per_user[f"hits_at_{k}"]/per_user["positives"]).mean()
        results[f"precision_at_{k}"] = (
            per_user[f"hits_at_{k}"]/np.minimum(k, per_user["candidates"])
        ).mean()

    return pd.DataFrame([results])

def per_user_recall(df, rank_col, k=10):
    """Recall@k for each user (users with no reordered products are dropped)."""
    d = df[["user_id", "label", rank_col]].copy()
    d["hit"] = d["label"] * (d[rank_col]<=k)
    g = d.groupby("user_id").agg(pos=("label", "sum"), hits=("hit", "sum"))
    g = g[g["pos"]>0]
    return g["hits"]/g["pos"]