"""Load, clean and check hand-made labels.

Expected CSV columns: point_id, lon, lat, label, labeller, confidence
  label      = one of the class names in config.yaml
  confidence = 'sure' or 'unsure' (unsure points are dropped)
"""
import numpy as np
import pandas as pd

REQUIRED = ["point_id", "lon", "lat", "label"]


def load_labels(path, cfg, drop_unsure=True):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Labels file is missing columns: {missing}")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    classes = cfg["classes"]
    bad = sorted(set(df["label"]) - set(classes))
    if bad:
        raise ValueError(f"Unknown label names {bad}. Allowed: {classes}")
    if drop_unsure and "confidence" in df.columns:
        df = df[df["confidence"].astype(str).str.lower() != "unsure"]
    if df["point_id"].duplicated().any():
        raise ValueError("Duplicate point_id values in labels file")
    df = df.copy()
    df["label"] = df["label"].map({c: i for i, c in enumerate(classes)}).astype(int)
    return df.reset_index(drop=True)


def agreement(df_a, df_b):
    """Percent agreement and Cohen's kappa between two labellers on shared points."""
    from sklearn.metrics import cohen_kappa_score
    m = df_a[["point_id", "label"]].merge(df_b[["point_id", "label"]], on="point_id",
                                          suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("The two label sets share no point_id")
    pct = float(np.mean(m["label_a"] == m["label_b"]))
    kappa = float(cohen_kappa_score(m["label_a"], m["label_b"]))
    return {"n_shared": len(m), "percent_agreement": pct, "kappa": kappa}


def class_counts(df, cfg):
    counts = df["label"].value_counts().reindex(range(len(cfg["classes"])), fill_value=0)
    counts.index = cfg["classes"]
    return counts
