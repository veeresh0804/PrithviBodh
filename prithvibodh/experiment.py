"""Run the two-step study on the same spatially held-out test points.

Step 1 - existing solutions:
    E1 Dynamic World, E2 ESA WorldCover, E3 Esri Land Cover  (global products)
    E4 Random Forest, optical only (traditional single-sensor approach)
    E5 Random Forest, SAR only
Step 2 - hybrid (all sensors x all models, weighted soft voting):
    H1 RF, H2 SVM, H3 TempCNN, H4 Transformer, H5 U-Net (optional), H* Hybrid ensemble
"""
import numpy as np

from . import classical, evaluate, hybrid
from .features import all_columns, feature_groups


def _xy(df, split, cols):
    d = df[df["split"] == split]
    return d[cols].to_numpy(dtype="float32"), d["label"].to_numpy()


def _col_idx(cfg, cols):
    pos = {c: i for i, c in enumerate(all_columns(cfg))}
    return [pos[c] for c in cols]


def run_existing(df, cfg):
    """df needs: label, split, all feature columns, dynamic_world, worldcover, esri."""
    k, rows, preds = len(cfg["classes"]), [], {}
    test = df[df["split"] == "test"]
    for code, name in [("E1", "dynamic_world"), ("E2", "worldcover"), ("E3", "esri")]:
        if name not in test:
            continue
        ok = test[name] < k
        y, p = test.loc[ok, "label"].to_numpy(), test.loc[ok, name].to_numpy().astype(int)
        note = f"{(~ok).sum()} test points had no matching class" if (~ok).any() else ""
        rows.append(evaluate.result_row(f"{code} {name}", "existing", y, p, k, note))
        preds[code] = (test.index[ok], p)

    cols = all_columns(cfg)
    X_tr, y_tr = _xy(df, "train", cols)
    X_va, y_va = _xy(df, "val", cols)
    X_te, y_te = _xy(df, "test", cols)
    X_fit, y_fit = np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va])
    groups = feature_groups(cfg)
    for code, g in [("E4", "optical"), ("E5", "sar")]:
        m = classical.ColumnModel(classical.make_rf(cfg), _col_idx(cfg, groups[g]), k)
        p = m.fit(X_fit, y_fit).predict_proba(X_te).argmax(1)
        rows.append(evaluate.result_row(f"{code} RF {g}-only", "existing", y_te, p, k))
        preds[code] = (df.index[df["split"] == "test"], p)
    return rows, preds


def hybrid_members(cfg, include_deep=True):
    k = len(cfg["classes"])
    idx = _col_idx(cfg, all_columns(cfg))
    members = {
        "H1 RF": lambda: classical.ColumnModel(classical.make_rf(cfg), idx, k),
        "H2 SVM": lambda: classical.ColumnModel(classical.make_svm(cfg), idx, k),
    }
    if include_deep:
        from .deep import TemporalModel
        members["H3 TempCNN"] = lambda: TemporalModel("cnn", cfg, k)
        members["H4 Transformer"] = lambda: TemporalModel("transformer", cfg, k)
    return members


def run_hybrid(df, cfg, include_deep=True, unet_val_probs=None, unet_test_probs=None):
    """Train every member on 'train', weight on 'val', refit on train+val, score on 'test'.

    unet_*_probs: optional (n, classes) U-Net probabilities for the val/test rows
    (same row order as df[df.split == 'val'/'test']), produced in notebook 03.
    """
    k, cols = len(cfg["classes"]), all_columns(cfg)
    X_tr, y_tr = _xy(df, "train", cols)
    X_va, y_va = _xy(df, "val", cols)
    X_te, y_te = _xy(df, "test", cols)
    X_fit, y_fit = np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va])

    val_f1, test_probs, rows = {}, {}, []
    for name, make in hybrid_members(cfg, include_deep).items():
        p_val = make().fit(X_tr, y_tr).predict_proba(X_va)
        val_f1[name] = evaluate.scores(y_va, p_val.argmax(1), k)["macro_F1"]
        test_probs[name] = make().fit(X_fit, y_fit).predict_proba(X_te)
        rows.append(evaluate.result_row(name, "hybrid member", y_te, test_probs[name].argmax(1), k,
                                        f"val macro-F1 {val_f1[name]:.3f}"))
    if unet_val_probs is not None and unet_test_probs is not None:
        val_f1["H5 U-Net"] = evaluate.scores(y_va, unet_val_probs.argmax(1), k)["macro_F1"]
        test_probs["H5 U-Net"] = unet_test_probs
        rows.append(evaluate.result_row("H5 U-Net", "hybrid member", y_te, unet_test_probs.argmax(1), k,
                                        f"val macro-F1 {val_f1['H5 U-Net']:.3f}"))

    weights = hybrid.normalise_weights(val_f1)
    p_h = hybrid.combine(test_probs, weights).argmax(1)
    note = ", ".join(f"{n.split()[0]}={w:.2f}" for n, w in weights.items())
    rows.append(evaluate.result_row("H* HYBRID ensemble", "hybrid", y_te, p_h, k, f"weights {note}"))
    return rows, {"weights": weights, "test_probs": test_probs, "hybrid_pred": p_h, "y_test": y_te}
