"""Spatial blocks: train, validation and test never share a block (no spatial leakage)."""
import numpy as np


def add_blocks(df, block_km):
    """Assign each point to a square block of block_km x block_km."""
    lat0 = np.deg2rad(df["lat"].mean())
    x_km = df["lon"].to_numpy() * 111.320 * np.cos(lat0)
    y_km = df["lat"].to_numpy() * 110.574
    bx = np.floor(x_km / block_km).astype(int)
    by = np.floor(y_km / block_km).astype(int)
    out = df.copy()
    out["block"] = [f"{a}_{b}" for a, b in zip(bx, by)]
    return out


def split_blocks(df, test_fraction, val_fraction, seed):
    """Label each point 'train', 'val' or 'test' by randomly assigning whole blocks."""
    rng = np.random.default_rng(seed)
    blocks = np.array(sorted(df["block"].unique()))
    rng.shuffle(blocks)
    n_test = max(1, int(round(len(blocks) * test_fraction)))
    n_val = max(1, int(round(len(blocks) * val_fraction)))
    test_b = set(blocks[:n_test])
    val_b = set(blocks[n_test:n_test + n_val])
    out = df.copy()
    out["split"] = np.where(out["block"].isin(test_b), "test",
                            np.where(out["block"].isin(val_b), "val", "train"))
    check_no_leakage(out)
    return out


def random_split(df, test_fraction, val_fraction, seed):
    """Point-level random split, ONLY to show how much it inflates accuracy."""
    rng = np.random.default_rng(seed)
    r = rng.random(len(df))
    out = df.copy()
    out["split"] = np.where(r < test_fraction, "test",
                            np.where(r < test_fraction + val_fraction, "val", "train"))
    return out


def check_no_leakage(df):
    groups = df.groupby("block")["split"].nunique()
    leaking = groups[groups > 1]
    if len(leaking):
        raise AssertionError(f"Blocks used in more than one split: {list(leaking.index)[:5]}")
    return True
