import pytest

from prithvibodh.spatial_cv import add_blocks, check_no_leakage, split_blocks


def test_blocks_never_shared(cfg, synthetic_df):
    df = split_blocks(add_blocks(synthetic_df, 5), 0.2, 0.2, 42)
    assert set(df["split"]) == {"train", "val", "test"}
    assert check_no_leakage(df)


def test_leakage_is_detected(cfg, synthetic_df):
    df = split_blocks(add_blocks(synthetic_df, 5), 0.2, 0.2, 42)
    counts = df["block"].value_counts()
    b = counts[counts >= 2].index[0]
    rows = df.index[df["block"] == b]
    df.loc[rows[0], "split"] = "train"
    df.loc[rows[1], "split"] = "test"
    with pytest.raises(AssertionError):
        check_no_leakage(df)


def test_split_is_reproducible(synthetic_df):
    a = split_blocks(add_blocks(synthetic_df, 5), 0.2, 0.2, 42)["split"]
    b = split_blocks(add_blocks(synthetic_df, 5), 0.2, 0.2, 42)["split"]
    assert (a == b).all()
