"""Server-side inference stub for large areas (two-state scale-up).

Strategy (PROJECT_CONTEXT §6.2/§6.5): train in Python (sklearn), re-train the
same RF config inside Earth Engine with ``ee.Classifier.smileRandomForest``,
then classify server-side so ~275k km² never downloads locally.

Credentials: service-account key path comes from ``GEE_SERVICE_ACCOUNT`` /
``GOOGLE_APPLICATION_CREDENTIALS`` env vars or interactive auth — never
hard-coded (NFR-09).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EEInferenceConfig:
    n_trees: int = 500
    variables_per_split: int | None = None  # -> sqrt(n_features) server-side
    min_leaf_population: int = 1
    seed: int = 42
    tile_scale: int = 4


def _ee():
    try:
        import ee  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("earthengine-api required: pip install earthengine-api") from e
    key = os.environ.get("GEE_SERVICE_ACCOUNT") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    # Auth handled by caller (ee.Authenticate / service account); just init.
    try:
        import ee as _ee_mod
        _ee_mod.Initialize()
    except Exception:
        pass  # caller authenticates in notebook/CI
    return ee


def build_ee_classifier(config: EEInferenceConfig | None = None):
    """ee.Classifier.smileRandomForest mirror of sklearn M3 (500 trees)."""
    ee = _ee()
    cfg = config or EEInferenceConfig()
    kwargs: dict = {"numberOfTrees": cfg.n_trees, "seed": cfg.seed,
                    "minLeafPopulation": cfg.min_leaf_population}
    if cfg.variables_per_split is not None:
        kwargs["variablesPerSplit"] = cfg.variables_per_split
    return ee.Classifier.smileRandomForest(**kwargs)


def train_ee_classifier(training_fc, input_properties: list[str], label_property: str = "label",
                        config: EEInferenceConfig | None = None):
    """Train server-side RF on an ee.FeatureCollection of sampled points."""
    clf = build_ee_classifier(config)
    return clf.train(training_fc, label_property, input_properties)


def classify_image(image, classifier, tile_scale: int = 4):
    """Classify an ee.Image stack server-side; returns classified ee.Image."""
    return image.classify(classifier).set({"tile_scale": tile_scale})
