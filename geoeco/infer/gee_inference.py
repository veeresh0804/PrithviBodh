"""Server-side inference stub for large areas (two-state scale-up).

Strategy (PROJECT_CONTEXT §6.2/§6.5): train in Python (sklearn), re-train the
same RF config inside Earth Engine with ``ee.Classifier.smileRandomForest``,
then classify server-side so ~275k km² never downloads locally.

Credentials: service-account key path comes from ``GEE_SERVICE_ACCOUNT`` /
``GOOGLE_APPLICATION_CREDENTIALS`` env vars or interactive auth — never
hard-coded (NFR-09).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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
    # Auth handled by caller (ee.Authenticate / service account); just init.
    # ee raises provider-specific auth/transport errors: log and let the
    # caller authenticate (notebooks) or fail closed (CI without creds).
    try:
        import ee as _ee_mod
        _ee_mod.Initialize()
    except Exception as e:  # noqa: BLE001 - provider-specific errors
        logger.debug("EE init deferred: %s", e)
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


def ee_credentials_present() -> bool:
    """True if EE credentials exist locally. No network, never imports ee."""
    for var in ("GEE_SERVICE_ACCOUNT", "GOOGLE_APPLICATION_CREDENTIALS"):
        candidate = os.environ.get(var, "")
        if candidate and Path(candidate).is_file():
            return True
    return (Path.home() / ".config" / "earthengine" / "credentials").is_file()


def ee_command_sequence(config: EEInferenceConfig | None = None,
                        project: str = "<GCP_PROJECT>") -> list[str]:
    """Exact server-side command sequence mirrored by this module (no faking)."""
    cfg = config or EEInferenceConfig()
    vps = (f", variablesPerSplit={cfg.variables_per_split}"
           if cfg.variables_per_split is not None else "")
    return [
        "# 1. Authenticate once (interactive) — never hard-code keys (NFR-09):",
        "earthengine authenticate  # or set GEE_SERVICE_ACCOUNT / GOOGLE_APPLICATION_CREDENTIALS",
        "from geoeco.infer.gee_inference import EEInferenceConfig, train_ee_classifier",
        "from geoeco.infer.gee_inference import classify_image",
        "import ee",
        f"ee.Initialize(project='{project}')",
        "# 2. Build the server-side feature stack (~275k km² never downloads locally):",
        "image = (stack_2019_plus_2025_mosaic)  # ee.Image with the trained input bands",
        "# 3. Mirror of sklearn M3 (500 trees) — see build_ee_classifier():",
        (f"classifier = ee.Classifier.smileRandomForest(numberOfTrees={cfg.n_trees}, "
         + f"seed={cfg.seed}, minLeafPopulation={cfg.min_leaf_population}{vps})"),
        "# 4. Train server-side on an ee.FeatureCollection of sampled points:",
        ("trained = train_ee_classifier(training_fc, input_properties, 'label', "
         + "EEInferenceConfig(...))"),
        "# 5. Classify server-side and export (tileScale avoids EE compute limits):",
        f"classified = classify_image(image, trained, tile_scale={cfg.tile_scale})",
        ("task = ee.batch.Export.image.toAsset(classified, "
         + f"'lc_hyderabad', scale=10, maxPixels=1e13, tileScale={cfg.tile_scale})"),
        "task.start()",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="GEE server-side inference: prints the exact Earth Engine command "
        + "sequence (train/classify inside EE so ~275k km² never downloads). "
        + "Needs prior `earthengine authenticate`; exits 2 without credentials "
        + "and never fakes a classification.")
    ap.add_argument("--n-trees", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-leaf-population", type=int, default=1)
    ap.add_argument("--variables-per-split", type=int, default=None)
    ap.add_argument("--tile-scale", type=int, default=4)
    ap.add_argument("--project", default="<GCP_PROJECT>")
    args = ap.parse_args(argv)
    if not ee_credentials_present():
        print("ERROR: no Earth Engine credentials found. Authenticate first:\n"
              "  earthengine authenticate\n"
              "or set GEE_SERVICE_ACCOUNT / GOOGLE_APPLICATION_CREDENTIALS to a "
              "service-account key file, then re-run. Refusing to fake EE output.",
              file=sys.stderr)
        return 2
    cfg = EEInferenceConfig(n_trees=args.n_trees, variables_per_split=args.variables_per_split,
                            min_leaf_population=args.min_leaf_population, seed=args.seed,
                            tile_scale=args.tile_scale)
    print("\n".join(ee_command_sequence(cfg, project=args.project)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
