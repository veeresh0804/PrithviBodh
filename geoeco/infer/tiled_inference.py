"""Tiled inference: 256px tiles, 32px overlap, Gaussian blending.

Streams arbitrarily large rasters through a torch segmentation model,
blending overlapping logits with a 2-D Gaussian kernel (no seams).
Outputs class map (argmax uint8) + confidence map (max prob 0-100 uint8).
Throughput target (PROJECT_CONTEXT §6.5): Hyderabad ~36M px in minutes/T4.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # Light CI closure has no torch; module import stays light.
    torch = None  # type: ignore[no-redef]

from geoeco.utils.config import load_yaml_config, require_keys

REPO = Path(__file__).resolve().parents[2]
AOI_CFG = REPO / "configs" / "aoi" / "hyderabad.yaml"


def _default_device() -> str:
    try:
        import torch as _torch

        return "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass(frozen=True)
class TilingConfig:
    tile: int = 256
    overlap: int = 32
    batch_size: int = 8
    device: str = field(default_factory=_default_device)


def gaussian_weight(tile: int, overlap: int) -> np.ndarray:
    """2-D Gaussian kernel peaking at tile centre, ~0 at edges."""
    ax = np.linspace(-1, 1, tile)
    xx, yy = np.meshgrid(ax, ax)
    sigma = 1.0 - overlap / tile
    w = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)
    return w / w.max()


def predict_tiled(
    image: np.ndarray,
    model: torch.nn.Module,
    config: TilingConfig | None = None,
    num_classes: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """image: (C,H,W) float32. Returns (labels HxW uint8, confidence HxW uint8)."""
    if torch is None:
        raise ImportError("torch is required for predict_tiled: pip install torch")
    cfg = config or TilingConfig()
    _, H, W = image.shape
    stride = cfg.tile - cfg.overlap
    device = torch.device(cfg.device)
    model = model.to(device).eval()
    kernel = gaussian_weight(cfg.tile, cfg.overlap)
    logit_acc = np.zeros((num_classes, H, W), dtype=np.float64)
    w_acc = np.zeros((H, W), dtype=np.float64)

    tiles: list[tuple[int, int, np.ndarray]] = []
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            r1, c1 = min(r, H - cfg.tile), min(c, W - cfg.tile)
            r1, c1 = max(r1, 0), max(c1, 0)
            tiles.append((r1, c1, image[:, r1:r1 + cfg.tile, c1:c1 + cfg.tile]))
    # Deduplicate edge-clamped tiles.
    seen, unique = set(), []
    for r1, c1, t in tiles:
        if (r1, c1) not in seen:
            seen.add((r1, c1))
            unique.append((r1, c1, t))

    with torch.no_grad():
        for i in range(0, len(unique), cfg.batch_size):
            batch = unique[i:i + cfg.batch_size]
            inp = torch.from_numpy(np.stack([t for _, _, t in batch])).to(device)
            logits = model(inp).detach().cpu().numpy()  # (B,K,h,w)
            for (r1, c1, _), lg in zip(batch, logits, strict=True):
                h, w = lg.shape[1], lg.shape[2]
                k = kernel[:h, :w]
                logit_acc[:, r1:r1 + h, c1:c1 + w] += lg * k
                w_acc[r1:r1 + h, c1:c1 + w] += k

        mean_logits = logit_acc / np.maximum(w_acc, 1e-9)
        prob = torch.softmax(torch.from_numpy(mean_logits), dim=0).numpy()
    labels = prob.argmax(axis=0).astype(np.uint8)
    confidence = (prob.max(axis=0) * 100).round().astype(np.uint8)
    return labels, confidence


def write_maps(labels: np.ndarray, confidence: np.ndarray, out_dir: Path,
               prefix: str = "lc") -> tuple[Path, Path]:
    """Save raw arrays; GeoTIFF + COG conversion handled by export pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lp, cp = out_dir / f"{prefix}_labels.npy", out_dir / f"{prefix}_confidence.npy"
    np.save(lp, labels)
    np.save(cp, confidence)
    return lp, cp


def aoi_pixel_dims(aoi: Mapping[str, Any]) -> tuple[int, int]:
    """Grid pixels (H, W) for an AOI config. Prefers size_km, else bounds."""
    res = float(aoi.get("resolution_m", 10))
    size = aoi.get("size_km")
    if size is not None:
        return (round(float(size[1]) * 1000 / res), round(float(size[0]) * 1000 / res))
    minlon, minlat, maxlon, maxlat = (float(v) for v in aoi["bounds_wgs84"])
    mean_lat = math.radians((minlat + maxlat) / 2)
    w_m = (maxlon - minlon) * 111_320.0 * math.cos(mean_lat)
    h_m = (maxlat - minlat) * 111_190.0
    return (round(h_m / res), round(w_m / res))


def estimate_tile_count(h_px: int, w_px: int, tile: int = 256,
                        overlap: int = 32) -> dict[str, int]:
    """Tile estimates for an HxW pixel grid.

    nominal_tiles: non-overlapping ceil(H/tile)*ceil(W/tile) grid — the plan
      ~550 figure for Hyderabad (6000x6000 px at 256 px => 24x24 = 576).
    strided_tiles: overlap-aware count with stride = tile - overlap.
    """
    if tile <= 0 or overlap < 0 or overlap >= tile:
        raise ValueError(f"Need 0 <= overlap < tile, got tile={tile} overlap={overlap}")
    stride = tile - overlap
    nominal = math.ceil(h_px / tile) * math.ceil(w_px / tile)
    strided = (math.ceil((h_px - tile) / stride) + 1) * (math.ceil((w_px - tile) / stride) + 1)
    return {"h_px": h_px, "w_px": w_px, "tile": tile, "overlap": overlap,
            "stride": stride, "nominal_tiles": nominal, "strided_tiles": strided}


def run_smoke() -> int:
    """Tiny synthetic overlap-blend demo (labelled synthetic, writes nothing)."""
    if torch is None:
        raise ImportError("torch is required for --smoke: pip install torch")
    rng = np.random.Generator(np.random.PCG64(42))
    image = rng.normal(0, 1, (4, 300, 300)).astype(np.float32)
    torch.manual_seed(42)
    model = torch.nn.Conv2d(4, 6, kernel_size=1)
    cfg = TilingConfig(tile=256, overlap=32, batch_size=4, device="cpu")
    labels, confidence = predict_tiled(image, model, config=cfg, num_classes=6)
    assert labels.shape == (300, 300) and labels.dtype == np.uint8
    assert confidence.shape == (300, 300) and confidence.dtype == np.uint8
    print(json.dumps({"smoke": True, "synthetic": True,
                      "note": "SMOKE demo on synthetic data — NOT a real map output",
                      "image_shape": list(image.shape), "tiles_used": 4,
                      "labels_shape": list(labels.shape),
                      "unique_labels": sorted(int(v) for v in np.unique(labels)),
                      "mean_confidence": round(float(confidence.mean()), 2)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Tiled inference plan check: validates AOI/tiling math (256-px tiles, "
                    "32-px overlap). Real inference needs --model weights + --stack raster; "
                    "without them it refuses (exit 1). --smoke runs predict_tiled on a tiny "
                    "synthetic array proving the overlap-blend path (labelled, writes nothing).")
    ap.add_argument("--aoi", default=str(AOI_CFG))
    ap.add_argument("--model", default=None, help="Model weights dir/file (required, real path).")
    ap.add_argument("--stack", default=None,
                      help="Input feature-stack raster (required, real path).")
    ap.add_argument("--out", default="data/outputs/maps/")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    try:
        aoi = load_yaml_config(args.aoi)
        require_keys(aoi, ["resolution_m"], name=args.aoi)
        if "size_km" not in aoi and "bounds_wgs84" not in aoi:
            raise KeyError(f"{args.aoi} missing required keys: ['size_km' or 'bounds_wgs84']")
        est = estimate_tile_count(*aoi_pixel_dims(aoi), tile=args.tile, overlap=args.overlap)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"aoi": args.aoi, "tile_estimate": est,
                      "plan_reference": "~550 nominal tiles for Hyderabad (plan §6.5)"}, indent=2))
    if args.smoke:
        try:
            return run_smoke()
        except ImportError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    missing = [n for n, v in (("--model", args.model), ("--stack", args.stack)) if not v]
    if missing:
        print(f"ERROR: refusing to run without {' and '.join(missing)} "
              f"(no weights/rasters here; --smoke for the synthetic demo only).",
              file=sys.stderr)
        return 1
    for label, path in (("model", args.model), ("stack", args.stack)):
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path} (fail loudly, no fake outputs).",
                  file=sys.stderr)
            return 1
    print("ERROR: weight loading + raster I/O are not wired in this CLI; "
          "refusing to write map outputs rather than faking them.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
