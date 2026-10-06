"""Download minimal public datasets for Batch 58 training.
Does NOT download full DW 5B-px set (too big for free tiers).
Gets: DW expert test sample (Zenodo 4766451) + manifest pointers for PANGAEA + DEM sample info.
S1/S2 imagery comes from Earth Engine (see configs/data/sentinel.yaml) - not direct download.

Usage:
  python scripts/download_datasets.py --out data --n_dw_test 5
  # then upload data/ to Kaggle Dataset for GPU training
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.request
from pathlib import Path

ZENODO_RECORD = "4766451"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
PANGAEA_DOI = "10.1594/PANGAEA.933475"
PANGAEA_URL = f"https://doi.pangaea.de/{PANGAEA_DOI}"

def fetch_zenodo_files() -> list[dict]:
    with urllib.request.urlopen(ZENODO_API, timeout=30) as r:
        j = json.load(r)
    return j.get("files", [])

def dl(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip exists {dest}")
        return
    print(f"GET {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_dw_test", type=int, default=5, help="how many DW test files to fetch")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "raw" / "dw_test").mkdir(parents=True, exist_ok=True)
    (out / "raw" / "pangaea").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    print(f"PANGAEA DW training (full, CC BY-4.0): {PANGAEA_URL}")
    print("Full set = ~24k tiles 510x510 (>5B px). DO NOT fetch all locally.")
    print("For pretraining use TorchGeo / EE pull per chip (see geoeco/labels/dw_tiles.py).")
    with open(out / "raw" / "pangaea" / "README.txt", "w") as f:
        f.write(f"DW training source: {PANGAEA_URL}\nUse dw_tiles.py to stream chips + pull S2/S1 from EE.\n")

    try:
        files = fetch_zenodo_files()
        print(f"Zenodo {ZENODO_RECORD}: {len(files)} files listed")
        for fi in files[: args.n_dw_test]:
            link = fi.get("links", {}).get("self", "")
            name = fi.get("key", "file.tif")
            if link:
                dl(link, out / "raw" / "dw_test" / name)
        with open(out / "raw" / "dw_test" / "_manifest.json", "w") as f:
            json.dump(files, f, indent=2)
    except Exception as e:
        print(f"Zenodo fetch failed (offline?): {e}", file=sys.stderr)
        print("Manual: https://zenodo.org/records/4766451", file=sys.stderr)

    print("\nNext:")
    print("1. EE auth: earthengine authenticate  # S1/S2/DEM/AlphaEarth/WorldCover come from EE")
    print("2. Hand labels mandatory: ~2000 Hyd points + 100 256x256 patches (QGIS) -> data/labels/")
    print("3. For free GPU: zip data/ -> upload as Kaggle Dataset -> train on Kaggle (30h T4/week).")
    print("4. Classical M1-M5 run on CPU; deep M6-M9 need T4 (Colab/Kaggle). See docs/DATA_TRAINING_GUIDE.md")

if __name__ == "__main__":
    main()
