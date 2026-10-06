"""Make synthetic hyd_points.geojson for pipeline smoke ONLY (not for report).
Real file needs QGIS photo-interp per docs/labelling_protocol_v1.md.
Writes data/labels/hyd_points_SYNTHETIC.geojson with required schema:
lon,lat + properties{label, label_name, block_id, season, year, synthetic:true}
"""
from __future__ import annotations
import json, random
from pathlib import Path
CLASSES = ["water","tree_cover","cropland","built_up","bare_rocky","grass_shrub"]
BOUNDS = [78.20,17.11,78.76,17.65]
random.seed(42)
def main(n: int = 120, out: str = "data/labels/hyd_points_SYNTHETIC.geojson"):
    feats=[]
    for i in range(n):
        ci=i%6
        lon=random.uniform(BOUNDS[0],BOUNDS[2]); lat=random.uniform(BOUNDS[1],BOUNDS[3])
        feats.append({"type":"Feature","geometry":{"type":"Point","coordinates":[lon,lat]},
          "properties":{"label":ci,"label_name":CLASSES[ci],"block_id":f"B{(i%10)}","season":"post","year":2025,"synthetic":True}})
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    json.dump({"type":"FeatureCollection","features":feats},open(out,"w"),indent=2)
    print(f"wrote {out} ({n} synthetic pts). Replace with real QGIS data/labels/hyd_points.geojson for M3.")
if __name__=="__main__":
    import sys; main(int(sys.argv[1]) if len(sys.argv)>1 else 120)
