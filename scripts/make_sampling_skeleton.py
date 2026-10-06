"""Stratified sampling skeleton for QGIS (locations only, NO labels).
Fill label/label_name in QGIS per docs/labelling_protocol_v1.md, save as data/labels/hyd_points.geojson.
Blocks B0-B9 (~6km grid) for spatial CV; test blocks B0-B4, train blocks B5-B9 (disjoint).
"""
from __future__ import annotations
import json, random
from pathlib import Path
BOUNDS=[78.20,17.11,78.76,17.65]
random.seed(42)
def main(n_test=2000,n_train=1500,out="data/labels/hyd_sampling_skeleton.geojson"):
    feats=[];fid=0
    for split,n,blocks in [("test",n_test,["B0","B1","B2","B3","B4"]),("train",n_train,["B5","B6","B7","B8","B9"])]:
        for i in range(n):
            lon=random.uniform(BOUNDS[0],BOUNDS[2]);lat=random.uniform(BOUNDS[1],BOUNDS[3])
            feats.append({"type":"Feature","geometry":{"type":"Point","coordinates":[lon,lat]},
              "properties":{"fid":fid,"split":split,"label":-1,"label_name":"","block_id":blocks[i%len(blocks)],"season":"post","year":2025}})
            fid+=1
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    json.dump({"type":"FeatureCollection","features":feats},open(out,"w"),indent=1)
    print(f"wrote {out} ({len(feats)} unlabelled). Open in QGIS, fill label 0-5 + label_name, save as hyd_points.geojson.")
if __name__=="__main__": main()
