# Licence Register

| # | Asset | Source / DOI | Licence | What we use | Attribution / action |
|---|---|---|---|---|---|
| 1 | Sentinel-2 L2A | `COPERNICUS/S2_SR_HARMONIZED` (Copernicus/ESA) | Free & open (Copernicus data policy) | Composites, indices, maps | Credit "Copernicus Sentinel-2" in dashboard About + report |
| 2 | Sentinel-1 GRD | `COPERNICUS/S1_GRD` (Copernicus/ESA) | Free & open (Copernicus data policy) | VV/VH features, water/built-up | Credit "Copernicus Sentinel-1" in dashboard About + report |
| 3 | Cloud Score+ | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | Google EE data catalogue terms | Cloud masking only | Catalogue citation in dataset card |
| 4 | Copernicus DEM GLO-30 | Copernicus DEM | DEM licence — **verify redistribution terms before publishing derived COGs** | Elevation + slope features | Licence check logged here before M1 export |
| 5 | Dynamic World train | PANGAEA DOI 10.1594/PANGAEA.933475 | **CC BY-4.0** | Pretraining (remapped 9→6) | Attribution in dashboard footer + report + model cards |
| 6 | Dynamic World / WorldCover / Esri LC (benchmark pulls) | Google / ESA / Esri EE collections | Product-specific terms; DW CC BY-4.0 | Benchmark comparison only | Attribute per-product in comparison view + report |
| 7 | Own Hyd labels | Team field/photo-interpretation | Internal (project-owned) | Train/test/eval | Protocol + agreement in dataset card |
| 8 | Admin boundaries | Survey of India / state official | Official-use only | Stats aggregation, display | Use SoI boundaries only (NFR-10); no third-party boundary data |

**Rules:** no secrets in repo; API keys via env/GH Secrets; public map read-only; CC BY assets always attributed in UI footer + report.
