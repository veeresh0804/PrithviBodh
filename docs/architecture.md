# Architecture

The project is built in two steps, both evaluated on the **same spatially held-out test points**
labelled by the team.

## Step 1 - Existing solutions (what we compare against)

```mermaid
flowchart LR
  S2["Sentinel-2 only"] --> DW["E1 Dynamic World<br/>(Google, deep learning)"]
  S12["Sentinel-1 + Sentinel-2"] --> WC["E2 ESA WorldCover 2021"]
  S2b["Sentinel-2 only"] --> ES["E3 Esri Land Cover"]
  S2c["Sentinel-2 bands + indices"] --> RFO["E4 Random Forest<br/>optical only"]
  S1["Sentinel-1 VV/VH"] --> RFS["E5 Random Forest<br/>SAR only"]
  DW --> T["Score on our test points<br/>OA, macro-F1, kappa, 95% CI"]
  WC --> T
  ES --> T
  RFO --> T
  RFS --> T
```

Limits of the existing solutions:
* Global products are trained for the whole world, not tuned or checked for the Deccan landscape.
* E1, E3 and E4 use optical data only, so monsoon clouds hurt them.
* Each uses one model type.

## Step 2 - Hybrid architecture (our contribution)

"Hybrid" = **all sensors from the project slides x all models from the slides**, combined.

```mermaid
flowchart TB
  subgraph DATA["1. Multi-sensor data (Google Earth Engine, free)"]
    S2["Sentinel-2<br/>10 bands + NDVI, NDWI, MNDWI, NDBI<br/>+ NIR texture"]
    S1["Sentinel-1<br/>VV, VH, VV-VH<br/>+ VV texture"]
    LS["Landsat 8/9<br/>surface temperature (LST)"]
    MO["MODIS<br/>NDVI mean + amplitude"]
    DEM["Copernicus DEM<br/>elevation, slope"]
  end
  subgraph FUSE["2. Fusion"]
    ST["Seasonal stack, 10 m<br/>pre-monsoon, monsoon, post-monsoon<br/>70 features per pixel"]
  end
  subgraph MODELS["3. Models (all from the slides)"]
    RF["Random Forest"]
    SVM["SVM"]
    CNN["CNN<br/>(TempCNN over seasons)"]
    TR["Transformer<br/>(attention over seasons)"]
    UN["U-Net<br/>(spatial, sparse labels)"]
  end
  ENS["4. Hybrid ensemble<br/>weighted soft voting<br/>weights = validation macro-F1"]
  OUT["5. Outputs<br/>land-cover maps 2019 + 2025, change map,<br/>NDVI change, indicators, comparison with existing"]
  RISK["6. Risk assessment<br/>flood, drought, deforestation"]
  PRED["7. Prediction<br/>CA-Markov land cover 2031"]
  DSS["8. Decision support<br/>ranked 2 km cells + suggested actions"]
  S2 --> ST
  S1 --> ST
  LS --> ST
  MO --> ST
  DEM --> ST
  ST --> RF
  ST --> SVM
  ST --> CNN
  ST --> TR
  ST --> UN
  RF --> ENS
  SVM --> ENS
  CNN --> ENS
  TR --> ENS
  UN --> ENS
  ENS --> OUT
  OUT --> RISK
  OUT --> PRED
  RISK --> DSS
  OUT --> DSS
```

### Why each part is there
| Part | Reason |
|---|---|
| Sentinel-1 | Sees through monsoon cloud; separates water and buildings by surface roughness |
| Sentinel-2 | Main spectral information; vegetation, water and built-up indices |
| Landsat LST | Built-up areas and bare rock are hotter than vegetation and water |
| MODIS NDVI stats | Year-round vegetation rhythm (crops vs trees vs scrub) |
| DEM | Terrain context (hills, valleys, tank beds) |
| Three seasons | Crops, tanks and grass change through the year; forests and buildings do not |
| RF, SVM | Strong classical models on tabular features |
| CNN, Transformer | Learn the seasonal pattern of each pixel |
| U-Net | Uses the neighbourhood (shape, texture), not only the pixel |
| Weighted ensemble | Models make different mistakes; voting cancels some of them |
| GLCM texture | Separates rough surfaces (buildings, rocks) from smooth ones (water, fields) |
| Risk indices | Flood: monsoon-only radar water nearby + low, flat ground. Drought: 1 - Vegetation Health Index (dry-season NDVI and LST). Deforestation: remaining trees near recent tree loss or new buildings |
| CA-Markov | 2019-2025 transition rates, placed where the new class is already nearby, predicts 2031 |
| Decision support | 2 km cells ranked by a transparent priority score, each with its main issue and a suggested action |

## Validation (same for both steps)
* Hand-labelled points, never taken from any global product.
* 5 km spatial blocks; whole blocks go to train, validation or test, so nearby pixels can't leak.
* Ensemble weights come from validation blocks only; the test blocks are touched once, at the end.
* Report OA, precision, recall, macro-F1, mIoU, kappa, per-class precision/recall/F1/IoU, 95% bootstrap CI,
  and a paired CI for "hybrid minus best existing".
* Risk indices and the 2031 prediction are screening tools, not validated forecasts: with only two dates the
  prediction cannot be back-tested.
* A random point split is also run once, only to show how much it inflates accuracy.

## Free tools only
| Need | Tool | Cost |
|---|---|---|
| Satellite data + processing | Google Earth Engine (noncommercial) | Free |
| Coding + GPU | Google Colab or Kaggle | Free |
| Storage | Google Drive (15 GB) | Free |
| Code | GitHub | Free |
| Labelling | QGIS | Free |
| Demo | Streamlit (local, or Streamlit Community Cloud) | Free |
