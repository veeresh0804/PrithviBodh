# PrithviBodh — UML Diagrams

This directory contains presentation-grade UML diagrams for the **PrithviBodh** project (*AI-Based Dynamic Geo-Ecosystem Assessment Using Multi-Sensor Remote Sensing*), designed specifically for major project viva presentations and documentation.

All diagrams are derived directly from the active codebase, configurations, outputs, and documentation in this repository.

---

## Visual Design Specification
* **Background**: Clean solid white (`#FFFFFF`)
* **Primary Accents & Borders**: Imperial Purple (`#7030A0`)
* **Node & Actor Fills**: Soft Pink (`#FCE4EC`)
* **Secondary / Storage Fills**: Warm Ivory / Light Lavender (`#FFF8E1`, `#EDE7F6`)
* **Typography**: Segoe UI / Arial sans-serif, scaled for slide legibility (16:9 aspect ratios, $\ge 1920\times 1080$ resolution renders).

---

## Index of Delivered Diagrams

| ID | Diagram Type | Presentation Context | Source Files in Repository | Diagram Source | Rendered PNG |
|---|---|---|---|---|---|
| **D1** | **Use Case Diagram** | Slide 18 | `app/streamlit_app.py`, `notebooks/00–05`, `docs/labelling_protocol.md`, `data/labels.csv`, `README.md` | [`d1_use_case.mmd`](d1_use_case.mmd) | [d1_use_case.png](d1_use_case.png) |
| **D2** | **Activity Diagram** (Pipeline Flow) | Slide 19 | `prithvibodh/` (`gee_data`, `features`, `spatial_cv`, `baselines`, `classical`, `deep`, `hybrid`, `inference`, `change`, `risk`, `prediction`, `decision`) | [`d2_activity.mmd`](d2_activity.mmd) | [d2_activity.png](d2_activity.png) |
| **D3** | **Class / Module Diagram** | Model Architecture | `prithvibodh/deep.py`, `classical.py`, `hybrid.py`, `inference.py`, `change.py`, `risk.py`, `prediction.py`, `decision.py` | [`d3_class.mmd`](d3_class.mmd) | [d3_class.png](d3_class.png) |
| **D4** | **Sequence Diagram** (Web Flow) | System Interaction | `app/streamlit_app.py`, `rasterio.vrt.WarpedVRT`, `outputs/landcover_*.tif`, `folium` | [`d4_sequence.mmd`](d4_sequence.mmd) | [d4_sequence.png](d4_sequence.png) |
| **D5** | **Component & Deployment** | Slide 20 | `README.md`, `config.yaml`, `app/streamlit_app.py`, `notebooks/`, Google Earth Engine API, QGIS | [`d5_component_deployment.mmd`](d5_component_deployment.mmd) | [d5_component_deployment.png](d5_component_deployment.png) |

---

## Detailed Summary of Each Diagram

### D1. Use Case Diagram (`d1_use_case.png`)
* **Actors**: Analyst / Planner (end user), Project Team (data & ML engineers), Google Earth Engine (external cloud data service).
* **Scope**: 9 core use cases divided into end-user analytical workflows (annual maps, change detection, risk layers, 2031 prediction, priority cells) and project team workflows (QGIS labelling, multi-sensor extraction, model benchmarking, full-AOI GeoTIFF export).
* **Relations**: Models `<<extend>>` from annual maps to change detection, and `<<include>>` from annual map generation to hybrid ensemble training.

### D2. Activity Diagram (`d2_activity.png`)
* **Pipeline Flow**: Follows the real 10-step execution pipeline from multi-sensor ingestion (`gee_data.py`), 70-band seasonal feature fusion (`features.py`), 5 km spatial block cross-validation (`spatial_cv.py`), parallel baseline benchmarking ($E1$–$E5$) and hybrid model training ($H1$–$H5$), validation-weighted soft voting ($H^*$), annual inference for 2019–2025, multi-year persistence filtering (21.4% noise suppression), multi-hazard risk & CA-Markov 2031 simulation, and 2 km decision grid synthesis.

### D3. Class & Architecture Diagram (`d3_class.png`)
* **Structure**: Illustrates PyTorch neural network modules (`TempCNN`, `SeasonTransformer`, `UNet` inheriting from `nn.Module`), encapsulated by the scikit-learn-compatible `TemporalModel` wrapper. Captures the `ClassicalSuite` and the `HybridEnsemble` weighted soft-voting aggregator, which feeds into the `InferenceEngine` and downstream `GeoEcosystemAnalytics`.

### D4. Sequence Diagram (`d4_sequence.png`)
* **Scenario**: *"Analyst selects a year in the web app and views the land-cover map and change summary."*
* **Call Sequence**: Traces the actual call flow in `app/streamlit_app.py` from year selection dropdown, VRT-based reprojection and downsampling (`WarpedVRT`), class area histogram computation (`numpy.bincount`), 6-class hex RGBA overlay rendering (`folium`), and 7-year multi-sensor trajectory plotting (`pandas` / `streamlit`).

### D5. Component & Deployment Diagram (`d5_component_deployment.png`)
* **Deployment Tiers**:
  1. *Source & External Services*: QGIS Desktop (labelling), Google Earth Engine (data catalog), GitHub (version control).
  2. *Compute & Pipeline Engine*: Google Colab / Kaggle (cloud GPU runtime), Google Drive (raw 70-band stacks), `prithvibodh` core package.
  3. *Serving & Delivery*: Local `outputs/` directory, Streamlit web application, and Analyst Web Browser.

---

## Rendering Locally
All diagrams can be re-rendered using `mermaid-cli` and the provided Puppeteer configuration:
```powershell
npx -y @mermaid-js/mermaid-cli -p docs/uml/puppeteer-config.json -i docs/uml/<diagram>.mmd -o docs/uml/<diagram>.png -b white -s 3
```
Or pasted directly into [mermaid.live](https://mermaid.live).
