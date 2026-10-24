# Labelling protocol (Hyderabad AOI)

**Tools (free):** QGIS + an imagery basemap (XYZ tiles: Google Satellite or Esri World Imagery), and the Sentinel-2 composite from Earth Engine for the same year. Google Earth Pro (free) is useful for its history slider.

**File:** `labels.csv` with columns `point_id, lon, lat, label, labeller, confidence`.

## Classes

| Label | Use for | Do not use for |
|---|---|---|
| `water` | Lakes, tanks, reservoirs, rivers, flooded fields at the time of the year | Dark shadows, dark rooftops |
| `tree_cover` | Forest patches, dense tree canopy, orchards with closed canopy | Scattered single trees in fields |
| `cropland` | Fields under crops or recently ploughed, paddy | Long-abandoned fields with scrub |
| `built_up` | Buildings, roads, paved areas, industrial sheds | Granite outcrops (see below) |
| `bare_rocky` | Granite boulders, rock sheets, quarries, bare soil, dry riverbeds | Ploughed fields (cropland) |
| `grass_shrub` | Scrub, grassland, open parks, sparse vegetation | Closed tree canopy |

## Hyderabad rules
1. **Granite vs rooftops:** rocks have rounded shapes, irregular shadows and no straight edges; buildings have straight edges, regular spacing and roads leading to them. When unsure, mark `confidence = unsure`.
2. **Seasonal tanks:** label the state seen for most of the year (check Google Earth history).
3. **Fallow vs bare:** if field boundaries are visible, it is `cropland`.
4. **Mixed pixel:** label what covers more than half of the 10 x 10 m pixel. If nothing does, mark `unsure`.
5. Never look at Dynamic World, WorldCover or Esri while labelling: they are what we test against.

## Quality check
* Each member labels a third; 10% of points are labelled by two members.
* Agreement must be at least 85% (`labels.agreement`). Below that, discuss the disagreements, fix the rules, relabel.
* `unsure` points are dropped automatically.
