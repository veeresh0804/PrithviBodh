"""All Google Earth Engine work: build the multi-sensor stack, sample points, export.

Sensors (as listed in the project slides):
  Sentinel-2 optical, Sentinel-1 SAR, Landsat 8/9 surface temperature (LST),
  MODIS NDVI (temporal statistics) and the Copernicus 30 m DEM.
Everything runs on Google's servers, so a free Colab session is enough.
"""
import ee
import pandas as pd

from .features import SAR_BANDS, STATIC, all_columns, optical_token, sar_token, scale_factors


def init(cfg):
    try:
        ee.Initialize(project=cfg["gee_project"])
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=cfg["gee_project"])


def aoi(cfg):
    return ee.Geometry.Rectangle(cfg["aoi"]["bounds"])


def _season_filter(year, months):
    m1, m2 = months
    return ee.Filter.And(
        ee.Filter.calendarRange(year, year, "year"),
        ee.Filter.calendarRange(m1, m2, "month"),
    )


# ---------------------------------------------------------------- Sentinel-2
def _s2_collection(cfg, region, flt):
    thr = cfg["sentinel2"]["cloud_score_threshold"]
    bands = cfg["sentinel2"]["bands"]
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filter(flt)
           .linkCollection(cs, ["cs_cdf"]))
    return col.map(lambda img: img.updateMask(img.select("cs_cdf").gte(thr))
                   .select(bands).divide(10000))


def _add_indices(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return img.addBands([ndvi, ndwi, mndwi, ndbi])


def s2_composite(cfg, region, year, months=None):
    flt = (_season_filter(year, months) if months
           else ee.Filter.calendarRange(year, year, "year"))
    return _add_indices(_s2_collection(cfg, region, flt).median())


def s2_clear_count(cfg, region, year, months):
    """Number of cloud-free observations per pixel (for the cloud analysis)."""
    col = _s2_collection(cfg, region, _season_filter(year, months))
    return col.select("B4").count().rename("clear_obs")


# ---------------------------------------------------------------- Sentinel-1
def _s1_collection(cfg, region, flt):
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region).filter(flt)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("orbitProperties_pass", cfg["sentinel1"]["orbit"]))
            .select(["VV", "VH"])
            .map(lambda img: img.updateMask(img.select("VV").gt(-30))))  # border noise


def s1_composite(cfg, region, year, months):
    col = _s1_collection(cfg, region, _season_filter(year, months))
    comp = col.median().focalMedian(cfg["sentinel1"]["speckle_radius_m"], "circle", "meters")
    ratio = comp.select("VV").subtract(comp.select("VH")).rename("VV_VH")
    return comp.addBands(ratio).select(SAR_BANDS)


# ---------------------------------------------------------------- Landsat LST
def _landsat_lst(img):
    qa = img.select("QA_PIXEL")
    clear = (qa.bitwiseAnd(1 << 1).eq(0)          # dilated cloud
             .And(qa.bitwiseAnd(1 << 3).eq(0))    # cloud
             .And(qa.bitwiseAnd(1 << 4).eq(0)))   # cloud shadow
    lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
    return lst.updateMask(clear).rename("LST")


def landsat_lst(region, year, months=None):
    flt = (_season_filter(year, months) if months
           else ee.Filter.calendarRange(year, year, "year"))
    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
           .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterBounds(region).filter(flt).map(_landsat_lst))
    return col.median()


# ---------------------------------------------------------------- MODIS + DEM
def modis_ndvi_stats(region, year):
    col = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterBounds(region).filter(ee.Filter.calendarRange(year, year, "year"))
           .select("NDVI").map(lambda i: i.multiply(0.0001)))
    mean = col.mean().rename("modis_ndvi_mean")
    pct = col.reduce(ee.Reducer.percentile([10, 90]))
    amp = pct.select("NDVI_p90").subtract(pct.select("NDVI_p10")).rename("modis_ndvi_amp")
    return mean.addBands(amp).resample("bilinear")


def terrain(region):
    col = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(region).select("DEM")
    dem = col.mosaic().setDefaultProjection(col.first().projection()).rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    return dem.addBands(slope)


# ---------------------------------------------------------------- texture (GLCM)
def _glcm(img_int, band, size=3):
    """Log-contrast and entropy of a grey-level co-occurrence matrix (7x7 window)."""
    g = img_int.glcmTexture(size=size)
    contrast = g.select(f"{band}_contrast").add(1).log().rename(f"{band}_contrast")
    entropy = g.select(f"{band}_ent").rename(f"{band}_ent")
    return contrast.addBands(entropy)


def s2_texture(s2_comp):
    nir = s2_comp.select("B8").multiply(10000).toInt32()
    return _glcm(nir, "B8")


def s1_texture(s1_comp):
    vv = s1_comp.select("VV").add(30).multiply(10).max(0).toInt32()   # dB -> positive integers
    return _glcm(vv, "VV")


# ---------------------------------------------------------------- full stack
def build_stack(cfg, year):
    """Multi-sensor stack with bands in features.all_columns(cfg) order.

    Seasons with no clear optical / thermal data (common in the monsoon) are
    filled from the annual composite so that no point is dropped; the
    clear-observation count is exported separately for the cloud analysis.
    """
    region = aoi(cfg)
    s2_year = s2_composite(cfg, region, year)
    lst_year = landsat_lst(region, year).unmask(0)
    parts = []
    for season, months in cfg["seasons"].items():
        s2 = s2_composite(cfg, region, year, months).unmask(s2_year)
        s1 = s1_composite(cfg, region, year, months)
        lst = landsat_lst(region, year, months).unmask(lst_year)
        img = (s2.addBands(s2_texture(s2)).select(optical_token(cfg))
               .addBands(s1.addBands(s1_texture(s1)).select(sar_token()))
               .addBands(lst))
        names = [f"{season}_{b}" for b in optical_token(cfg) + sar_token() + ["LST"]]
        parts.append(img.rename(names))
    static = terrain(region).addBands(modis_ndvi_stats(region, year)).select(STATIC)
    return ee.Image.cat(parts + [static]).select(all_columns(cfg)).toFloat().clip(region)


# ---------------------------------------------------------------- availability
def availability_report(cfg, years=None):
    """Scenes per month for S1 (by orbit) and S2 (<20% cloud) over the AOI."""
    region, rows = aoi(cfg), []
    for y in years or cfg["years"]:
        for m in range(1, 13):
            start = ee.Date.fromYMD(y, m, 1)
            end = start.advance(1, "month")
            s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region)
                  .filterDate(start, end).filter(ee.Filter.eq("instrumentMode", "IW")))
            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(region).filterDate(start, end))
            rows.append({
                "year": y, "month": m,
                "s1_ascending": s1.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING")).size(),
                "s1_descending": s1.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")).size(),
                "s2_scenes": s2.size(),
                "s2_under_20pct_cloud": s2.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).size(),
            })
    values = ee.List([ee.Dictionary(r) for r in rows]).getInfo()
    return pd.DataFrame(values)[list(rows[0].keys())]


# ---------------------------------------------------------------- labels / sampling
def candidate_points(cfg, year=None):
    """Points to label, spread across 12 unsupervised clusters of the imagery.

    Clustering (not Dynamic World / WorldCover) keeps the sample independent
    of the products we benchmark against, while still covering rare classes.
    """
    region = aoi(cfg)
    year = year or cfg["train_year"]
    img = (s2_composite(cfg, region, year).select(["NDVI", "MNDWI", "NDBI", "B11"])
           .addBands(s1_composite(cfg, region, year, [1, 12]).select("VV")))
    training = img.sample(region=region, scale=30, numPixels=5000, seed=cfg["seed"])
    clusterer = ee.Clusterer.wekaKMeans(cfg["labels"]["candidate_clusters"]).train(training)
    clustered = img.cluster(clusterer).rename("cluster")
    pts = clustered.stratifiedSample(
        numPoints=cfg["labels"]["points_per_cluster"], classBand="cluster",
        region=region, scale=10, seed=cfg["seed"], geometries=True)
    pts = pts.map(lambda f: f.set({"lon": f.geometry().coordinates().get(0),
                                   "lat": f.geometry().coordinates().get(1)}))
    info = pts.getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in info])
    df.insert(0, "point_id", [f"P{i:05d}" for i in range(len(df))])
    return df[["point_id", "lon", "lat", "cluster"]]


def _to_fc(df):
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"point_id": r.point_id})
             for r in df.itertuples()]
    return ee.FeatureCollection(feats)


def sample_image(image, df, scale=10, batch=500):
    """Sample an image at the points in df (point_id, lon, lat) -> DataFrame."""
    out = []
    for i in range(0, len(df), batch):
        fc = _to_fc(df.iloc[i:i + batch])
        res = image.sampleRegions(collection=fc, properties=["point_id"],
                                  scale=scale, geometries=False, tileScale=4).getInfo()
        out += [f["properties"] for f in res["features"]]
    sampled = pd.DataFrame(out)
    missing = set(df.point_id) - set(sampled.get("point_id", []))
    if missing:
        print(f"Warning: {len(missing)} points returned no data (masked pixels).")
    return sampled


# ---------------------------------------------------------------- exports
def export_stack(cfg, year, scale=None):
    """Export the full stack as scaled int16 GeoTIFF (for U-Net and local map inference)."""
    img = build_stack(cfg, year)
    scaled = img.multiply(ee.Image.constant([float(v) for v in scale_factors(cfg)])).round().toInt16()
    task = ee.batch.Export.image.toDrive(
        image=scaled, description=f"stack_{year}", folder=cfg["export"]["drive_folder"],
        fileNamePrefix=f"stack_{year}", region=aoi(cfg), scale=scale or cfg["export"]["scale"],
        crs=cfg["aoi"]["crs"], maxPixels=1e10, fileFormat="GeoTIFF")
    task.start()
    return task


def export_map(image, name, cfg, scale=None, crs="EPSG:4326"):
    """Export a uint8 map (EPSG:4326 by default, easy to show in the Streamlit app)."""
    task = ee.batch.Export.image.toDrive(
        image=image.toUint8(), description=name, folder=cfg["export"]["drive_folder"],
        fileNamePrefix=name, region=aoi(cfg), scale=scale or cfg["export"]["scale"],
        crs=crs, maxPixels=1e10, fileFormat="GeoTIFF")
    task.start()
    return task


def gee_random_forest_map(cfg, year, train_df, columns):
    """Fast full-AOI map: train the same RF inside Earth Engine and classify the stack."""
    feats = [ee.Feature(None, {**{c: float(r[c]) for c in columns}, "label": int(r["label"])})
             for _, r in train_df.iterrows()]
    clf = ee.Classifier.smileRandomForest(
        numberOfTrees=cfg["models"]["rf"]["n_estimators"], seed=cfg["seed"]
    ).train(ee.FeatureCollection(feats), "label", columns)
    return build_stack(cfg, year).select(columns).classify(clf).rename("landcover")


# ---------------------------------------------------------------- indicator stack (risk inputs)
INDICATOR_BANDS = ["ndvi_dry", "ndvi_wet", "ndvi_mean", "lst_dry", "water_dry", "water_wet",
                   "elevation", "slope"]
INDICATOR_SCALE = [10000, 10000, 10000, 100, 1, 1, 1, 100]


def indicator_stack(cfg, year):
    """Small stack used for vegetation change and risk layers.

    dry = pre-monsoon, wet = monsoon. Water in the monsoon uses radar (clouds),
    in the dry season optical MNDWI or radar.
    """
    region = aoi(cfg)
    seasons = cfg["seasons"]
    dry_m, wet_m = seasons["pre_monsoon"], seasons["monsoon"]
    thr = cfg["risk"]["water_vv_db"]
    s2_year = s2_composite(cfg, region, year)
    s2_dry = s2_composite(cfg, region, year, dry_m).unmask(s2_year)
    s2_wet = s2_composite(cfg, region, year, wet_m).unmask(s2_year)
    s1_dry = s1_composite(cfg, region, year, dry_m)
    s1_wet = s1_composite(cfg, region, year, wet_m)
    lst_dry = landsat_lst(region, year, dry_m).unmask(landsat_lst(region, year).unmask(0))
    water_dry = s2_dry.select("MNDWI").gt(0).Or(s1_dry.select("VV").lt(thr)).unmask(0)
    water_wet = s1_wet.select("VV").lt(thr).unmask(0)
    img = ee.Image.cat([
        s2_dry.select("NDVI").rename("ndvi_dry"),
        s2_wet.select("NDVI").rename("ndvi_wet"),
        s2_year.select("NDVI").rename("ndvi_mean"),
        lst_dry.rename("lst_dry"),
        water_dry.rename("water_dry"),
        water_wet.rename("water_wet"),
        terrain(region),
    ]).select(INDICATOR_BANDS).toFloat().clip(region)
    return img


def export_indicator_stack(cfg, year, crs="EPSG:4326", scale=None):
    """Export on the same grid as export_map (same region, CRS and scale)."""
    img = indicator_stack(cfg, year)
    scaled = img.multiply(ee.Image.constant([float(v) for v in INDICATOR_SCALE])).round().toInt16()
    task = ee.batch.Export.image.toDrive(
        image=scaled, description=f"indicators_{year}", folder=cfg["export"]["drive_folder"],
        fileNamePrefix=f"indicators_{year}", region=aoi(cfg), scale=scale or cfg["export"]["scale"],
        crs=crs, maxPixels=1e10, fileFormat="GeoTIFF")
    task.start()
    return task
