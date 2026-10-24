"""Existing solutions: free global 10 m land-cover products, remapped to our 6 classes.

Our classes (index): 0 water, 1 tree_cover, 2 cropland, 3 built_up, 4 bare_rocky, 5 grass_shrub
255 = no matching class / no data (excluded from that product's scores).
"""
NODATA = 255

# Dynamic World 'label': 0 water, 1 trees, 2 grass, 3 flooded_vegetation, 4 crops,
# 5 shrub_and_scrub, 6 built, 7 bare, 8 snow_and_ice
DYNAMIC_WORLD = {0: 0, 1: 1, 2: 5, 3: 0, 4: 2, 5: 5, 6: 3, 7: 4, 8: 4}

# ESA WorldCover v200 'Map'
WORLDCOVER = {10: 1, 20: 5, 30: 5, 40: 2, 50: 3, 60: 4, 70: 4, 80: 0, 90: 0, 95: 1, 100: 5}

# Esri 10 m Land Cover time series 'b1' (10 = clouds -> no data)
ESRI = {1: 0, 2: 1, 4: 0, 5: 2, 7: 3, 8: 4, 9: 4, 11: 5}

CROSSWALKS = {"dynamic_world": DYNAMIC_WORLD, "worldcover": WORLDCOVER, "esri": ESRI}


def remap_value(value, crosswalk):
    """Pure-python remap (used for tests and for sampled tables)."""
    try:
        return crosswalk.get(int(value), NODATA)
    except (TypeError, ValueError):
        return NODATA


# ---------------------------------------------------------------- Earth Engine
def _remap(img, crosswalk):
    return img.remap(list(crosswalk.keys()), list(crosswalk.values()), NODATA)


def dynamic_world(cfg, region, year):
    import ee
    col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
           .filter(ee.Filter.calendarRange(year, year, "year")).select("label"))
    return _remap(col.mode(), DYNAMIC_WORLD).rename("dynamic_world")


def worldcover(cfg, region, year=None):
    import ee
    img = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")  # 2021 only
    return _remap(img, WORLDCOVER).rename("worldcover")


def esri(cfg, region, year):
    import ee
    y = min(year, cfg["esri_last_year"])
    col = (ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
           .filterBounds(region).filter(ee.Filter.calendarRange(y, y, "year")))
    return _remap(col.mosaic().select("b1"), ESRI).rename("esri")


def all_products(cfg, region, year):
    import ee
    return ee.Image.cat([dynamic_world(cfg, region, year),
                         worldcover(cfg, region, year),
                         esri(cfg, region, year)])
