"""Small raster helpers (read, write on the same grid, match grids)."""
import numpy as np


def read(path, band=1):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(band), src.profile.copy()


def read_bands(path, names, all_names, scales):
    """Read selected bands of a scaled int16 stack into a dict of float arrays."""
    import rasterio
    out = {}
    with rasterio.open(path) as src:
        for n in names:
            i = all_names.index(n)
            arr = src.read(i + 1).astype("float32") / float(scales[i])
            out[n] = arr
    return out


def write(path, arr, profile, dtype="uint8", nodata=255):
    import rasterio
    p = profile.copy()
    p.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.asarray(arr).astype(dtype), 1)
    return path


def match_grid(src_path, ref_path, out_path, categorical=True):
    """Reproject src onto ref's grid (nearest for class maps, bilinear otherwise)."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        prof = ref.profile.copy()
        prof.update(count=src.count, dtype=src.dtypes[0], nodata=src.nodata)
        with rasterio.open(out_path, "w", **prof) as dst:
            for b in range(1, src.count + 1):
                reproject(rasterio.band(src, b), rasterio.band(dst, b),
                          resampling=Resampling.nearest if categorical else Resampling.bilinear)
    return out_path


def to_uint8_score(score):
    """0-1 float score (NaN = not applicable) -> 0-100 uint8, 255 = no data."""
    out = np.full(score.shape, 255, dtype=np.uint8)
    ok = np.isfinite(score)
    out[ok] = np.clip(np.round(score[ok] * 100), 0, 100).astype(np.uint8)
    return out
