"""Full-map hybrid prediction over the exported stack (runs in Colab).

Reads the int16 stack window by window, so it fits in free Colab memory.
SVM is left out of map inference by default: its RBF kernel is too slow for
~36 million pixels. It is still scored at the test points.
"""
import numpy as np

from . import hybrid
from .features import scale_factors


def predict_map(stack_path, out_path, cfg, members: dict, weights: dict, unet=None, tile=512):
    """members: name -> fitted model with predict_proba(X_all); unet: fitted UNetModel or None.

    Writes a 2-band uint8 GeoTIFF: band 1 = class (255 = no data), band 2 = confidence 0-100.
    """
    import rasterio
    from rasterio.windows import Window

    scales = scale_factors(cfg)
    use = {n: m for n, m in members.items() if weights.get(n, 0) > 0}
    w = {n: weights[n] for n in use}
    if unet is not None and weights.get("H5 U-Net", 0) > 0:
        w["H5 U-Net"] = weights["H5 U-Net"]
    pad = 32
    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        profile.update(count=2, dtype="uint8", nodata=255, compress="deflate", tiled=True,
                       blockxsize=256, blockysize=256)
        with rasterio.open(out_path, "w", **profile) as dst:
            for r0 in range(0, src.height, tile):
                for c0 in range(0, src.width, tile):
                    h, wd = min(tile, src.height - r0), min(tile, src.width - c0)
                    arr = src.read(window=Window(c0, r0, wd, h)).astype("float32")
                    valid = np.any(arr != 0, axis=0)
                    X = (arr.reshape(arr.shape[0], -1).T / scales)
                    probs = {n: m.predict_proba(X).reshape(h, wd, -1) for n, m in use.items()}
                    if "H5 U-Net" in w:
                        big = src.read(window=Window(c0 - pad, r0 - pad, wd + 2 * pad, h + 2 * pad),
                                       boundless=True, fill_value=0).astype("float32")
                        big = big / scales[:, None, None]
                        H = int(np.ceil(big.shape[1] / 4) * 4)
                        W = int(np.ceil(big.shape[2] / 4) * 4)
                        big = np.pad(big, ((0, 0), (0, H - big.shape[1]), (0, W - big.shape[2])))
                        p = unet.predict_proba_chips(big[None])[0]          # (classes, H, W)
                        probs["H5 U-Net"] = p[:, pad:pad + h, pad:pad + wd].transpose(1, 2, 0)
                    p = hybrid.combine(probs, w)
                    cls = p.argmax(-1).astype("uint8")
                    conf = (p.max(-1) * 100).round().astype("uint8")
                    cls[~valid], conf[~valid] = 255, 0
                    dst.write(np.stack([cls, conf]), window=Window(c0, r0, wd, h))
    return out_path
