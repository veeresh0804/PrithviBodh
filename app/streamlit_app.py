"""PrithviBodh demo app (free: run locally or on Streamlit Community Cloud).

Put the exported files in outputs/:
  landcover_2019.tif, landcover_2025.tif   (band 1 = class, optional band 2 = confidence)
  dynamic_world_2025.tif                    (optional, for comparison)
  results.csv                               (comparison table from notebook 03)
  change_*.tif, ndvi_change_*.tif, indicators.csv, transition_matrix.csv   (notebook 04)
  risk_*.tif, landcover_<year>_predicted.tif, projection_*.csv, priority_cells.csv (notebook 05)
Run:  streamlit run app/streamlit_app.py
"""
import io
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from streamlit_folium import st_folium

OUT = Path(__file__).resolve().parents[1] / "outputs"
CLASSES = ["water", "tree_cover", "cropland", "built_up", "bare_rocky", "grass_shrub"]
COLOURS = ["#2b83ba", "#1a9641", "#f4d03f", "#d7191c", "#a6a6a6", "#a6d96a"]
MAX_PX = 1500

st.set_page_config(page_title="PrithviBodh", layout="wide")


@st.cache_data
def load(path):
    with rasterio.open(path) as src, WarpedVRT(src, crs="EPSG:4326") as vrt:
        f = max(vrt.width, vrt.height) / MAX_PX
        shape = (max(1, int(vrt.height / max(f, 1))), max(1, int(vrt.width / max(f, 1))))
        arr = vrt.read(1, out_shape=shape, resampling=Resampling.nearest)
        b = vrt.bounds
        px_ha = abs(src.res[0] * src.res[1]) / 10000 if src.crs.is_projected else 0.01
        full = src.read(1)
    return arr, [[b.bottom, b.left], [b.top, b.right]], full, px_ha


def to_rgba(arr):
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    for i, hexc in enumerate(COLOURS):
        rgb = [int(hexc[j:j + 2], 16) for j in (1, 3, 5)]
        rgba[arr == i] = rgb + [210]
    return rgba


def ramp(arr, stops, lo, hi):
    """Continuous colours for values lo..hi; 255 = transparent."""
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    ok = arr != 255
    t = np.clip((arr.astype(float) - lo) / (hi - lo), 0, 1)
    cols = np.array([[int(c[j:j + 2], 16) for j in (1, 3, 5)] for c in stops], dtype=float)
    pos = t * (len(stops) - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    f = (pos - i0)[..., None]
    rgb = cols[i0] * (1 - f) + cols[i0 + 1] * f
    rgba[..., :3][ok] = rgb[ok].astype(np.uint8)
    rgba[..., 3][ok] = 210
    return rgba


def png(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


def overlay_map(rgba, bounds, key, name="layer"):
    import base64
    centre = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    m = folium.Map(location=centre, zoom_start=10, tiles="OpenStreetMap")
    folium.raster_layers.ImageOverlay(image="data:image/png;base64," + base64.b64encode(png(rgba)).decode(),
                                      bounds=bounds, opacity=0.8, name=name).add_to(m)
    st_folium(m, height=480, use_container_width=True, key=key)


def show_map(path, key):
    arr, bounds, _, _ = load(path)
    overlay_map(to_rgba(arr), bounds, key, path.stem)


def areas(path):
    _, _, full, px_ha = load(path)
    v = full[full < len(CLASSES)]
    return pd.Series(np.bincount(v, minlength=len(CLASSES)) * px_ha, index=CLASSES)


st.title("PrithviBodh - Hybrid multi-sensor land cover, Hyderabad")
st.caption("Sentinel-1 + Sentinel-2 + Landsat LST + MODIS + DEM, fused with RF, SVM, CNN, "
           "Transformer and U-Net. Validated on spatially independent hand-labelled points.")
st.markdown(" ".join(f"<span style='background:{c};padding:2px 8px;border-radius:4px;"
                     f"color:white;margin-right:4px'>{n}</span>" for n, c in zip(CLASSES, COLOURS)),
            unsafe_allow_html=True)

maps = sorted(p for p in OUT.glob("landcover_*.tif") if "predicted" not in p.stem)
tabs = st.tabs(["Map", "Before / after", "Change map", "Compare with existing",
                "Risk", "Future (prediction)", "Decision support", "Results"])
tab_map, tab_ba, tab_change, tab_cmp, tab_risk, tab_future, tab_dss, tab_res = tabs

with tab_map:
    if maps:
        choice = st.selectbox("Year", maps, format_func=lambda p: p.stem.split("_")[-1])
        show_map(choice, "single")
        st.bar_chart(areas(choice).rename("area (ha)"))
    else:
        st.info("No maps yet. Run notebook 04 and copy landcover_<year>.tif into outputs/.")

with tab_ba:
    if len(maps) >= 2:
        a, b = st.columns(2)
        with a:
            st.subheader(maps[0].stem.split("_")[-1])
            show_map(maps[0], "before")
        with b:
            st.subheader(maps[-1].stem.split("_")[-1])
            show_map(maps[-1], "after")
        st.bar_chart((areas(maps[-1]) - areas(maps[0])).rename("change (ha)"))
        tm = OUT / "transition_matrix.csv"
        if tm.exists():
            st.write("Transition matrix (hectares, rows = from, columns = to)")
            st.dataframe(pd.read_csv(tm, index_col=0).round(1))
    else:
        st.info("Needs two years of maps in outputs/.")

with tab_change:
    if len(maps) >= 2:
        a0, bounds, _, _ = load(maps[0])
        a1, _, _, _ = load(maps[-1])
        changed = (a0 != a1) & (a0 < len(CLASSES)) & (a1 < len(CLASSES))
        show = np.where(changed, a1, 255).astype(np.uint8)
        st.caption("Coloured pixels changed between the two years; the colour is the NEW class.")
        overlay_map(to_rgba(show), bounds, "changemap", "change")
        ndvi = sorted(OUT.glob("ndvi_change_*.tif"))
        if ndvi:
            st.subheader("Vegetation condition change (NDVI)")
            arr, bnd, _, _ = load(ndvi[-1])
            st.caption("Brown = less green, green = greener (vegetated pixels only).")
            overlay_map(ramp(arr, ["#8c510a", "#f5f5f5", "#1a9850"], 70, 130), bnd, "ndvi", "ndvi")
        ind = OUT / "indicators.csv"
        if ind.exists():
            st.dataframe(pd.read_csv(ind, index_col=0).rename(columns={"0": "value"}).round(2))
    else:
        st.info("Needs two years of maps in outputs/.")

with tab_cmp:
    dw = sorted(OUT.glob("dynamic_world_*.tif"))
    if maps and dw:
        a, b = st.columns(2)
        with a:
            st.subheader("Our hybrid map")
            show_map(maps[-1], "ours")
        with b:
            st.subheader("Dynamic World (existing)")
            show_map(dw[-1], "dw")
    else:
        st.info("Export dynamic_world_<year>.tif in notebook 04 to compare here.")

with tab_risk:
    layers = {p.stem.replace("risk_", "").title(): p for p in sorted(OUT.glob("risk_*.tif"))}
    if layers:
        name = st.radio("Risk layer", list(layers), horizontal=True)
        arr, bnd, _, _ = load(layers[name])
        st.caption("Yellow = low, red = high. Screening index, not a calibrated hazard model.")
        overlay_map(ramp(arr, ["#ffffb2", "#fd8d3c", "#bd0026"], 0, 100), bnd, f"risk_{name}", name)
        v = arr[arr != 255]
        if v.size:
            st.bar_chart(pd.Series([(v < 33).mean(), ((v >= 33) & (v < 66)).mean(), (v >= 66).mean()],
                                   index=["low", "medium", "high"], name="share of pixels"))
    else:
        st.info("Run notebook 05 to create the risk layers.")

with tab_future:
    pred = sorted(OUT.glob("landcover_*_predicted.tif"))
    if pred:
        yr = pred[-1].stem.split("_")[1]
        st.subheader(f"Predicted land cover {yr} (CA-Markov)")
        st.caption("Assumes the 2019-2025 rate of change continues. Not validated: only two dates exist.")
        show_map(pred[-1], "future")
        proj = sorted(OUT.glob("projection_*.csv"))
        if proj:
            st.dataframe(pd.read_csv(proj[-1], index_col=0))
    else:
        st.info("Run notebook 05 to create the prediction.")

with tab_dss:
    pc = OUT / "priority_cells.csv"
    if pc.exists():
        df = pd.read_csv(pc)
        top = df.head(15)
        m = folium.Map(location=[top.lat.mean(), top.lon.mean()], zoom_start=10)
        for r in top.itertuples():
            folium.CircleMarker([r.lat, r.lon], radius=6 + 14 * r.priority, color="#bd0026",
                                fill=True, fill_opacity=0.6,
                                tooltip=f"{r.main_issue}: {r.suggested_action}").add_to(m)
        st.caption("Top 15 priority 2 km cells. Bigger circle = higher priority. Hover for the suggested action.")
        st_folium(m, height=480, use_container_width=True, key="dss")
        st.dataframe(top[["lat", "lon", "priority", "main_issue", "suggested_action"]].round(4),
                     use_container_width=True)
    else:
        st.info("Run notebook 05 to create priority_cells.csv.")

with tab_res:
    res = OUT / "results.csv"
    if res.exists():
        df = pd.read_csv(res)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("model")[["macro_F1", "mIoU"]])
    else:
        st.info("Run notebook 03 to create outputs/results.csv.")
