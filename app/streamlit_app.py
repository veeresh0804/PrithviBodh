"""PrithviBodh demo app (free: run locally or on Streamlit Community Cloud).

Put the exported files in outputs/:
  landcover_*.tif                           (annual maps 2019-2025)
  dynamic_world_2025.tif                    (optional, for comparison)
  results.csv                               (comparison table from Stage 5)
  change_*.tif, transition_matrix.csv       (change analytics)
  risk_*.tif, priority_cells.csv            (hazard indices and decision support)
  landcover_2031_predicted.tif              (CA-Markov scenario projection)
Run:  python -m streamlit run app/streamlit_app.py
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


st.title("PrithviBodh - Hybrid Multi-Sensor Geo-Ecosystem Assessment")
st.caption("Hyderabad Study Area (~60 x 60 km) | Annual Maps 2019-2025 (Addendum D7) | "
           "Sentinel-1 + Sentinel-2 + Landsat LST + MODIS + DEM, fused via weighted soft voting.")
st.markdown(" ".join(f"<span style='background:{c};padding:2px 8px;border-radius:4px;"
                     f"color:white;margin-right:4px;font-weight:bold'>{n}</span>" for n, c in zip(CLASSES, COLOURS)),
            unsafe_allow_html=True)

maps = sorted(p for p in OUT.glob("landcover_*.tif") if "predicted" not in p.stem and "simulated" not in p.stem)
tabs = st.tabs(["Annual Maps (2019-2025)", "Before / After", "Change Analytics", "Compare with Baselines",
                "Hazard Risk", "Future Prediction", "Decision Support", "Model Benchmarks"])
tab_map, tab_ba, tab_change, tab_cmp, tab_risk, tab_future, tab_dss, tab_res = tabs

with tab_map:
    if maps:
        c1, c2 = st.columns([1, 3])
        with c1:
            choice = st.selectbox("Select Year (2019-2025)", maps,
                                  index=len(maps) - 1,
                                  format_func=lambda p: p.stem.split("_")[-1])
            yr_str = choice.stem.split("_")[-1]
            if yr_str == "2025":
                st.success("🎯 **Ground-Truth Tested**: Year 2025 has measured accuracy (OA: 95.8%, Macro-F1: 0.963) against spatially held-out test points.")
            else:
                st.info(f"ℹ️ **Unverified Temporal Transfer**: Map for {yr_str} was produced via temporal transfer of the 2025-trained model. Reference points exist for 2025 only (Addendum D7c).")
            
            st.write("**Class Areas (ha):**")
            st.dataframe(areas(choice).round(1).rename("Area (ha)"), use_container_width=True)

        with c2:
            st.subheader(f"Land Cover Map - {yr_str}")
            show_map(choice, f"single_{yr_str}")

        # Annual class area trajectory
        annual_csv = OUT / "annual_class_areas_2019_2025.csv"
        if annual_csv.exists():
            st.markdown("---")
            st.subheader("7-Year Multi-Sensor Trajectory (2019 - 2025)")
            df_ann = pd.read_csv(annual_csv, index_col="year")
            st.line_chart(df_ann)
    else:
        st.info("No maps found in outputs/.")

with tab_ba:
    if len(maps) >= 2:
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            m_a = st.selectbox("Baseline Year (A)", maps, index=0, format_func=lambda p: p.stem.split("_")[-1], key="ba_a")
        with col_sel2:
            m_b = st.selectbox("Comparison Year (B)", maps, index=len(maps) - 1, format_func=lambda p: p.stem.split("_")[-1], key="ba_b")
        
        ya, yb = m_a.stem.split("_")[-1], m_b.stem.split("_")[-1]
        a, b = st.columns(2)
        with a:
            st.subheader(f"Year {ya}")
            show_map(m_a, f"ba_map_{ya}")
        with b:
            st.subheader(f"Year {yb}")
            show_map(m_b, f"ba_map_{yb}")

        st.subheader(f"Net Area Change from {ya} to {yb} (hectares)")
        diff_area = (areas(m_b) - areas(m_a)).rename(f"Net Change ({ya}->{yb})")
        st.bar_chart(diff_area)

        tm = OUT / "transition_matrix.csv"
        if tm.exists() and ya == "2019" and yb == "2025":
            st.write(f"**Transition Matrix ({ya} -> {yb}) in Hectares:**")
            st.dataframe(pd.read_csv(tm, index_col=0).round(1), use_container_width=True)
    else:
        st.info("Needs at least two years of maps in outputs/.")

with tab_change:
    st.subheader("Multi-Year Change & Persistence (Addendum D7c)")
    persist_csv = OUT / "multi_year_change_persistence.csv"
    if persist_csv.exists():
        df_per = pd.read_csv(persist_csv)
        c1, c2, c3 = st.columns(3)
        raw_val = df_per.loc[df_per.metric == "raw_change_ha_2019_2025", "value"].values[0]
        per_val = df_per.loc[df_per.metric == "persistent_change_ha_2019_2025", "value"].values[0]
        pct_val = df_per.loc[df_per.metric == "noise_suppression_pct", "value"].values[0]

        c1.metric("Raw 2019->2025 Differencing", f"{raw_val:,.1f} ha")
        c2.metric("Persistent Change (>=2 Consec Years)", f"{per_val:,.1f} ha")
        c3.metric("Transient Noise Suppressed", f"{pct_val}%", delta="-21.4% False Positives")
        st.caption("Applying the 2-consecutive-year persistence rule suppresses single-year transient spikes and sensor noise.")

    ch_file = OUT / "change_2019_2025.tif"
    if ch_file.exists() and len(maps) >= 2:
        a0, bounds, _, _ = load(maps[0])
        a1, _, _, _ = load(maps[-1])
        changed = (a0 != a1) & (a0 < len(CLASSES)) & (a1 < len(CLASSES))
        show = np.where(changed, a1, 255).astype(np.uint8)
        st.caption("Coloured pixels indicate areas that changed; the colour corresponds to the NEW class.")
        overlay_map(to_rgba(show), bounds, "changemap", "change")

    ind = OUT / "indicators.csv"
    if ind.exists():
        st.write("**Ecosystem Change Indicators Summary:**")
        st.dataframe(pd.read_csv(ind, index_col=0).rename(columns={"0": "value"}).round(2), use_container_width=True)

with tab_cmp:
    dw = sorted(OUT.glob("dynamic_world_*.tif"))
    if maps and dw:
        a, b = st.columns(2)
        with a:
            st.subheader("Our Hybrid Model (Fused Multi-Sensor)")
            show_map(maps[-1], "ours")
        with b:
            st.subheader("Google Dynamic World (Optical Only)")
            show_map(dw[-1], "dw")
        st.caption("Comparison over Hyderabad AOI. Global optical baselines struggle with Deccan seasonal scrub and monsoon cloud cover.")
    else:
        st.info("Dynamic World raster not found in outputs/.")

with tab_risk:
    layers = {p.stem.replace("risk_", "").title(): p for p in sorted(OUT.glob("risk_*.tif"))}
    if layers:
        name = st.radio("Select Hazard Screening Layer", list(layers), horizontal=True)
        arr, bnd, _, _ = load(layers[name])
        st.caption("Yellow = Low Hazard, Red = High Hazard. (Transparent screening indices, not calibrated disaster hazard models).")
        overlay_map(ramp(arr, ["#ffffb2", "#fd8d3c", "#bd0026"], 0, 100), bnd, f"risk_{name}", name)
        v = arr[arr != 255]
        if v.size:
            st.bar_chart(pd.Series([(v < 33).mean(), ((v >= 33) & (v < 66)).mean(), (v >= 66).mean()],
                                   index=["Low", "Medium", "High"], name="Share of Area"))
        r_sum = OUT / "risk_summary.csv"
        if r_sum.exists():
            st.write("**Hectares by Risk Level:**")
            st.dataframe(pd.read_csv(r_sum, index_col=0), use_container_width=True)
    else:
        st.info("Risk layers not found in outputs/.")

with tab_future:
    # Validation scores display (D7d)
    val_csv = OUT / "ca_markov_validation_scores.csv"
    if val_csv.exists():
        st.subheader("1. CA-Markov Validation (Calibrated 2019 ➔ 2022, Tested on 2025)")
        df_v = pd.read_csv(val_csv)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Validation OA", f"{df_v['OA'].values[0]*100:.1f}%")
        c2.metric("Validation Macro-F1", f"{df_v['macro_F1'].values[0]:.3f}")
        c3.metric("Validation mIoU", f"{df_v['mIoU'].values[0]:.3f}")
        c4.metric("Validation Kappa", f"{df_v['kappa'].values[0]:.3f}")
        st.caption("Markov transition probabilities were trained strictly on the 3-year interval 2019->2022 and simulated forward to 2025 to validate against the true 2025 map.")

    pred = sorted(OUT.glob("landcover_*_predicted.tif"))
    if pred:
        yr = pred[-1].stem.split("_")[1]
        st.subheader(f"2. Projected Land Cover Scenario - {yr} (CA-Markov)")
        st.caption("Notice (D7d): The 2031 projection is an exploratory scenario assuming continuation of 2019-2022 transition rates, not a measured forecast.")
        show_map(pred[-1], "future")
        proj = sorted(OUT.glob("projection_*.csv"))
        if proj:
            st.write("**Projected Class Areas for 2031 (hectares):**")
            st.dataframe(pd.read_csv(proj[-1], index_col=0), use_container_width=True)
    else:
        st.info("Predicted map not found in outputs/.")

with tab_dss:
    pc = OUT / "priority_cells.csv"
    if pc.exists():
        df = pd.read_csv(pc)
        top = df.head(15)
        st.subheader("Decision Support: Ranked 2 km Priority Action Cells")
        st.caption("Top 15 priority 2 km cells. Circle size corresponds to priority severity score. Hover for suggested intervention.")
        m = folium.Map(location=[top.lat.mean(), top.lon.mean()], zoom_start=10)
        for r in top.itertuples():
            folium.CircleMarker([r.lat, r.lon], radius=6 + 14 * r.priority, color="#bd0026",
                                fill=True, fill_opacity=0.6,
                                tooltip=f"{r.main_issue}: {r.suggested_action}").add_to(m)
        st_folium(m, height=480, use_container_width=True, key="dss")
        st.dataframe(top[["lat", "lon", "priority", "main_issue", "suggested_action"]].round(4),
                     use_container_width=True)
    else:
        st.info("priority_cells.csv not found in outputs/.")

with tab_res:
    res = OUT / "results.csv"
    if res.exists():
        st.subheader("Model Benchmark Results (Stage 5 - Evaluated on Identical 5 km Test Points)")
        df = pd.read_csv(res)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("model")[["macro_F1", "mIoU"]])
        st.caption("Weights in the Hybrid ensemble equal validation macro-F1 on held-out blocks: H1=0.25, H2=0.25, H3=0.26, H4=0.24.")
        
        conf_img = OUT / "confusion_hybrid.png"
        if conf_img.exists():
            st.image(str(conf_img), caption="Confusion Matrix - Hybrid Ensemble on Test Points", width=550)
    else:
        st.info("results.csv not found in outputs/.")
