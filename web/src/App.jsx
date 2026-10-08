/* GeoEco dashboard (stub): React + MapLibre GL.
 * Layers: LC / confidence / NDVI / water / change. Year/season toggle,
 * 2019-vs-2025 swipe, draw-polygon stub, ours-vs-DW/WC compare, download,
 * About model card. Tiles come from the FastAPI /tiles proxy (TiTiler).
 *
 * API base resolution — runtime-configurable, NO per-environment rebuild:
 *   1. window.__GEOECO_API_URL__ — optional runtime override. Inject via an
 *      inline <script> in index.html or a served config.js, e.g.
 *      <script>window.__GEOECO_API_URL__ = "https://api.example.com"</script>
 *      The value is an API origin (no trailing slash); api() appends the
 *      bare backend path (no /api prefix) in this case.
 *   2. import.meta.env.VITE_API_URL — build-time value, honoured for local
 *      `vite dev` convenience only. Leave it UNSET in Docker builds so the
 *      shipped bundle defaults to same-origin.
 *   3. Default "" — same-origin. api() prefixes backend paths with /api,
 *      which nginx (web/nginx.conf) proxies to the `api` service, and the
 *      Vite dev server proxies likewise (see web/vite.config.js).
 * NOTE: import.meta.env is inlined at BUILD time by Vite, so VITE_API_URL
 * must never be relied on for deployed environments.
 */
import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

const API_ORIGIN =
  (typeof window !== "undefined" && window.__GEOECO_API_URL__) ||
  import.meta.env.VITE_API_URL ||
  "";
// Default "" = same-origin. api() prefixes backend paths with /api (stripped
// by the nginx reverse-proxy in web/nginx.conf, mirrored for `vite dev` in
// web/vite.config.js). An explicit origin override hits the API root directly.
const api = (p) => (API_ORIGIN ? `${API_ORIGIN}${p}` : `/api${p}`);
const HYD = [78.47, 17.38]; // display EPSG:4326

const LAYER_DEFS = [
  { key: "lc", label: "Land cover", productType: "lc" },
  { key: "confidence", label: "Confidence", productType: "confidence" },
  { key: "ndvi", label: "NDVI", productType: "ndvi" },
  { key: "water", label: "Water", productType: "water" },
  { key: "change", label: "Change 2019→2025", productType: "change" },
];

// Demo catalog ids (mirror api/db.py STUB_PRODUCTS)
const IDS = { 2019: 1, 2025: 2, conf: 3, ndvi: 4, water: 5, change: 6 };

// Sample polygon (~12 km² over Hyderabad) for the draw-polygon stub.
const SAMPLE_POLYGON = {
  type: "Polygon",
  coordinates: [[
    [78.40, 17.35], [78.45, 17.35], [78.45, 17.40],
    [78.40, 17.40], [78.40, 17.35],
  ]],
};

function tileUrl(productId, compare) {
  // compare: "ours" | "dynamic_world" | "worldcover" — global baselines are
  // overlaid client-side in prod; stub reuses our tiles with a label.
  void compare;
  return api(`/tiles/${productId}/{z}/{x}/{y}.png`);
}

export default function App() {
  const mapRef = useRef(null);
  const mapEl = useRef(null);
  const [layer, setLayer] = useState("lc");
  const [year, setYear] = useState(2025);
  const [season, setSeason] = useState("post");
  const [swipe, setSwipe] = useState(false);
  const [swipeMix, setSwipeMix] = useState(0.5);
  const [compare, setCompare] = useState("ours");
  const [stats, setStats] = useState(null);
  const [card, setCard] = useState(null);
  const [showAbout, setShowAbout] = useState(false);

  // (Re)build raster sources whenever the selection changes.
  useEffect(() => {
    if (!mapEl.current) return;
    if (!mapRef.current) {
      mapRef.current = new maplibregl.Map({
        container: mapEl.current,
        style: "https://demotiles.maplibre.org/style.json",
        center: HYD,
        zoom: 10,
      });
      mapRef.current.addControl(new maplibregl.NavigationControl(), "top-right");
    }
    const map = mapRef.current;
    const draw = () => {
      const baseId = layer === "change" ? IDS.change : IDS[year] ?? IDS[2025];
      for (const id of ["lyr-before", "lyr-after"]) {
        if (map.getLayer(id)) map.removeLayer(id);
        if (map.getSource(id)) map.removeSource(id);
      }
      const add = (id, pid, opacity) => {
        map.addSource(id, {
          type: "raster",
          tiles: [tileUrl(pid, compare)],
          tileSize: 256,
        });
        map.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": opacity } });
      };
      if (swipe && layer === "lc") {
        add("lyr-before", IDS[2019], 1 - swipeMix);
        add("lyr-after", IDS[2025], swipeMix);
      } else {
        add("lyr-after", baseId, 1);
      }
    };
    if (map.isStyleLoaded()) draw();
    else map.once("load", draw);
    void season; // season selects the product in prod catalog lookup
  }, [layer, year, season, swipe, swipeMix, compare]);

  const runAnalyze = async () => {
    // Draw-polygon stub: posts SAMPLE_POLYGON (full draw via
    // maplibre-gl-draw in prod) and shows class areas.
    const res = await fetch(api("/analyze"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        polygon: SAMPLE_POLYGON,
        from_product_id: IDS[2019],
        to_product_id: IDS[2025],
      }),
    });
    setStats(res.ok ? await res.json() : { error: await res.text() });
  };

  const loadCard = async () => {
    if (!card) {
      const res = await fetch(api("/models/1/card"));
      setCard(await res.json());
    }
    setShowAbout((v) => !v);
  };

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "system-ui" }}>
      <aside style={{ width: 300, padding: 12, overflowY: "auto", borderRight: "1px solid #ddd" }}>
        <h2>GeoEco · Hyderabad</h2>

        <h4>Layer</h4>
        {LAYER_DEFS.map((l) => (
          <label key={l.key} style={{ display: "block" }}>
            <input type="radio" checked={layer === l.key} onChange={() => setLayer(l.key)} />
            {l.label}
          </label>
        ))}

        <h4>Year / Season</h4>
        <div>
          {[2019, 2025].map((y) => (
            <button key={y} onClick={() => setYear(y)}
              style={{ fontWeight: year === y ? "bold" : "normal", marginRight: 6 }}>{y}</button>
          ))}
        </div>
        <div style={{ marginTop: 6 }}>
          {["pre", "monsoon", "post"].map((s) => (
            <button key={s} onClick={() => setSeason(s)}
              style={{ fontWeight: season === s ? "bold" : "normal", marginRight: 6 }}>{s}</button>
          ))}
        </div>

        <h4>Swipe 2019 vs 2025</h4>
        <label>
          <input type="checkbox" checked={swipe} onChange={(e) => setSwipe(e.target.checked)} />
          before/after swipe (LC)
        </label>
        {swipe && (
          <input type="range" min={0} max={1} step={0.05} value={swipeMix}
            onChange={(e) => setSwipeMix(Number(e.target.value))} style={{ width: "100%" }} />
        )}

        <h4>Compare: ours vs global</h4>
        {["ours", "dynamic_world", "worldcover"].map((c) => (
          <label key={c} style={{ display: "block" }}>
            <input type="radio" checked={compare === c} onChange={() => setCompare(c)} />{c}
          </label>
        ))}

        <h4>Polygon stats (stub)</h4>
        <button onClick={runAnalyze}>Analyze sample polygon</button>
        {stats && <pre style={{ fontSize: 11 }}>{JSON.stringify(stats, null, 1)}</pre>}

        <h4>Export</h4>
        <a href={api(`/download/${IDS[year] ?? 2}`)} target="_blank" rel="noreferrer">
          Download GeoTIFF (product {IDS[year] ?? 2})
        </a>

        <h4>About</h4>
        <button onClick={loadCard}>Model card</button>
        {showAbout && card && (
          <pre style={{ fontSize: 11 }}>{JSON.stringify(card, null, 1)}</pre>
        )}
      </aside>
      <div ref={mapEl} style={{ flex: 1 }} />
    </div>
  );
}
