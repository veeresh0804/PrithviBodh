# Product readiness — serving stack (M6 prep)

Verified 2026-10-07 on Windows (PowerShell), repo `C:\Users\manoh\Desktop\Major Project`.
Scope: is the stack demonstrably runnable. No images built/pulled, no `compose up` (no daemon expected).

| Component | Status | Evidence | What remains |
|---|---|---|---|
| docker-compose.yml | VERIFIED (static) | `docker compose config` exits 0 with dummy `POSTGRES_PASSWORD`/`ADMIN_API_KEY`; 4 services resolve (postgis 16-3.4, titiler→Dockerfile.titiler, api→Dockerfile.api, web→web/Dockerfile). Without the two required vars it fails loudly by design (`:?` interpolation error). | Runtime `up` on a daemon host; real secrets; `cog-data` volume is empty (see demo COGs). |
| FastAPI (`api/`) | VERIFIED (live) | `uvicorn api.app:app --port 8001` booted (uvicorn 0.54.0, fastapi 0.142.2); `/health` → `{"status":"ok"}`; `/products?year=2025` → 5 stub rows; `pytest tests/test_api.py` → **11 passed**. | Stub catalog only (`s3://geoeco-demo/…` don't exist); no DB — `DATABASE_URL` unused by routers yet. |
| TiTiler | NOT-VERIFIED-LOCALLY | `Dockerfile.titiler` is just a pin (`ghcr.io/developmentseed/titiler:0.18`, matches `titiler.application>=0.18`); `titiler`/`osgeo` not importable in this env; no image build attempted. | Serve one real COG and `GET /cog/tiles` smoke-test; confirm `COG_ROOT=/cog` mount. |
| PostGIS | NOT-VERIFIED-LOCALLY | Only pin checked (`postgis/postgis:16-3.4`, port `127.0.0.1:5432`); no daemon here, no migrations/schema/data load exist in repo. | Init schema + load regions/products; wire API off stubs onto `DATABASE_URL`. |
| web (`web/`) | VERIFIED (build) | node v24.20.0 / npm present; `npm install` (102 pkgs) + `npm run build` exit 0 → `dist/index.html` + assets in 9.18s (vite v5.4.21; chunk-size warning: 951 kB JS). Deps: react 18.3.1, maplibre-gl 4.5.0, vite 5.4.0, @vitejs/plugin-react 4.3.0. | Fix `VITE_API_URL` bake-in (below); code-split the 951 kB chunk. |
| Dependency pins | VERIFIED (no conflicts) | Same package + same lower bound everywhere: fastapi ≥0.110, uvicorn ≥0.29 (api adds `[standard]` extras), sqlalchemy ≥2.0, psycopg2-binary ≥2.9, httpx ≥0.27. Installed env satisfies all. Gaps, not conflicts: `titiler.application`/`GDAL` and `pydantic` (api needs ≥2.6) are absent from root `requirements.txt`/base `pyproject` deps. | Add `pydantic` to base deps or document api-only install; root env can't serve tiles without `requirements-api.txt`. |

## Biggest deployment risk
**The `web` image bakes in `VITE_API_URL=http://localhost:8000` and it cannot be fixed at deploy time.**
`web/Dockerfile` runs `npm run build` with no `ARG VITE_API_URL`, and `compose` passes `VITE_API_URL` as *runtime* environment — but Vite inlines `import.meta.env.*` at **build** time, so the shipped bundle always falls back to `http://localhost:8000` (`App.jsx:10`). Any non-local deployment (different host/URL) gets a UI whose every API call misses. Fix: add `ARG VITE_API_URL` + `--build-arg` in compose, or serve API under same-origin `/api` via `nginx.conf`. (Runner-up: all tile URLs point at nonexistent `s3://geoeco-demo/` COGs — TiTiler will 5xx until demo COGs are published and `cog-data` is populated.)
