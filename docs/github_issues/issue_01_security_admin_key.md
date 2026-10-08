TITLE: [security] ADMIN_API_KEY failed open + default Postgres password (fixed, needs review)

**Finding**
- `api/auth.py` defaulted the admin key to `"changeme-local-only"`.
- `docker-compose.yml` defaulted `ADMIN_API_KEY` the same way, set
  `POSTGRES_PASSWORD: geoeco`, and bound Postgres to all interfaces.

**Fix applied**
- `expected_admin_key()` now raises `RuntimeError` when the key is unset (fail closed).
- Compose requires `${ADMIN_API_KEY:?…}` / `${POSTGRES_PASSWORD:?…}` and binds
  `127.0.0.1:5432:5432`.
- Verified by grep + compose config on 2026-10-08 [V].

**Residual**
- `api/db.py` still defaults `DATABASE_URL` to the dev credential pair when the
  env var is unset (compose always sets it); hardening scheduled for Stage 6.
- Human action: rotate any exposed key out-of-band; set real values in deploy env.
