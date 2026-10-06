"""API-key guard for write/admin routes.

Public (read-only): /health, /products, /tiles/*, /stats/*, /analyze,
/change, /compare, /models/*/card, /download/*.
Keyed (admin): POST /admin/jobs/inference.

Key comes from env ADMIN_API_KEY; never committed to the repo.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


def expected_admin_key() -> str:
    key = os.environ.get("ADMIN_API_KEY", "")
    if not key.strip():
        raise RuntimeError(
            "ADMIN_API_KEY is not set: export ADMIN_API_KEY "
            "before starting the API."
        )
    return key


async def require_admin_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    if x_api_key != expected_admin_key():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key",
        )
    return x_api_key
