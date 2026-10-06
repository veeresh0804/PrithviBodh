"""Model-card route (public, read-only)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.db import STUB_MODEL_CARDS

router = APIRouter(tags=["models"])


@router.get("/models/{model_id}/card")
def model_card(model_id: int) -> dict:
    """GET /models/{id}/card — transparency card (NFR-08)."""
    card = STUB_MODEL_CARDS.get(model_id)
    if card is None:
        raise HTTPException(status_code=404,
                            detail=f"Model {model_id} not found")
    return card
