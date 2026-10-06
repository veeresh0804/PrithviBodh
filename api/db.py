"""SQLAlchemy models mirroring the PostGIS schema (PROJECT_CONTEXT.md section 5.5).

Tables: REGION / ADMIN_UNIT / MAP_PRODUCT / MODEL_VERSION / AREA_STAT / CHANGE_STAT.
Geometry columns are Text (WKT) in this stub so the API imports without
geoalchemy2; in prod they become GeoAlchemy2 Geometry columns against
PostGIS and ST_Area/ST_Intersection do the real work.

Routers serve in-memory STUB_* rows when DATABASE_URL is unset, so the
stack is runnable with zero infra (docker-compose wires real PostGIS).
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Region(Base):
    __tablename__ = "region"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False, unique=True)
    aoi = Column(Text, nullable=False)  # prod: Geometry("MULTIPOLYGON", 4326)
    admin_units = relationship("AdminUnit", back_populates="region")
    products = relationship("MapProduct", back_populates="region")


class AdminUnit(Base):
    __tablename__ = "admin_unit"
    id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey("region.id"), nullable=False)
    name = Column(String(128), nullable=False)
    level = Column(String(64), nullable=False, default="ward")
    geom = Column(Text, nullable=False)  # prod: Geometry("MULTIPOLYGON", 4326)
    region = relationship("Region", back_populates="admin_units")


class ModelVersion(Base):
    __tablename__ = "model_version"
    id = Column(Integer, primary_key=True)
    mlflow_run_id = Column(String(64), nullable=False)
    name = Column(String(128), nullable=False)
    test_macro_f1 = Column(Float, nullable=True)
    model_card_url = Column(String(512), nullable=True)
    products = relationship("MapProduct", back_populates="model_version")


class MapProduct(Base):
    __tablename__ = "map_product"
    id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey("region.id"), nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_version.id"), nullable=True)
    product_type = Column(String(64), nullable=False)  # lc | confidence | ndvi | water | change
    year = Column(Integer, nullable=False)
    season = Column(String(16), nullable=False)  # pre | monsoon | post | annual
    cog_url = Column(String(1024), nullable=False)
    region = relationship("Region", back_populates="products")
    model_version = relationship("ModelVersion", back_populates="products")


class AreaStat(Base):
    __tablename__ = "area_stat"
    id = Column(Integer, primary_key=True)
    map_product_id = Column(Integer, ForeignKey("map_product.id"), nullable=False)
    admin_unit_id = Column(Integer, ForeignKey("admin_unit.id"), nullable=True)
    class_name = Column(String(64), nullable=False)
    area_ha = Column(Float, nullable=False)
    ci_low_ha = Column(Float, nullable=True)
    ci_high_ha = Column(Float, nullable=True)


class ChangeStat(Base):
    __tablename__ = "change_stat"
    id = Column(Integer, primary_key=True)
    from_product_id = Column(Integer, ForeignKey("map_product.id"), nullable=False)
    to_product_id = Column(Integer, ForeignKey("map_product.id"), nullable=False)
    from_class = Column(String(64), nullable=False)
    to_class = Column(String(64), nullable=False)
    area_ha = Column(Float, nullable=False)


# ---------------------------------------------------------------------------
# Engine helpers (lazy; import-safe without a live DB)
# ---------------------------------------------------------------------------


def database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://geoeco:geoeco@postgis:5432/geoeco",
    )


def get_session():
    engine = create_engine(database_url(), pool_pre_ping=True, future=True)
    return sessionmaker(bind=engine, autoflush=False)()


# ---------------------------------------------------------------------------
# Dependency-free stub rows (used when PostGIS is unreachable)
# ---------------------------------------------------------------------------

STUB_PRODUCTS: list[dict] = [
    {"id": 1, "region": "hyderabad", "product_type": "lc", "year": 2019,
     "season": "post", "cog_url": "s3://geoeco-demo/hyd_2019_post_lc.tif"},
    {"id": 2, "region": "hyderabad", "product_type": "lc", "year": 2025,
     "season": "post", "cog_url": "s3://geoeco-demo/hyd_2025_post_lc.tif"},
    {"id": 3, "region": "hyderabad", "product_type": "confidence", "year": 2025,
     "season": "post", "cog_url": "s3://geoeco-demo/hyd_2025_post_conf.tif"},
    {"id": 4, "region": "hyderabad", "product_type": "ndvi", "year": 2025,
     "season": "post", "cog_url": "s3://geoeco-demo/hyd_2025_post_ndvi.tif"},
    {"id": 5, "region": "hyderabad", "product_type": "water", "year": 2025,
     "season": "monsoon", "cog_url": "s3://geoeco-demo/hyd_2025_monsoon_water.tif"},
    {"id": 6, "region": "hyderabad", "product_type": "change", "year": 2025,
     "season": "annual", "cog_url": "s3://geoeco-demo/hyd_2019_2025_change.tif"},
]

STUB_AREA_HA: dict[str, float] = {
    "water": 4200.0,
    "tree_cover": 18500.0,
    "cropland": 96000.0,
    "built_up": 78000.0,
    "bare_rocky": 31000.0,
    "grass_shrub": 42000.0,
}

STUB_MODEL_CARDS: dict[int, dict] = {
    1: {
        "id": 1,
        "name": "rf-early-fusion-hyd-v1",
        "mlflow_run_id": "local-stub-run-001",
        "test_macro_f1": 0.78,
        "classes": ["water", "tree_cover", "cropland", "built_up",
                    "bare_rocky", "grass_shrub"],
        "benchmarks": {"dynamic_world": 0.66, "worldcover": 0.71},
        "limits": ["granite vs rooftop confusion", "seasonal tank edges",
                   "validated for Hyderabad 60x60km only"],
    }
}
