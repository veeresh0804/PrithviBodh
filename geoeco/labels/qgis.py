"""Per-member QGIS projects (.qgz) for human labelling.

Each project loads ONLY: the member's points, Esri World Imagery, Google
Satellite, and (if exported) the Sentinel-2 post-monsoon median COG plus the
admin boundary. NO classification layer is ever added (rule: no suggested
labels, no benchmark-derived display).

.qgz is a zip containing a .qgs XML project. Layer sources are relative so
the whole data/labels/ folder stays portable. Missing optional files
(S2 COG, admin boundary) are omitted with a loud warning — re-run this
module after the Earth Engine export instead of shipping a broken project.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from geoeco.labels.pipeline import REPO, load_labelling_config
from geoeco.labels.protocol import CLASSES

ESRI_XYZ = ("type=xyz&url=https://server.arcgisonline.com/ArcGIS/rest/services"
            "/World_Imagery/MapServer/tile/{z}/{y}/{x}&zmax=19")
GOOGLE_XYZ = "type=xyz&url=https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&zmax=20"


def _raster_xyz(layer_id: str, name: str, source: str) -> ET.Element:
    ml = ET.Element("maplayer", {"type": "raster"})
    ET.SubElement(ml, "id").text = layer_id
    ET.SubElement(ml, "datasource").text = source
    ET.SubElement(ml, "provider").text = "wms"
    ET.SubElement(ml, "name").text = name
    ET.SubElement(ml, "srs").text = "EPSG:4326"
    return ml


def _points_layer(layer_id: str, name: str, source: str, labeller: str) -> ET.Element:
    ml = ET.Element("maplayer", {"type": "vector", "geometry": "Point"})
    ET.SubElement(ml, "id").text = layer_id
    ET.SubElement(ml, "datasource").text = source
    ET.SubElement(ml, "provider").text = "ogr"
    ET.SubElement(ml, "name").text = name
    ET.SubElement(ml, "srs").text = "EPSG:4326"
    fc = ET.SubElement(ml, "fieldConfiguration")
    for fname in ["id", "block", "label", "label_name", "labeller", "notes",
                  "split", "fine_block", "overlap_id"]:
        f = ET.SubElement(fc, "field", {"name": fname})
        if fname == "label":
            ew = ET.SubElement(f, "editWidget", {"type": "ValueMap"})
            cfg = ET.SubElement(ew, "config")
            opt = ET.SubElement(cfg, "Option", {"type": "Map"})
            for i, c in enumerate(CLASSES):
                ET.SubElement(opt, "Option", {"value": str(i), "type": "QString",
                                              "name": f"{c} ({i})"})
        elif fname == "labeller":
            ew = ET.SubElement(f, "editWidget", {"type": "TextEdit"})
            cfg = ET.SubElement(ew, "config")
            ET.SubElement(cfg, "Option", {"value": labeller, "type": "QString",
                                          "name": "DefaultValue"})
        else:
            ET.SubElement(f, "editWidget", {"type": "TextEdit"})
    return ml


def build_qgs(member_id: str, warnings: list[str]) -> ET.ElementTree:
    """Build the project XML. Appends warnings for omitted missing files."""
    cfg = load_labelling_config()["labelling"]
    qgis = ET.Element("qgis", {"projectname": f"Hyderabad labelling — member {member_id}",
                               "version": "3.28.0"})
    crs = ET.SubElement(qgis, "projectCrs")
    srs = ET.SubElement(crs, "spatialrefsys")
    ET.SubElement(srs, "proj4").text = "+proj=longlat +datum=WGS84 +no_defs"
    ET.SubElement(srs, "srsid").text = "3452"
    ET.SubElement(srs, "srid").text = "4326"
    ET.SubElement(srs, "authid").text = "EPSG:4326"
    tree = ET.SubElement(qgis, "layer-tree-group", {"name": "Layers"})
    canvas = ET.SubElement(qgis, "mapcanvas", {"name": "theMapCanvas"})
    ET.SubElement(canvas, "units").text = "degrees"
    aoi = load_labelling_config()["aoi"]["bounds_wgs84"]
    ext = ET.SubElement(canvas, "extent")
    for tag, val in [("xmin", aoi[0]), ("ymin", aoi[1]),
                     ("xmax", aoi[2]), ("ymax", aoi[3])]:
        ET.SubElement(ext, tag).text = str(val)
    layers = ET.SubElement(qgis, "maplayers")
    layers.append(_raster_xyz("esri_xyz", "Esri World Imagery", ESRI_XYZ))
    layers.append(_raster_xyz("google_xyz", "Google Satellite", GOOGLE_XYZ))
    for lid in ["esri_xyz", "google_xyz"]:
        ET.SubElement(tree, "layer-tree-layer", {"id": lid})
    qgis_dir = REPO / "data" / "labels" / "qgis"
    s2 = REPO / str(cfg["s2_post_monsoon_cog"])
    admin = REPO / str(cfg["admin_boundary"])
    if s2.is_file():
        rel = f"../../raw/composites/{s2.name}"
        rl = ET.Element("maplayer", {"type": "raster"})
        ET.SubElement(rl, "id").text = "s2_post"
        ET.SubElement(rl, "datasource").text = rel
        ET.SubElement(rl, "provider").text = "gdal"
        ET.SubElement(rl, "name").text = "S2 post-monsoon median (EE export)"
        layers.append(rl)
        ET.SubElement(tree, "layer-tree-layer", {"id": "s2_post"})
    else:
        warnings.append(f"S2 layer omitted (missing {s2}); export via Earth Engine, then re-run.")
    if admin.is_file():
        rl = ET.Element("maplayer", {"type": "vector", "geometry": "Polygon"})
        ET.SubElement(rl, "id").text = "admin"
        ET.SubElement(rl, "datasource").text = "../../admin/hyderabad_boundary.geojson"
        ET.SubElement(rl, "provider").text = "ogr"
        ET.SubElement(rl, "name").text = "Admin boundary (SoI)"
        layers.append(rl)
        ET.SubElement(tree, "layer-tree-layer", {"id": "admin"})
    else:
        warnings.append(f"Admin boundary omitted (missing {admin}); add the SoI file, then re-run.")
    pts = _points_layer("pts_" + member_id, f"Points — member {member_id}",
                        f"../members/member_{member_id}.geojson", member_id)
    layers.append(pts)
    ET.SubElement(tree, "layer-tree-layer", {"id": "pts_" + member_id})
    return ET.ElementTree(qgis)


def write_qgz(member_id: str, dest: Path) -> list[str]:
    """Write data/labels/qgis/member_<id>.qgz. Returns warnings."""
    warnings: list[str] = []
    tree = build_qgs(member_id, warnings)
    dest.parent.mkdir(parents=True, exist_ok=True)
    qgs_name = dest.stem + ".qgs"
    import io
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(qgs_name, buf.getvalue())
    return warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate per-member .qgz projects.")
    ap.parse_args(argv)
    cfg = load_labelling_config()["labelling"]
    all_warnings: dict[str, list[str]] = {}
    for m in cfg["members"]:
        dest = REPO / "data" / "labels" / "qgis" / f"member_{m['id']}.qgz"
        all_warnings[m["id"]] = write_qgz(m["id"], dest)
        print(f"wrote {dest}")
    for mid, ws in all_warnings.items():
        for w in ws:
            print(f"WARNING[{mid}]: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
