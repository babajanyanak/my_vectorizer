#!/usr/bin/env python3
"""
vectorize_floor.py — Floor Plan Vectorizer
Converts raster floor plan images with colored room fills into clean SVG polygons.

Usage:
    python vectorize_floor.py \
        --plan plan.png \
        --overlay overlay.png \
        --mapping lots.csv \
        --out-dir ./output

Author: Internal Real Estate Dev Team
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import math
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("vectorize_floor")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LotMeta:
    """Metadata from the mapping file for a single lot."""
    lot_id: str
    label: str = ""
    status: str = ""
    area: str = ""


@dataclass
class LotPolygon:
    """Detected polygon + metadata for one room/lot."""
    lot_id: str
    polygon_points: list[list[float]]
    bbox: list[float]          # [x, y, w, h]
    centroid: list[float]      # [cx, cy]
    area_px: float
    status: str = ""
    label: str = ""
    suspicious: bool = False
    suspicious_reason: str = ""


@dataclass
class ValidationReport:
    """Quality report produced after processing."""
    total_regions_found: int = 0
    lots_matched: int = 0
    lots_unmatched: int = 0
    unmatched_lot_ids: list[str] = field(default_factory=list)
    unknown_polygons: int = 0
    suspicious_polygons: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load image from disk, raise on failure."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.size == 0:
        raise ValueError(f"Empty image: {path}")
    logger.info("Loaded image %s — %dx%d", path.name, img.shape[1], img.shape[0])
    return img


def align_images(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """
    Resize overlay to match base dimensions if they differ.
    For production use, consider SIFT-based homography alignment.
    """
    bh, bw = base.shape[:2]
    oh, ow = overlay.shape[:2]
    if (bh, bw) == (oh, ow):
        return overlay
    logger.warning(
        "Image sizes differ (base=%dx%d, overlay=%dx%d) — resizing overlay.",
        bw, bh, ow, oh,
    )
    return cv2.resize(overlay, (bw, bh), interpolation=cv2.INTER_LINEAR)


def load_mapping(path: Path) -> dict[str, LotMeta]:
    """
    Load lot mapping from CSV or JSON.
    CSV must have at least a 'lot_id' column.
    JSON must be a list of objects with at least 'lot_id'.
    """
    suffix = path.suffix.lower()
    rows: list[dict] = []

    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and "lots" in data:
            rows = data["lots"]
        else:
            raise ValueError("JSON mapping must be a list or {'lots': [...]}")
    else:
        raise ValueError(f"Unsupported mapping format: {suffix}")

    mapping: dict[str, LotMeta] = {}
    for row in rows:
        lid = str(row.get("lot_id", "")).strip()
        if not lid:
            continue
        mapping[lid] = LotMeta(
            lot_id=lid,
            label=str(row.get("label", lid)).strip(),
            status=str(row.get("status", "")).strip(),
            area=str(row.get("area", "")).strip(),
        )
    logger.info("Loaded %d lots from mapping: %s", len(mapping), path.name)
    return mapping


# ---------------------------------------------------------------------------
# Color segmentation
# ---------------------------------------------------------------------------

def build_room_mask(overlay: np.ndarray, debug_dir: Optional[Path] = None) -> np.ndarray:
    """
    Segment colored room fills from the overlay image.

    Strategy:
      1. Convert to LAB, compute saturation-like channel.
      2. Threshold to find non-white, non-black filled regions.
      3. Morphological closing to bridge door gaps.
      4. Hole filling.
      5. Remove small noise regions.

    Returns a binary uint8 mask (255 = room area).
    """
    h, w = overlay.shape[:2]

    # --- Remove near-white background ---
    gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    _, bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # white bg
    _, dark_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)  # black walls/lines

    # --- Work in LAB for better color discrimination ---
    lab = cv2.cvtColor(overlay, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Low-saturation filter: rooms are lightly tinted
    # Detect any pixel that's not pure white/gray by checking a+b deviation
    a_dev = np.abs(a_ch.astype(np.int16) - 128).astype(np.uint8)
    b_dev = np.abs(b_ch.astype(np.int16) - 128).astype(np.uint8)
    color_dev = cv2.add(a_dev, b_dev)

    # Threshold: pixels with some color tint (even very light beige/green)
    _, tinted = cv2.threshold(color_dev, 6, 255, cv2.THRESH_BINARY)

    # Also catch gray rooms (light fill without color) using L channel
    _, light_gray = cv2.threshold(l_ch, 170, 255, cv2.THRESH_BINARY)
    _, not_white = cv2.threshold(l_ch, 248, 255, cv2.THRESH_BINARY_INV)
    gray_rooms = cv2.bitwise_and(light_gray, not_white)

    # Combine tinted + gray rooms
    room_raw = cv2.bitwise_or(tinted, gray_rooms)

    # Remove background and black walls/lines
    room_raw = cv2.bitwise_and(room_raw, cv2.bitwise_not(bg_mask))
    room_raw = cv2.bitwise_and(room_raw, cv2.bitwise_not(dark_mask))

    # --- Morphological closing: bridge door gaps ---
    # Door gaps are typically 60–120px wide; use a conservative kernel
    # that closes gaps without merging adjacent rooms
    min_dim = min(h, w)
    door_close_size = max(7, min_dim // 120)
    door_close_size = min(door_close_size, 25)  # cap to avoid over-merging
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (door_close_size, door_close_size)
    )
    room_closed = cv2.morphologyEx(room_raw, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # --- Fill holes inside rooms (columns, text, etc.) ---
    room_filled = _fill_holes(room_closed)

    # --- Remove small noise artifacts ---
    min_area = int((min_dim ** 2) * 0.0005)  # 0.05% of image area
    room_clean = _remove_small_components(room_filled, min_area)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "dbg_tinted.png"), tinted)
        cv2.imwrite(str(debug_dir / "dbg_room_raw.png"), room_raw)
        cv2.imwrite(str(debug_dir / "dbg_room_closed.png"), room_closed)
        cv2.imwrite(str(debug_dir / "dbg_room_clean.png"), room_clean)
        logger.debug("Debug masks saved to %s", debug_dir)

    return room_clean


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside binary mask regions using floodFill."""
    flood = mask.copy()
    h, w = flood.shape
    seed_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, seed_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, flood_inv)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


# ---------------------------------------------------------------------------
# Contour extraction & geometry post-processing
# ---------------------------------------------------------------------------

def extract_room_polygons(
    mask: np.ndarray,
    debug_dir: Optional[Path] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Find contours of individual rooms from binary mask.
    Returns list of (contour, hierarchy_row) tuples for top-level contours only.
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if not contours:
        logger.warning("No contours found in room mask.")
        return []

    # Filter by minimum contour area
    min_dim = min(mask.shape[:2])
    min_area = int((min_dim ** 2) * 0.001)
    max_area = mask.shape[0] * mask.shape[1] * 0.9  # exclude full-image blobs

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area and len(c) >= 4:
            valid.append(c)

    logger.info("Extracted %d room contour candidates.", len(valid))

    if debug_dir:
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, valid, -1, (0, 200, 0), 2)
        cv2.imwrite(str(debug_dir / "dbg_contours.png"), vis)

    return valid


def contour_to_shapely(contour: np.ndarray) -> Optional[Polygon]:
    """Convert OpenCV contour to Shapely Polygon."""
    pts = contour.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = make_valid(poly)
    if isinstance(poly, MultiPolygon):
        # Take the largest component
        poly = max(poly.geoms, key=lambda p: p.area)
    if poly.area < 1:
        return None
    return poly


def simplify_polygon(
    poly: Polygon,
    image_diag: float,
    aggressive: bool = False,
) -> Polygon:
    """
    Simplify polygon geometry:
      1. Shapely simplify (Douglas-Peucker) to remove micro-segments.
      2. Remove door-like spurs (short protruding segments).
      3. Snap near-collinear points to straighten walls.
    """
    # Tolerance proportional to image size (~0.3% of diagonal)
    tol = image_diag * (0.005 if aggressive else 0.003)
    simplified = poly.simplify(tol, preserve_topology=True)

    if simplified.is_empty or not simplified.is_valid:
        simplified = poly

    # Remove door spurs: very short "tab" protrusions
    simplified = _remove_door_spurs(simplified, image_diag)

    # Straighten near-collinear vertices
    simplified = _straighten_walls(simplified, angle_threshold_deg=4.0)

    if not simplified.is_valid:
        simplified = make_valid(simplified)
    if isinstance(simplified, MultiPolygon):
        simplified = max(simplified.geoms, key=lambda p: p.area)

    return simplified


def _remove_door_spurs(poly: Polygon, image_diag: float) -> Polygon:
    """
    Remove small rectangular protrusions (door arcs / door openings).
    Heuristic: a spur is a run of points that goes out and comes back
    within a short distance, creating a short convex notch.
    """
    coords = list(poly.exterior.coords[:-1])
    n = len(coords)
    if n < 6:
        return poly

    max_spur_len = image_diag * 0.02  # 2% of diagonal = typical door width

    cleaned = []
    skip_until = -1
    for i in range(n):
        if i <= skip_until:
            continue
        cleaned.append(coords[i])
        # Check if a spur starts here: look ahead for a loop that closes nearby
        p0 = np.array(coords[i])
        for j in range(i + 2, min(i + 8, n)):
            pj = np.array(coords[j % n])
            dist = np.linalg.norm(pj - p0)
            if dist < max_spur_len:
                # Likely a spur between i and j — skip it
                skip_until = j - 1
                break

    if len(cleaned) < 3:
        return poly
    try:
        result = Polygon(cleaned)
        if result.is_valid and result.area > poly.area * 0.5:
            return result
    except Exception:
        pass
    return poly


def _straighten_walls(poly: Polygon, angle_threshold_deg: float = 4.0) -> Polygon:
    """
    Remove near-collinear vertices: if the angle at a vertex is less than
    threshold degrees from 180°, remove the vertex (straighten the wall).
    """
    coords = list(poly.exterior.coords[:-1])
    n = len(coords)
    if n < 4:
        return poly

    threshold_rad = math.radians(angle_threshold_deg)
    result = []

    for i in range(n):
        prev_pt = np.array(coords[(i - 1) % n], dtype=float)
        curr_pt = np.array(coords[i], dtype=float)
        next_pt = np.array(coords[(i + 1) % n], dtype=float)

        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-9 or n2 < 1e-9:
            continue  # degenerate point

        cos_a = np.dot(v1, v2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angle = math.acos(cos_a)  # angle at vertex
        # Straight wall → angle ≈ π; spur → angle ≪ π
        if abs(angle - math.pi) < threshold_rad:
            continue  # skip near-collinear vertex
        result.append(tuple(curr_pt))

    if len(result) < 3:
        return poly
    try:
        new_poly = Polygon(result)
        if new_poly.is_valid and new_poly.area > poly.area * 0.8:
            return new_poly
    except Exception:
        pass
    return poly


def polygon_to_lot(
    poly: Polygon,
    lot_id: str,
    meta: Optional[LotMeta],
) -> LotPolygon:
    """Build a LotPolygon dataclass from Shapely geometry + metadata."""
    exterior = list(poly.exterior.coords[:-1])
    pts = [[round(x, 2), round(y, 2)] for x, y in exterior]

    minx, miny, maxx, maxy = poly.bounds
    bbox = [round(minx, 2), round(miny, 2), round(maxx - minx, 2), round(maxy - miny, 2)]

    cx, cy = poly.centroid.x, poly.centroid.y
    centroid = [round(cx, 2), round(cy, 2)]

    # Suspiciousness checks
    suspicious = False
    suspicious_reason = ""
    convex_hull_area = poly.convex_hull.area
    if convex_hull_area > 0:
        convexity = poly.area / convex_hull_area
        if convexity < 0.4:
            suspicious = True
            suspicious_reason = f"low convexity ratio {convexity:.2f}"

    aspect = max(maxx - minx, maxy - miny) / (min(maxx - minx, maxy - miny) + 1e-9)
    if aspect > 20:
        suspicious = True
        suspicious_reason = (suspicious_reason + f"; extreme aspect ratio {aspect:.1f}").lstrip("; ")

    if len(pts) < 3:
        suspicious = True
        suspicious_reason = (suspicious_reason + "; degenerate polygon").lstrip("; ")

    return LotPolygon(
        lot_id=lot_id,
        polygon_points=pts,
        bbox=bbox,
        centroid=centroid,
        area_px=round(poly.area, 1),
        status=meta.status if meta else "",
        label=meta.label if meta else lot_id,
        suspicious=suspicious,
        suspicious_reason=suspicious_reason,
    )


# ---------------------------------------------------------------------------
# Lot ID assignment
# ---------------------------------------------------------------------------

def assign_lot_ids(
    polygons_shapely: list[Polygon],
    mapping: dict[str, LotMeta],
    overlay: np.ndarray,
    use_ocr: bool = False,
) -> list[tuple[Polygon, str, Optional[LotMeta]]]:
    """
    Assign lot_id to each detected polygon.

    Priority:
      1. Mapping file: match by spatial proximity of centroid to provided grid,
         or — if mapping contains coordinates — spatial overlap.
         (Without coordinates in mapping, we use centroid ordering heuristic.)
      2. OCR fallback.
      3. Auto-name: unknown_NN.
    """
    results: list[tuple[Polygon, str, Optional[LotMeta]]] = []
    used_ids: set[str] = set()
    unknown_counter = 0

    # Sort polygons top-left to bottom-right for stable assignment
    sorted_polys = sorted(
        polygons_shapely,
        key=lambda p: (round(p.centroid.y / 50), round(p.centroid.x / 50)),
    )

    # Build unmatched mapping pool
    unmatched_mapping = dict(mapping)

    for poly in sorted_polys:
        cx, cy = poly.centroid.x, poly.centroid.y

        assigned_id: Optional[str] = None
        assigned_meta: Optional[LotMeta] = None

        # --- Strategy 1: OCR inside polygon bounding box ---
        if use_ocr and not assigned_id:
            ocr_id = _ocr_polygon_region(overlay, poly)
            if ocr_id and ocr_id in unmatched_mapping:
                assigned_id = ocr_id
                assigned_meta = unmatched_mapping.pop(ocr_id)
                used_ids.add(assigned_id)
                logger.debug("OCR matched: %s at (%.0f, %.0f)", assigned_id, cx, cy)

        # --- Strategy 2: Mapping with OCR partial match ---
        if use_ocr and not assigned_id:
            ocr_id = _ocr_polygon_region(overlay, poly)
            if ocr_id:
                # Try fuzzy match
                for mid in list(unmatched_mapping.keys()):
                    if ocr_id.replace(" ", "") == mid.replace("-", "").replace(" ", ""):
                        assigned_id = mid
                        assigned_meta = unmatched_mapping.pop(mid)
                        used_ids.add(assigned_id)
                        break

        # --- Strategy 3: Auto-assign from mapping in reading order ---
        if not assigned_id and unmatched_mapping:
            # Simple greedy: take the next unmapped lot_id in insertion order
            # In practice, use coordinate-based matching when mapping has coords
            next_id = next(iter(unmatched_mapping))
            assigned_id = next_id
            assigned_meta = unmatched_mapping.pop(next_id)
            used_ids.add(assigned_id)
            logger.debug("Auto-assigned: %s", assigned_id)

        # --- Fallback: unknown ---
        if not assigned_id:
            unknown_counter += 1
            assigned_id = f"unknown_{unknown_counter:02d}"
            logger.warning("No lot_id found for polygon at (%.0f, %.0f) → %s", cx, cy, assigned_id)

        results.append((poly, assigned_id, assigned_meta))

    return results


def _ocr_polygon_region(overlay: np.ndarray, poly: Polygon) -> Optional[str]:
    """
    OCR the bounding box of a polygon and extract a lot_id pattern.
    Requires pytesseract to be installed.
    """
    try:
        import pytesseract
        import re

        minx, miny, maxx, maxy = [int(v) for v in poly.bounds]
        pad = 5
        h, w = overlay.shape[:2]
        roi = overlay[
            max(0, miny - pad): min(h, maxy + pad),
            max(0, minx - pad): min(w, maxx + pad),
        ]
        if roi.size == 0:
            return None

        # Preprocess for OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(
            thresh,
            config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
        )
        # Match patterns like F2-401, R-101, OF-203
        pattern = re.compile(r"\b([A-Z]{1,3}[-_]?\d{1,4})\b")
        matches = pattern.findall(text.upper())
        if matches:
            return matches[0]
    except ImportError:
        logger.debug("pytesseract not available; OCR fallback skipped.")
    except Exception as e:
        logger.debug("OCR error: %s", e)
    return None


# ---------------------------------------------------------------------------
# SVG generation
# ---------------------------------------------------------------------------

def _points_to_svg_d(points: list[list[float]]) -> str:
    """Convert polygon points to SVG path 'd' attribute."""
    if not points:
        return ""
    parts = [f"M {points[0][0]},{points[0][1]}"]
    for pt in points[1:]:
        parts.append(f"L {pt[0]},{pt[1]}")
    parts.append("Z")
    return " ".join(parts)


def generate_lots_svg(
    lots: list[LotPolygon],
    width: int,
    height: int,
) -> str:
    """Generate clean SVG with polygons only (no background)."""
    lines = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
        f'  <title>Floor Plan Lots</title>',
        f'  <g id="lots">',
    ]

    for lot in lots:
        d = _points_to_svg_d(lot.polygon_points)
        attrs = [
            f'id="{lot.lot_id}"',
            f'data-lot-id="{lot.lot_id}"',
        ]
        if lot.status:
            attrs.append(f'data-status="{lot.status}"')
        if lot.area_px:
            attrs.append(f'data-area-px="{lot.area_px}"')
        if lot.label:
            attrs.append(f'data-label="{lot.label}"')

        # Default neutral styling for overlay usage
        attrs += [
            'fill="rgba(100,160,200,0.3)"',
            'stroke="#2266aa"',
            'stroke-width="1.5"',
            'class="lot-polygon"',
        ]

        lines.append(f'    <path {" ".join(attrs)} d="{d}"/>')

    lines += ["  </g>", "</svg>"]
    return "\n".join(lines)


def generate_preview_svg(
    lots: list[LotPolygon],
    plan_image: np.ndarray,
    width: int,
    height: int,
) -> str:
    """Generate SVG with embedded base64 plan background + polygons."""
    # Encode plan image as base64
    success, buf = cv2.imencode(".png", plan_image)
    if not success:
        raise RuntimeError("Failed to encode plan image for SVG preview.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    lines = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
        f'  <title>Floor Plan Preview</title>',
        f'  <image id="plan-background" '
        f'x="0" y="0" width="{width}" height="{height}" '
        f'href="data:image/png;base64,{b64}" preserveAspectRatio="none"/>',
        f'  <g id="lots" opacity="0.6">',
    ]

    for lot in lots:
        d = _points_to_svg_d(lot.polygon_points)
        color = _status_color(lot.status)
        attrs = [
            f'id="preview-{lot.lot_id}"',
            f'data-lot-id="{lot.lot_id}"',
            f'fill="{color}"',
            'stroke="#ffffff"',
            'stroke-width="1.5"',
            'class="lot-polygon"',
        ]
        if lot.status:
            attrs.append(f'data-status="{lot.status}"')
        lines.append(f'    <path {" ".join(attrs)} d="{d}"/>')

        # Add centroid label
        cx, cy = lot.centroid
        label = lot.label or lot.lot_id
        lines.append(
            f'    <text x="{cx}" y="{cy}" '
            f'font-size="11" text-anchor="middle" dominant-baseline="middle" '
            f'fill="#ffffff" font-weight="bold" '
            f'style="text-shadow:0 1px 2px rgba(0,0,0,0.8);">{label}</text>'
        )

    lines += ["  </g>", "</svg>"]
    return "\n".join(lines)


def _status_color(status: str) -> str:
    palette = {
        "available": "rgba(72,199,116,0.55)",
        "sold": "rgba(220,76,76,0.55)",
        "reserved": "rgba(255,165,0,0.55)",
        "": "rgba(100,160,220,0.45)",
    }
    return palette.get(status.lower(), "rgba(100,160,220,0.45)")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def generate_lots_json(
    lots: list[LotPolygon],
    floor_id: str,
) -> dict:
    """Build the lots.json output structure."""
    return {
        "floor_id": floor_id,
        "lots": [
            {
                "lot_id": lot.lot_id,
                "polygon_points": lot.polygon_points,
                "bbox": lot.bbox,
                "centroid": lot.centroid,
                "area_px": lot.area_px,
                "status": lot.status,
                "label": lot.label,
            }
            for lot in lots
        ],
    }


def generate_validation_report(
    lots: list[LotPolygon],
    mapping: dict[str, LotMeta],
    report: ValidationReport,
) -> dict:
    """Build the validation_report.json output structure."""
    matched_ids = {lot.lot_id for lot in lots if not lot.lot_id.startswith("unknown_")}
    mapping_ids = set(mapping.keys())

    report.lots_matched = len(matched_ids & mapping_ids)
    report.lots_unmatched = len(mapping_ids - matched_ids)
    report.unmatched_lot_ids = sorted(mapping_ids - matched_ids)
    report.unknown_polygons = sum(1 for lot in lots if lot.lot_id.startswith("unknown_"))

    report.suspicious_polygons = [
        {
            "lot_id": lot.lot_id,
            "reason": lot.suspicious_reason,
            "area_px": lot.area_px,
            "centroid": lot.centroid,
        }
        for lot in lots
        if lot.suspicious
    ]

    return {
        "total_regions_found": report.total_regions_found,
        "lots_matched": report.lots_matched,
        "lots_unmatched": report.lots_unmatched,
        "unmatched_lot_ids": report.unmatched_lot_ids,
        "unknown_polygons": report.unknown_polygons,
        "suspicious_polygons": report.suspicious_polygons,
        "warnings": report.warnings,
        "errors": report.errors,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    plan_path: Path,
    overlay_path: Path,
    mapping_path: Optional[Path],
    out_dir: Path,
    use_ocr: bool = False,
    debug: bool = False,
) -> ValidationReport:
    """
    Full processing pipeline.

    Returns ValidationReport with summary statistics.
    """
    report = ValidationReport()
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug" if debug else None

    # --- Load images ---
    plan_img = load_image(plan_path)
    overlay_img = load_image(overlay_path)
    overlay_img = align_images(plan_img, overlay_img)

    h, w = plan_img.shape[:2]
    image_diag = math.sqrt(w ** 2 + h ** 2)
    floor_id = plan_path.stem

    # --- Load mapping ---
    mapping: dict[str, LotMeta] = {}
    if mapping_path:
        try:
            mapping = load_mapping(mapping_path)
        except Exception as e:
            msg = f"Failed to load mapping: {e}"
            logger.error(msg)
            report.errors.append(msg)

    if not mapping and not use_ocr:
        report.warnings.append(
            "No mapping file provided and OCR fallback is disabled. "
            "Lots will be named unknown_NN."
        )

    # --- Build room mask ---
    logger.info("Building room mask...")
    try:
        room_mask = build_room_mask(overlay_img, debug_dir=debug_dir)
    except Exception as e:
        msg = f"Room mask generation failed: {e}"
        logger.error(msg)
        report.errors.append(msg)
        raise

    # --- Extract contours ---
    logger.info("Extracting room contours...")
    contours = extract_room_polygons(room_mask, debug_dir=debug_dir)

    if not contours:
        msg = "No room regions detected. Check overlay image and color fills."
        logger.error(msg)
        report.errors.append(msg)
        raise RuntimeError(msg)

    report.total_regions_found = len(contours)
    logger.info("Found %d room regions.", len(contours))

    # --- Convert to Shapely + simplify ---
    logger.info("Post-processing geometries...")
    valid_polys: list[Polygon] = []
    for c in contours:
        poly = contour_to_shapely(c)
        if poly is None:
            report.warnings.append("Skipped degenerate contour.")
            continue
        poly = simplify_polygon(poly, image_diag)
        if poly and not poly.is_empty:
            valid_polys.append(poly)

    if not valid_polys:
        msg = "All polygons were invalid after post-processing."
        logger.error(msg)
        report.errors.append(msg)
        raise RuntimeError(msg)

    logger.info("Valid polygons after post-processing: %d", len(valid_polys))

    # --- Assign lot IDs ---
    logger.info("Assigning lot IDs...")
    assigned = assign_lot_ids(valid_polys, mapping, overlay_img, use_ocr=use_ocr)

    # --- Build LotPolygon objects ---
    lots: list[LotPolygon] = []
    for poly, lot_id, meta in assigned:
        lot = polygon_to_lot(poly, lot_id, meta)
        lots.append(lot)

    # --- Write outputs ---
    logger.info("Writing output files to %s...", out_dir)

    # lots.svg
    svg_clean = generate_lots_svg(lots, w, h)
    (out_dir / "lots.svg").write_text(svg_clean, encoding="utf-8")
    logger.info("Wrote lots.svg")

    # lots_preview.svg
    svg_preview = generate_preview_svg(lots, plan_img, w, h)
    (out_dir / "lots_preview.svg").write_text(svg_preview, encoding="utf-8")
    logger.info("Wrote lots_preview.svg")

    # lots.json
    lots_json = generate_lots_json(lots, floor_id)
    (out_dir / "lots.json").write_text(
        json.dumps(lots_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote lots.json")

    # validation_report.json
    val_report = generate_validation_report(lots, mapping, report)
    (out_dir / "validation_report.json").write_text(
        json.dumps(val_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote validation_report.json")

    # Summary
    logger.info(
        "Done. Lots: %d total, %d matched, %d unmatched, %d unknown, %d suspicious.",
        len(lots),
        report.lots_matched,
        report.lots_unmatched,
        report.unknown_polygons,
        len(report.suspicious_polygons),
    )

    if report.unmatched_lot_ids:
        logger.warning("Unmatched lot_ids from mapping: %s", report.unmatched_lot_ids)
    if report.suspicious_polygons:
        logger.warning(
            "Suspicious polygons: %s",
            [s["lot_id"] for s in report.suspicious_polygons],
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Floor Plan Vectorizer — converts raster floor plans to SVG polygons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with mapping
  python vectorize_floor.py --plan plan.png --overlay overlay.png --mapping lots.csv --out-dir ./output

  # Same image for plan and overlay
  python vectorize_floor.py --plan plan.png --mapping lots.json --out-dir ./output

  # With OCR fallback and debug output
  python vectorize_floor.py --plan plan.png --overlay overlay.png --ocr-fallback --debug --out-dir ./output
        """,
    )
    p.add_argument("--plan", required=True, type=Path, help="Path to floor plan image (PNG/JPG)")
    p.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help="Path to overlay image with colored fills (defaults to --plan if omitted)",
    )
    p.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Path to lot mapping CSV or JSON file",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    p.add_argument(
        "--ocr-fallback",
        action="store_true",
        help="Enable OCR fallback for lot ID detection (requires pytesseract)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate debug images to output/debug/",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return p


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Validate inputs
    plan_path: Path = args.plan
    if not plan_path.exists():
        logger.error("Plan image not found: %s", plan_path)
        return 1

    overlay_path: Path = args.overlay if args.overlay else plan_path
    if not overlay_path.exists():
        logger.error("Overlay image not found: %s", overlay_path)
        return 1

    mapping_path: Optional[Path] = args.mapping
    if mapping_path and not mapping_path.exists():
        logger.error("Mapping file not found: %s", mapping_path)
        return 1

    try:
        report = run_pipeline(
            plan_path=plan_path,
            overlay_path=overlay_path,
            mapping_path=mapping_path,
            out_dir=args.out_dir,
            use_ocr=args.ocr_fallback,
            debug=args.debug,
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        return 2

    # Exit code 0 = success, 1 = partial (warnings), 2 = failure
    if report.errors:
        return 2
    if report.warnings or report.unmatched_lot_ids or report.suspicious_polygons:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
