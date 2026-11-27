#!/usr/bin/env python3
"""
Build tree-data.js for all Busan bus stops in ecobit-solar-dashboard
using Meta & WRI High-Resolution Canopy Height Maps v2 (DINOv3, 1m, CC BY 4.0).

Usage:
    uv run --with rasterio --with pyproj --with numpy --with scipy scripts/build_tree_data.py

Outputs:
    tree-data.js  (gzip + base64 binary arrays loaded by index.html)
"""

import base64
import datetime
import gzip
import json
import math
import os
import re
import struct
import sys
import time
import urllib.request
from collections import Counter

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, ".cache", "chmv2")
STATION_JS = os.path.join(ROOT, "station-data.js")
OUT_JS = os.path.join(ROOT, "tree-data.js")

S3_BASE = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"
QUADKEY_ZOOM = 10
PANEL_HEIGHT_M = 3.0
ANALYSIS_RADIUS_M = 30.0
CLUSTER_RADIUS_M = 28.0
MAX_CLUSTERS_PER_STOP = 12


def lat_lon_to_quadkey(lat: float, lon: float, zoom: int = QUADKEY_ZOOM) -> str:
    sin_lat = math.sin(math.radians(lat))
    x = int(((lon + 180.0) / 360.0) * (1 << zoom))
    y = int((0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * (1 << zoom))
    qk = []
    for i in range(zoom, 0, -1):
        d = 0
        m = 1 << (i - 1)
        if x & m:
            d += 1
        if y & m:
            d += 2
        qk.append(str(d))
    return "".join(qk)


def load_stops():
    with open(STATION_JS, "r", encoding="utf-8") as f:
        raw = f.read()
    eq = raw.index("=")
    payload_str = json.loads(raw[eq + 1 :].strip().rstrip(";"))
    payload = json.loads(payload_str)
    # Row: [id, name, lat, lon, ars, district, annual, winter, loss, detailed, ...]
    return payload["stops"]


def download_if_missing(url: str, dest: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"Downloading {url} -> {os.path.basename(dest)} ...", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ecobit-solar-dashboard/1.0"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            total = int(resp.headers.get("Content-Length", "0") or "0")
            done = 0
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
            os.replace(tmp, dest)
            print(f"  OK ({done / 1048576:.1f} MB)", flush=True)
            return True
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"  WARN: failed to download {url}: {e}", flush=True)
        return False


def point_in_ring(x: float, y: float, ring) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        x1, y1 = ring[j]
        x2, y2 = ring[i]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
        j = i
    return inside


def load_metadata_polygons(meta_path: str):
    if not os.path.exists(meta_path):
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    polys = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        acq = props.get("acq_date") or props.get("date") or ""
        if not acq:
            continue
        rings = []
        if geom.get("type") == "Polygon":
            rings = [geom["coordinates"][0]]
        elif geom.get("type") == "MultiPolygon":
            rings = [p[0] for p in geom["coordinates"] if p]
        for r in rings:
            xs = [p[0] for p in r]
            ys = [p[1] for p in r]
            polys.append((min(xs), min(ys), max(xs), max(ys), r, acq))
    return polys


def lookup_acq_date(lon: float, lat: float, polys) -> str:
    for minx, miny, maxx, maxy, ring, acq in polys:
        if minx <= lon <= maxx and miny <= lat <= maxy and point_in_ring(lon, lat, ring):
            return acq
    return ""


def precompute_sun_weights(lat_deg: float = 35.16, lon_deg: float = 129.07):
    """Precompute half-hourly solar altitude, 2-deg azimuth bin, and panel incidence weight
    for winter (Dec-Feb) and leaf-on (May-Oct) at Busan's representative lat/lon."""
    rad = math.pi / 180.0

    def sun_pos(month, minutes_kst):
        year = 2021 if month == 12 else 2022
        dt = datetime.datetime(year, month, 15, 0, 0, tzinfo=datetime.timezone.utc) + datetime.timedelta(
            minutes=minutes_kst - 540
        )
        jd = (dt - datetime.datetime(2000, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)).total_seconds() / 86400.0
        t = jd / 36525.0
        mean = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
        anom = (357.52911 + t * (35999.05029 - 0.0001537 * t)) * rad
        ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
        center = (
            math.sin(anom) * (1.914602 - t * (0.004817 + 0.000014 * t))
            + math.sin(2 * anom) * (0.019993 - 0.000101 * t)
            + math.sin(3 * anom) * 0.000289
        )
        omega = (125.04 - 1934.136 * t) * rad
        apparent = (mean + center - 0.00569 - 0.00478 * math.sin(omega)) * rad
        eps = (
            23
            + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
            + 0.00256 * math.cos(omega)
        ) * rad
        dec = math.asin(math.sin(eps) * math.sin(apparent))
        y = math.tan(eps / 2.0) ** 2
        l = mean * rad
        corr = (4.0 / rad) * (
            y * math.sin(2 * l)
            - 2 * ecc * math.sin(anom)
            + 4 * ecc * y * math.sin(anom) * math.cos(2 * l)
            - 0.5 * y * y * math.sin(4 * l)
            - 1.25 * ecc * ecc * math.sin(2 * anom)
        )
        utc_min = (minutes_kst - 540) % 1440
        hour = (((utc_min + corr + lon_deg * 4.0) % 1440.0) / 4.0 - 180.0) * rad
        phi = lat_deg * rad
        sine = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(hour)
        alt = math.asin(max(-1.0, min(1.0, sine))) / rad
        az = (
            math.atan2(math.sin(hour), math.cos(hour) * math.sin(phi) - math.tan(dec) * math.cos(phi)) / rad + 180.0
        ) % 360.0
        return alt, az

    def build_season(months):
        alts, bins, weights = [], [], []
        for m in months:
            for step in range(48):
                alt, az = sun_pos(m, step * 30)
                if alt <= 0:
                    continue
                inc = math.sin(alt * rad) * math.cos(30 * rad) + math.cos(alt * rad) * math.sin(30 * rad) * math.cos(
                    (az - 180.0) * rad
                )
                w = max(0.0, inc) * math.sin(alt * rad)
                if w > 0:
                    alts.append(alt)
                    bins.append(int(az / 2.0) % 180)
                    weights.append(w)
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()
        return np.array(alts, dtype=np.float32), np.array(bins, dtype=np.int32), weights

    return build_season([12, 1, 2]), build_season([5, 6, 7, 8, 9, 10])


def main():
    t0 = time.time()
    stops = load_stops()
    n = len(stops)
    print(f"Loaded {n:,} stops from station-data.js", flush=True)

    # Group stops by Zoom-10 QuadKey
    by_qk = {}
    for idx, s in enumerate(stops):
        lat, lon = float(s[2]), float(s[3])
        qk = lat_lon_to_quadkey(lat, lon, QUADKEY_ZOOM)
        by_qk.setdefault(qk, []).append((idx, lat, lon))

    print("Required CHMv2 Zoom-10 QuadKeys:", {k: len(v) for k, v in sorted(by_qk.items())}, flush=True)

    # Download tiles & metadata
    for qk in sorted(by_qk.keys()):
        download_if_missing(f"{S3_BASE}/chm/{qk}.tif", os.path.join(CACHE_DIR, f"{qk}.tif"))
        download_if_missing(f"{S3_BASE}/metadata/{qk}.geojson", os.path.join(CACHE_DIR, f"{qk}.geojson"))

    # Precompute solar weights
    (w_alts, w_bins, w_weights), (s_alts, s_bins, s_weights) = precompute_sun_weights()

    # Allocate output arrays
    risk_arr = np.zeros(n, dtype=np.uint8)
    max_h15_arr = np.zeros(n, dtype=np.uint8)
    south_max_h_arr = np.zeros(n, dtype=np.uint8)
    south_pct_arr = np.zeros(n, dtype=np.uint8)
    max_elev_arr = np.zeros(n, dtype=np.uint8)  # 0.5 deg units (0..180)
    winter_blk_arr = np.zeros(n, dtype=np.uint8)
    summer_blk_arr = np.zeros(n, dtype=np.uint8)
    acq_idx_arr = np.zeros(n, dtype=np.uint8)
    horizon_arr = np.zeros((n, 180), dtype=np.uint8)  # 0.5 deg units (0..180)

    acq_date_map = {"": 0}
    acq_dates = [""]
    clusters_by_stop = [[] for _ in range(n)]

    processed = 0
    for qk, items in sorted(by_qk.items()):
        tif_path = os.path.join(CACHE_DIR, f"{qk}.tif")
        meta_path = os.path.join(CACHE_DIR, f"{qk}.geojson")
        if not os.path.exists(tif_path):
            print(f"Skipping missing tile {qk} ({len(items)} stops)", flush=True)
            continue
        meta_polys = load_metadata_polygons(meta_path)
        with rasterio.open(tif_path) as src:
            to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            is_3857 = src.crs.to_epsg() == 3857
            pixel_area_base = abs(src.res[0] * src.res[1])

            for idx, lat, lon in items:
                cx, cy = to_src.transform(lon, lat)
                sf = 1.0 / math.cos(math.radians(lat)) if is_3857 else 1.0
                span = ANALYSIS_RADIUS_M * sf
                win = from_bounds(cx - span, cy - span, cx + span, cy + span, src.transform)
                arr = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)

                acq = lookup_acq_date(lon, lat, meta_polys)
                if acq not in acq_date_map:
                    acq_date_map[acq] = len(acq_dates)
                    acq_dates.append(acq)
                acq_idx_arr[idx] = acq_date_map[acq]

                if arr.max() < 1.0:
                    processed += 1
                    continue

                rows, cols = np.indices(arr.shape)
                wt = src.window_transform(win)
                dx_m = (wt.c + (cols + 0.5) * wt.a - cx) / sf
                dy_m = (wt.f + (rows + 0.5) * wt.e - cy) / sf
                dist_m = np.hypot(dx_m, dy_m)
                az_deg = (np.degrees(np.arctan2(dx_m, dy_m)) + 360.0) % 360.0

                within_30 = dist_m <= ANALYSIS_RADIUS_M
                within_15 = dist_m <= 15.0
                south_25 = (dist_m >= 1.0) & (dist_m <= 25.0) & (az_deg >= 90.0) & (az_deg <= 270.0)

                m15 = float(arr[within_15].max()) if within_15.any() else 0.0
                sm25 = float(arr[south_25].max()) if south_25.any() else 0.0
                spct = float((arr[south_25] >= PANEL_HEIGHT_M).mean() * 100.0) if south_25.any() else 0.0

                max_h15_arr[idx] = min(255, int(round(m15)))
                south_max_h_arr[idx] = min(255, int(round(sm25)))
                south_pct_arr[idx] = min(100, int(round(spct)))

                # Horizon profile (2 deg bins) for canopy taller than 3m panel
                above = (arr > PANEL_HEIGHT_M) & within_30
                h_deg = np.zeros(180, dtype=np.float32)
                if above.any():
                    eff_dist = np.maximum(dist_m[above], 0.6)
                    elev = np.degrees(np.arctan2(arr[above] - PANEL_HEIGHT_M, eff_dist))
                    bins = (az_deg[above] / 2.0).astype(np.int32) % 180
                    np.maximum.at(h_deg, bins, elev)
                    # Direct overhead canopy (within 1.5m of panel and >= 4m high)
                    if ((dist_m <= 1.5) & (arr >= 4.0)).any():
                        h_deg = np.maximum(h_deg, 75.0)
                    horizon_arr[idx] = np.clip(np.round(h_deg * 2.0), 0, 180).astype(np.uint8)

                south_elev = float(h_deg[45:136].max())  # 90 deg .. 270 deg
                max_elev_arr[idx] = min(180, int(round(south_elev * 2.0)))

                w_blk = float(w_weights[w_alts <= h_deg[w_bins]].sum() * 100.0)
                s_blk = float(s_weights[s_alts <= h_deg[s_bins]].sum() * 100.0)
                winter_blk_arr[idx] = min(100, int(round(w_blk)))
                summer_blk_arr[idx] = min(100, int(round(s_blk)))

                # Risk classification (0=미감지, 1=낮음, 2=주의, 3=높음)
                if s_blk >= 25.0 or w_blk >= 25.0 or spct >= 25.0 or (m15 >= 5.0 and south_elev >= 30.0):
                    risk_arr[idx] = 3
                elif s_blk >= 8.0 or w_blk >= 8.0 or spct >= 8.0 or south_elev >= 15.0:
                    risk_arr[idx] = 2
                elif m15 >= 3.0 or sm25 >= 3.0:
                    risk_arr[idx] = 1
                else:
                    risk_arr[idx] = 0

                # Extract compact 3D canopy clusters within 28m (h >= 3m)
                canopy_mask = (arr >= PANEL_HEIGHT_M) & (dist_m <= CLUSTER_RADIUS_M)
                if canopy_mask.any():
                    labeled, num_features = ndimage.label(canopy_mask)
                    px_area_m2 = pixel_area_base / (sf * sf)
                    blobs = []
                    for cid in range(1, num_features + 1):
                        cmask = labeled == cid
                        cnt = int(cmask.sum())
                        vals = arr[cmask]
                        h_p90 = float(np.percentile(vals, 90))
                        if cnt == 1 and h_p90 < 4.0:
                            continue
                        cdx = float(dx_m[cmask].mean())
                        cdy = float(dy_m[cmask].mean())
                        cdist = math.hypot(cdx, cdy)
                        rad_m = math.sqrt((cnt * px_area_m2) / math.pi)
                        rad_m = max(1.0, min(14.0, rad_m))
                        # Rank by shading importance (taller & closer & more southerly first)
                        score = (h_p90 - 2.0) * rad_m / max(3.0, cdist)
                        blobs.append((
                            score,
                            max(-30, min(30, int(round(cdx)))),
                            max(-30, min(30, int(round(cdy)))),
                            max(2, min(28, int(round(rad_m * 2.0)))),
                            max(3, min(60, int(round(h_p90)))),
                        ))
                    blobs.sort(key=lambda b: b[0], reverse=True)
                    clusters_by_stop[idx] = [(b[1], b[2], b[3], b[4]) for b in blobs[:MAX_CLUSTERS_PER_STOP]]

                processed += 1
                if processed % 2000 == 0:
                    print(f"  Processed {processed:,}/{n:,} stops ({time.time() - t0:.1f}s)", flush=True)

    print(f"Finished sampling {processed:,} stops in {time.time() - t0:.1f}s", flush=True)

    # Pack clusters into flat offsets (uint32) + clusters (int8 x4: dx, dy, rad_half_m, height_m)
    offsets = np.zeros(n + 1, dtype=np.uint32)
    flat_clusters = []
    for i, cl in enumerate(clusters_by_stop):
        offsets[i] = len(flat_clusters) // 4
        for dx_i, dy_i, r_i, h_i in cl:
            flat_clusters.extend([dx_i, dy_i, r_i, h_i])
    offsets[n] = len(flat_clusters) // 4
    clusters_arr = np.array(flat_clusters, dtype=np.int8)

    # Concatenate aligned binary payload (4-byte aligned, matching loadVworld() convention)
    parts = [
        ("risk", risk_arr),
        ("maxH15", max_h15_arr),
        ("southMaxH", south_max_h_arr),
        ("southPct", south_pct_arr),
        ("maxElev", max_elev_arr),
        ("winterBlk", winter_blk_arr),
        ("summerBlk", summer_blk_arr),
        ("acqIdx", acq_idx_arr),
        ("horizon", horizon_arr.reshape(-1)),
        ("clusterOffsets", offsets),
        ("clusters", clusters_arr),
    ]
    buf = bytearray()
    sizes = {}
    for name, arr in parts:
        pad = (-len(buf)) % 4
        if pad:
            buf.extend(b"\x00" * pad)
        buf.extend(arr.tobytes())
        sizes[name] = int(arr.size)

    gz = gzip.compress(bytes(buf), compresslevel=9)
    b64 = base64.b64encode(gz).decode("ascii")

    meta = {
        "version": 1,
        "source": "Meta & WRI Canopy Height Maps v2 (DINOv3)",
        "license": "CC BY 4.0",
        "generated": datetime.date.today().isoformat(),
        "count": n,
        "panelHeight": PANEL_HEIGHT_M,
        "radius": ANALYSIS_RADIUS_M,
        "acqDates": acq_dates,
        "arrays": sizes,
    }

    with open(OUT_JS, "w", encoding="utf-8", newline="\n") as f:
        f.write("// © Meta & World Resources Institute (WRI), Canopy Height Maps v2 (DINOv3), CC BY 4.0.\n")
        f.write("// Generated automatically by scripts/build_tree_data.py\n")
        f.write(f"window.treeMeta = {json.dumps(meta, ensure_ascii=False, separators=(',', ':'))};\n")
        f.write(f"document.getElementById('treeData').textContent = \"{b64}\";\n")

    # Summary statistics
    rc = Counter(risk_arr.tolist())
    labels = {0: "미감지(0)", 1: "낮음(1)", 2: "주의(2)", 3: "높음(3)"}
    print("\n=== Busan Tree Canopy Summary (10,043 stops) ===")
    for k in [3, 2, 1, 0]:
        print(f"  {labels[k]:10s}: {rc[k]:5,d} ({rc[k] * 100 / n:.1f}%)")
    print(f"Total 3D canopy clusters: {len(flat_clusters) // 4:,} (avg {len(flat_clusters) / 4 / n:.2f}/stop)")
    print(f"Raw binary: {len(buf) / 1024:.1f} KB -> gzip: {len(gz) / 1024:.1f} KB -> tree-data.js: {os.path.getsize(OUT_JS) / 1024:.1f} KB")
    print(f"Wrote {OUT_JS}")


if __name__ == "__main__":
    main()
