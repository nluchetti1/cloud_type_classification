#!/usr/bin/env python3
"""
CloudScope - model cloud-type classification over the Florida spaceport corridor.

Classifies every grid column of an HRRR forecast into one cloud type and renders a map
per forecast hour, plus a manifest the viewer reads.

WHY HYDROMETEORS, NOT RELATIVE HUMIDITY
    An RH threshold is only a proxy for "is there cloud here", and it is a bad one aloft:
    at cirrus temperatures the dewpoint depression collapses, so a model with any moist bias
    near the tropopause paints phantom cloud. HRRR carries the condensate fields directly
    (CLMR = cloud water, CIMIXR = cloud ice), so this classifies on what the model actually
    predicts is in the air, and gets the liquid/ice split for free.

WHY S3, NOT THE NOMADS FILTER
    NOMADS' grib_filter would crop server-side and cost far fewer bytes, but it rate-limits
    per IP and answers with "302 Over Rate Limit" once a runner has been busy. S3 has no such
    limit. The cost is that a byte-ranged HRRR message is the whole CONUS grid, cropped here
    after download.
"""

import datetime
import json
import logging
import os
import re

import numpy as np
import pygrib
import requests
from scipy.ndimage import map_coordinates

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
HRRR_ROOT = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
OUT_DIR = "docs"                     # GitHub Pages serves from /docs
MAP_DIR = os.path.join(OUT_DIR, "maps")
CACHE_DIR = "_cache"

# Florida spaceport corridor
DOMAIN = {"lat_min": 26.5, "lat_max": 30.5, "lon_min": -82.5, "lon_max": -79.0}

SITES = {
    "LC-39A": (28.608, -80.604),
    "SLC-40": (28.562, -80.577),
    "SLC-41": (28.583, -80.583),
    "KXMR":   (28.468, -80.556),
    "KMLB":   (28.103, -80.645),
    "KDAB":   (29.183, -81.048),
}

# Every 50 mb is enough to resolve deck depth; every 25 would double the download for
# little classification benefit.
LEVELS_HPA = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550,
              500, 450, 400, 350, 300, 250, 200, 150]
ANVIL_LEVELS = [300, 250, 200, 150]          # steering layer for anvil debris

FORECAST_HOURS = list(range(1, 19))          # HRRR's standard 18-h reach
MAX_CYCLE_LOOKBACK_H = 6

# ---- classification thresholds (all tunable; see README) ----
QC_MIN = 1e-6          # kg/kg total condensate for "cloud present"
ICE_FRAC = 0.70        # >= this fraction ice -> treat the column as glaciated
DEEP_KFT = 8.0         # deck depth separating vertically-developed cu from layered cloud
CONV_DBZ = 40.0        # composite reflectivity marking a convective core
ANVIL_BASE_C = -20.0   # an anvil's base must be colder than this
ANVIL_NEAR_NM = 12.0   # ignore the parent core's own neighbourhood
ANVIL_FAR_NM = 100.0   # how far debris can stream from its source

CLEAR, CIRRUS, STRATIFORM, CUMULUS, ANVIL, CONVECTIVE = range(6)
CLASSES = [
    # ordered surface -> top, which is also how the viewer stacks the legend
    {"id": CLEAR,      "key": "clear",      "name": "Clear",      "color": "#0D1117"},
    {"id": STRATIFORM, "key": "stratiform", "name": "Stratiform", "color": "#5C7A99"},
    {"id": CUMULUS,    "key": "cumulus",    "name": "Cumulus",    "color": "#D9A441"},
    {"id": CONVECTIVE, "key": "convective", "name": "Convective", "color": "#A11D33"},
    {"id": ANVIL,      "key": "anvil",      "name": "Anvil",      "color": "#E2703A"},
    {"id": CIRRUS,     "key": "cirrus",     "name": "Cirrus",     "color": "#BFD3E6"},
]
PALETTE = [c["color"] for c in sorted(CLASSES, key=lambda c: c["id"])]


# --------------------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------------------
def _session():
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16,
                                                      max_retries=3))
    s.headers.update({"User-Agent": "CloudScope/1.0 (+github actions)"})
    return s


def _parse_idx(text):
    """NCEP .idx lines: num:startbyte:date:SHORTNAME:level:fcst:"""
    out, lines = [], [l for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        p = line.split(":")
        if len(p) < 5:
            continue
        start = int(p[1])
        end = None
        for nxt in lines[i + 1:]:
            q = nxt.split(":")
            if len(q) > 1 and q[1].isdigit() and int(q[1]) > start:
                end = int(q[1]) - 1
                break
        out.append({"short": p[3], "level": p[4], "start": start, "end": end})
    return out


def _merge(entries, gap=8192):
    """Collapse adjacent byte ranges. Messages for one variable sit contiguously, so this
    turns ~100 requests into a handful without pulling materially more data."""
    rngs = sorted(((e["start"], e["end"]) for e in entries), key=lambda x: x[0])
    merged = []
    for s, e in rngs:
        if merged and merged[-1][1] is not None and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _url(kind, date_str, cycle, fh):
    return f"{HRRR_ROOT}/hrrr.{date_str}/conus/hrrr.t{cycle}z.{kind}f{fh:02d}.grib2"


def find_cycle(sess):
    """Newest cycle whose f01 wrfprs index is posted."""
    now = datetime.datetime.now(datetime.timezone.utc)
    for back in range(MAX_CYCLE_LOOKBACK_H + 1):
        t = now - datetime.timedelta(hours=back)
        d, cc = t.strftime("%Y%m%d"), t.strftime("%H")
        try:
            r = sess.get(_url("wrfprs", d, cc, 1) + ".idx", timeout=15)
            if r.status_code == 200 and "CLMR" in r.text:
                return d, cc, t.replace(minute=0, second=0, microsecond=0)
        except Exception:
            pass
    return None, None, None


def fetch_hour(sess, date_str, cycle, fh):
    """Byte-range the fields needed for one forecast hour. Returns a local GRIB path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, f"h{cycle}z_f{fh:02d}.grib2")
    lvl_re = re.compile(r"^(\d+)\s*mb$")
    total = 0
    with open(local, "wb") as out:
        # isobaric fields from wrfprs
        r = sess.get(_url("wrfprs", date_str, cycle, fh) + ".idx", timeout=20)
        if r.status_code != 200:
            return None, 0
        want = []
        for e in _parse_idx(r.text):
            m = lvl_re.match(e["level"].strip())
            if not m:
                continue
            lv = int(m.group(1))
            if e["short"] in ("TMP", "CLMR", "CIMIXR") and lv in LEVELS_HPA:
                want.append(e)
            elif e["short"] in ("UGRD", "VGRD") and lv in ANVIL_LEVELS:
                want.append(e)
        if not want:
            return None, 0
        for s, e in _merge(want):
            rng = f"bytes={s}-{'' if e is None else e}"
            rr = sess.get(_url("wrfprs", date_str, cycle, fh), headers={"Range": rng}, timeout=90)
            if rr.status_code in (200, 206):
                out.write(rr.content)
                total += len(rr.content)
        # composite reflectivity from wrfsfc
        r2 = sess.get(_url("wrfsfc", date_str, cycle, fh) + ".idx", timeout=20)
        if r2.status_code == 200:
            refc = [e for e in _parse_idx(r2.text)
                    if e["short"] == "REFC" and "entire atmosphere" in e["level"]]
            for s, e in _merge(refc):
                rng = f"bytes={s}-{'' if e is None else e}"
                rr = sess.get(_url("wrfsfc", date_str, cycle, fh), headers={"Range": rng}, timeout=90)
                if rr.status_code in (200, 206):
                    out.write(rr.content)
                    total += len(rr.content)
    return local, total


# --------------------------------------------------------------------------------------
# Read + crop
# --------------------------------------------------------------------------------------
def read_fields(path):
    """Pull the cropped 3-D stacks out of one hour's GRIB."""
    lvl = {v: {} for v in ("TMP", "CLMR", "CIMIXR", "UGRD", "VGRD")}
    refc = lats = lons = None
    grbs = pygrib.open(path)
    for g in grbs:
        short = getattr(g, "shortName", "") or ""
        name = (getattr(g, "name", "") or "").lower()
        if lats is None:
            lats, lons = g.latlons()
        if short.upper() == "REFC" or "composite" in name:
            refc = np.asarray(g.values, dtype=float)
            continue
        try:
            if getattr(g, "typeOfLevel", "") != "isobaricInhPa":
                continue
            lv = int(g.level)
        except Exception:
            continue
        key = {"t": "TMP", "clwmr": "CLMR", "cimixr": "CIMIXR",
               "u": "UGRD", "v": "VGRD"}.get(short, short.upper())
        if key in lvl:
            lvl[key][lv] = np.asarray(g.values, dtype=float)
    grbs.close()
    if lats is None or not lvl["TMP"]:
        return None

    lons = np.where(lons > 180, lons - 360.0, lons)
    box = ((lats >= DOMAIN["lat_min"]) & (lats <= DOMAIN["lat_max"]) &
           (lons >= DOMAIN["lon_min"]) & (lons <= DOMAIN["lon_max"]))
    ys, xs = np.where(box)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = lambda a: a[y0:y1, x0:x1]

    levels = sorted(set(lvl["TMP"]) & set(lvl["CLMR"]), reverse=True)   # surface -> top
    if not levels:
        return None
    stack = lambda v: np.stack([crop(lvl[v][L]) for L in levels]) if lvl[v] else None
    tmpc = stack("TMP") - 273.15
    qliq = stack("CLMR")
    qice = (np.stack([crop(lvl["CIMIXR"][L]) for L in levels])
            if set(levels) <= set(lvl["CIMIXR"]) else np.zeros_like(qliq))

    av = [L for L in ANVIL_LEVELS if L in lvl["UGRD"] and L in lvl["VGRD"]]
    if av:
        u_anv = np.mean([crop(lvl["UGRD"][L]) for L in av], axis=0)
        v_anv = np.mean([crop(lvl["VGRD"][L]) for L in av], axis=0)
    else:
        u_anv = v_anv = np.zeros_like(tmpc[0])

    # Standard-atmosphere height per level. Only used for deck DEPTH, where a hypsometric
    # profile would change nothing that matters at an 8 kft threshold.
    z_kft = np.array([44330.0 * (1 - (L / 1013.25) ** 0.1903) / 304.8 for L in levels])
    hgt = z_kft[:, None, None] * np.ones_like(tmpc)

    return {"tmpc": tmpc, "qliq": qliq, "qice": qice, "hgt": hgt,
            "refc": crop(refc) if refc is not None else np.zeros_like(tmpc[0]),
            "u_anv": u_anv, "v_anv": v_anv,
            "lats": crop(lats), "lons": crop(lons), "levels": levels}


# --------------------------------------------------------------------------------------
# Classify
# --------------------------------------------------------------------------------------
def classify(f, dx_nm=1.6):
    """One cloud type per grid column. Returns (class_grid, diagnostics)."""
    qliq, qice, tmpc, hgt, refc = f["qliq"], f["qice"], f["tmpc"], f["hgt"], f["refc"]
    nlev, ny, nx = qliq.shape
    cloud = (qliq + qice) > QC_MIN
    any_cloud = cloud.any(axis=0)

    idx = np.arange(nlev)[:, None, None] * np.ones((1, ny, nx))
    lo_i = np.where(cloud, idx, nlev + 1).min(axis=0)
    hi_i = np.where(cloud, idx, -1).max(axis=0)

    def at(arr, ind):
        return np.take_along_axis(arr, np.clip(ind, 0, nlev - 1).astype(int)[None], axis=0)[0]

    base_kft, top_kft = at(hgt, lo_i), at(hgt, hi_i)
    base_c = at(tmpc, lo_i)
    depth = np.where(any_cloud, top_kft - base_kft, 0.0)

    ice_sum, liq_sum = qice.sum(axis=0), qliq.sum(axis=0)
    tot = ice_sum + liq_sum
    ice_frac = np.where(tot > 0, ice_sum / np.maximum(tot, 1e-12), 0.0)
    glaciated = ice_frac >= ICE_FRAC

    # Sample reflectivity UPWIND along the anvil-level flow: an anvil is debris that came
    # from somewhere, so the parent core has to be findable in that direction.
    spd = np.hypot(f["u_anv"], f["v_anv"])
    moving = spd > 0.5
    ux = np.where(moving, f["u_anv"] / np.maximum(spd, 1e-6), 0.0)
    uy = np.where(moving, f["v_anv"] / np.maximum(spd, 1e-6), 0.0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    upstream = np.zeros((ny, nx))
    for d_nm in np.arange(ANVIL_NEAR_NM, ANVIL_FAR_NM + 1, 8.0):
        step = d_nm / dx_nm
        samp = map_coordinates(refc, [yy - uy * step, xx - ux * step], order=1, mode="nearest")
        upstream = np.maximum(upstream, samp)

    out = np.full((ny, nx), CLEAR, dtype=np.uint8)
    out[any_cloud & glaciated] = CIRRUS
    out[any_cloud & ~glaciated] = STRATIFORM
    out[any_cloud & ~glaciated & (depth >= DEEP_KFT)] = CUMULUS
    out[any_cloud & glaciated & (base_c < ANVIL_BASE_C) &
        (refc < CONV_DBZ) & (upstream >= CONV_DBZ)] = ANVIL
    out[refc >= CONV_DBZ] = CONVECTIVE
    return out, {"depth": depth, "ice_frac": ice_frac, "upstream": upstream}


# --------------------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------------------
def render(cls, f, valid, cycle, fh, path):
    cmap = mcolors.ListedColormap(PALETTE)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(PALETTE) + 0.5, 1), len(PALETTE))
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(6.4, 6.4), dpi=130, facecolor="#0D1117")
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([DOMAIN["lon_min"], DOMAIN["lon_max"],
                   DOMAIN["lat_min"], DOMAIN["lat_max"]], crs=proj)
    ax.set_facecolor("#0D1117")
    ax.pcolormesh(f["lons"], f["lats"], cls, cmap=cmap, norm=norm,
                  shading="nearest", transform=proj, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#5A6B7D",
                   linewidth=0.8, zorder=4)
    ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines",
                                                "10m", facecolor="none"),
                   edgecolor="#3C4A59", linewidth=0.5, zorder=4)
    for name, (la, lo) in SITES.items():
        ax.plot(lo, la, marker="+", markersize=7, markeredgewidth=1.4,
                color="#F2F4F0", transform=proj, zorder=6)
        ax.text(lo + 0.045, la + 0.03, name, fontsize=6.2, color="#F2F4F0",
                family="monospace", transform=proj, zorder=6)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor="#0D1117")
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    os.makedirs(MAP_DIR, exist_ok=True)
    sess = _session()
    date_str, cycle, cyc_dt = find_cycle(sess)
    if not cycle:
        logging.error("No HRRR cycle available; leaving the previous run in place.")
        return
    logging.info(f"HRRR cycle {date_str} {cycle}z")

    frames, bytes_total = [], 0
    for fh in FORECAST_HOURS:
        path, n = fetch_hour(sess, date_str, cycle, fh)
        bytes_total += n
        if not path or n == 0:
            logging.warning(f"f{fh:02d}: no data")
            continue
        try:
            f = read_fields(path)
            if f is None:
                logging.warning(f"f{fh:02d}: fields missing")
                continue
            cls, diag = classify(f)
            valid = cyc_dt + datetime.timedelta(hours=fh)
            png = f"maps/cloudtype_{cycle}z_f{fh:02d}.png"
            render(cls, f, valid, cycle, fh, os.path.join(OUT_DIR, png))
            counts = {c["key"]: int((cls == c["id"]).sum()) for c in CLASSES}
            frames.append({"fh": fh, "valid": valid.strftime("%Y-%m-%dT%H:%MZ"),
                           "valid_short": valid.strftime("%d/%HZ"), "image": png,
                           "counts": counts})
            logging.info(f"f{fh:02d} valid {valid:%d/%HZ}: " +
                         " ".join(f"{k}={v}" for k, v in counts.items() if v))
        finally:
            if os.path.exists(path):
                os.remove(path)

    if not frames:
        logging.error("No frames rendered; manifest not rewritten.")
        return

    # prune PNGs from older cycles
    keep = {os.path.basename(fr["image"]) for fr in frames}
    for fn in os.listdir(MAP_DIR):
        if fn.endswith(".png") and fn not in keep:
            os.remove(os.path.join(MAP_DIR, fn))

    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "model": "HRRR", "cycle": f"{date_str} {cycle}Z",
        "domain": DOMAIN, "classes": CLASSES, "frames": frames,
        "thresholds": {"qc_min_kgkg": QC_MIN, "ice_fraction": ICE_FRAC,
                       "deep_kft": DEEP_KFT, "convective_dbz": CONV_DBZ,
                       "anvil_base_c": ANVIL_BASE_C, "anvil_reach_nm": ANVIL_FAR_NM},
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh_:
        json.dump(manifest, fh_, indent=1)
    logging.info(f"{len(frames)} frames, {bytes_total/1024/1024:.0f} MB transferred.")


if __name__ == "__main__":
    main()
