#!/usr/bin/env python3
"""
CloudScope - model cloud-type classification over the Florida spaceport corridor.

Classifies every grid column of an HRRR forecast into one cloud type and renders a map
per forecast hour, plus a manifest the viewer reads.

WHAT CHANGED (v2) AND WHY
    1. LAYERS, NOT COLUMNS. v1 measured cloud base at the lowest cloudy level and cloud top
       at the highest, so a 2 kft stratocumulus deck under unrelated cirrus became a 35 kft
       "deep" column and got painted cumulus. v2 segments each column into contiguous cloud
       layers and classifies the lowest (liquid) and highest (ice) layers separately.

    2. CONDENSATE PATH, NOT MIXING RATIO. A fixed 1e-6 kg/kg threshold is density-blind:
       the same mixing ratio at 200 mb is ~4x less condensate than at 900 mb. v2 integrates
       q * dp / g into g/m^2, which is what actually controls opacity, and lets the anvil /
       cirrus split fall out of ice water path (anvil 50+, aged debris 5-50, thin cirrus <5).

    3. ANVIL BY DETRAINMENT PHYSICS, NOT BASE TEMPERATURE. An anvil's base is set by where
       the tower detrains, and a thick or attached anvil routinely has a base warmer than
       -20 C, so v1's ANVIL_BASE_C threw those away. v2 keys on what an anvil actually is:
       an optically substantial, glaciated layer whose TOP sits at the homogeneous-freezing
       level or above (<= -38 C, i.e. it came out of the top of a tower, not a mid-level
       deck), traced back along its own layer-mean wind to a convective source within a
       physical advection time rather than a fixed distance in nm.

    4. ATTACHED vs DETACHED vs DEBRIS. The LLCC treats those differently, so the classifier
       does too: attached = core inside 10 nm; detached = core upwind within the trajectory
       reach; debris = same trajectory but the ice has thinned below anvil opacity.

    5. GRAUPEL AS AN UPDRAFT PROXY. Riming needs supercooled water and an updraft to hold it,
       so a graupel path aloft flags a convective core that reflectivity may not have caught
       yet (or that sits under a beam-blind 40 dBZ threshold).

    6. REAL HEIGHTS. HGT is pulled from the file instead of inverting the standard atmosphere,
       so layer depth is the model's own, not an assumption.

WHY S3, NOT THE NOMADS FILTER
    NOMADS' grib_filter crops server-side and costs far fewer bytes, but rate-limits per IP
    and answers "302 Over Rate Limit" once a runner has been busy. S3 has no such limit. The
    cost is that a byte-ranged HRRR message is the whole CONUS grid, cropped after download.
    v2 pulls ~2x v1's bytes; per-variable level sets below claw some of that back by only
    asking for each field where it can physically be non-zero.
"""

import datetime
import json
import logging
import os
import re

import numpy as np
import pygrib
import requests
from scipy.ndimage import map_coordinates, maximum_filter, uniform_filter

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

DOMAIN = {"lat_min": 26.5, "lat_max": 30.5, "lon_min": -82.5, "lon_max": -79.0}

SITES = {
    "LC-39A": (28.608, -80.604),
    "SLC-40": (28.562, -80.577),
    "SLC-41": (28.583, -80.583),
    "KXMR":   (28.468, -80.556),
    "KMLB":   (28.103, -80.645),
    "KDAB":   (29.183, -81.048),
}

DX_KM = 3.0                                   # HRRR CONUS grid spacing

# 50 mb resolves deck depth to ~1.5 kft, which is finer than any threshold here.
LEVELS_HPA = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550,
              500, 450, 400, 350, 300, 250, 200, 150]

# Only ask for each field where it can physically be non-zero. Liquid above 400 mb in HRRR
# is numerical dust; cloud ice below 700 mb likewise; graupel lives in the riming layer.
LEVEL_SETS = {
    "HGT":    LEVELS_HPA,
    "TMP":    LEVELS_HPA,
    "UGRD":   LEVELS_HPA,
    "VGRD":   LEVELS_HPA,
    "CLMR":   [L for L in LEVELS_HPA if L >= 400],
    "CIMIXR": [L for L in LEVELS_HPA if L <= 700],
    "SNMR":   [L for L in LEVELS_HPA if L <= 700],
    "GRLE":   [L for L in LEVELS_HPA if 200 <= L <= 900],
}

FORECAST_HOURS = list(range(1, 19))
MAX_CYCLE_LOOKBACK_H = 6

# ---- classification thresholds (all tunable; see README) ----
LAYER_PATH_MIN = 0.20   # g/m^2 of condensate in one layer to call it cloudy
GLACIATED_C    = -38.0  # homogeneous freezing: a top this cold is cirriform by construction
ICE_FRAC       = 0.80   # ice share WITHIN the layer for it to count as glaciated
ANVIL_IWP      = 50.0   # g/m^2 - optically substantial ice; anvils run 100-1000
DEBRIS_IWP     = 5.0    # g/m^2 - aged, thinning ice still worth flagging
CONV_DBZ       = 40.0   # composite reflectivity marking a convective core
GRAUPEL_CONV   = 200.0  # g/m^2 of graupel: riming implies an updraft holding supercooled water
ATTACH_NM      = 10.0   # core this close and the anvil is attached, not advected
ANVIL_TAU_H    = 3.0    # ice lifetime; trajectory reach = layer wind x this, not a fixed nm
ANVIL_MAX_NM   = 150.0  # cap, so a 90 kt jet doesn't sweep the whole domain
TCU_TOP_C      = -10.0  # liquid-based layer glaciating at its top
CU_DEPTH_KFT   = 3.0    # depth separating cumuliform from a layered deck
CU_TEX_KFT     = 1.2    # or lumpiness: sigma of cloud-top height over ~15 km

(CLEAR, STRATIFORM, CUMULUS, TCU, CONVECTIVE,
 ANVIL_ATT, ANVIL_DET, DEBRIS, CIRRUS) = range(9)

CLASSES = [
    {"id": CLEAR,      "key": "clear",      "name": "Clear",           "color": "#0D1117"},
    {"id": STRATIFORM, "key": "stratiform", "name": "Stratiform",      "color": "#5C7A99"},
    {"id": CUMULUS,    "key": "cumulus",    "name": "Cumulus",         "color": "#E3C36A"},
    {"id": TCU,        "key": "tcu",        "name": "Towering cumulus","color": "#C08A21"},
    {"id": CONVECTIVE, "key": "convective", "name": "Convective",      "color": "#A11D33"},
    {"id": ANVIL_ATT,  "key": "anvil_att",  "name": "Anvil, attached", "color": "#E2703A"},
    {"id": ANVIL_DET,  "key": "anvil_det",  "name": "Anvil, detached", "color": "#F2A278"},
    {"id": DEBRIS,     "key": "debris",     "name": "Debris",          "color": "#9AA7B4"},
    {"id": CIRRUS,     "key": "cirrus",     "name": "Cirrus",          "color": "#BFD3E6"},
]
PALETTE = [c["color"] for c in sorted(CLASSES, key=lambda c: c["id"])]
KEY_BY_ID = {c["id"]: c["key"] for c in CLASSES}


# --------------------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------------------
def _session():
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16,
                                                      max_retries=3))
    s.headers.update({"User-Agent": "CloudScope/2.0 (+github actions)"})
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
    turns ~120 requests into a handful without pulling materially more data."""
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
            if r.status_code == 200 and "CLMR" in r.text and "GRLE" in r.text:
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
        r = sess.get(_url("wrfprs", date_str, cycle, fh) + ".idx", timeout=20)
        if r.status_code != 200:
            return None, 0
        want = []
        for e in _parse_idx(r.text):
            m = lvl_re.match(e["level"].strip())
            if not m:
                continue
            want_lv = LEVEL_SETS.get(e["short"])
            if want_lv and int(m.group(1)) in want_lv:
                want.append(e)
        if not want:
            return None, 0
        for s, e in _merge(want):
            rng = f"bytes={s}-{'' if e is None else e}"
            rr = sess.get(_url("wrfprs", date_str, cycle, fh), headers={"Range": rng}, timeout=90)
            if rr.status_code in (200, 206):
                out.write(rr.content)
                total += len(rr.content)
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
# pygrib's shortName for the microphysics fields varies with the eccodes build, so match on
# name as well and let either route win.
SHORT2KEY = {"gh": "HGT", "t": "TMP", "clwmr": "CLMR", "cimixr": "CIMIXR", "ciwmr": "CIMIXR",
             "cice": "CIMIXR", "snmr": "SNMR", "grle": "GRLE", "u": "UGRD", "v": "VGRD"}
NAME2KEY = [("geopotential height", "HGT"), ("temperature", "TMP"),
            ("cloud mixing ratio", "CLMR"), ("cloud water", "CLMR"),
            ("ice water mixing ratio", "CIMIXR"), ("cloud ice", "CIMIXR"),
            ("snow mixing ratio", "SNMR"), ("graupel", "GRLE"),
            ("u component of wind", "UGRD"), ("v component of wind", "VGRD")]
VARS = ("HGT", "TMP", "CLMR", "CIMIXR", "SNMR", "GRLE", "UGRD", "VGRD")


def _key_for(g):
    k = SHORT2KEY.get((getattr(g, "shortName", "") or "").lower())
    if k:
        return k
    nm = (getattr(g, "name", "") or "").lower().replace("-", " ")
    for frag, key in NAME2KEY:
        if frag in nm:
            return key
    return None


def read_fields(path):
    """Pull the cropped 3-D stacks out of one hour's GRIB."""
    lvl = {v: {} for v in VARS}
    refc = lats = lons = None
    grbs = pygrib.open(path)
    for g in grbs:
        if lats is None:
            lats, lons = g.latlons()
        short = (getattr(g, "shortName", "") or "").upper()
        name = (getattr(g, "name", "") or "").lower()
        if short == "REFC" or "composite" in name:
            refc = np.asarray(g.values, dtype=float)
            continue
        if getattr(g, "typeOfLevel", "") != "isobaricInhPa":
            continue
        key = _key_for(g)
        if key is None:
            continue
        try:
            lvl[key][int(g.level)] = np.asarray(g.values, dtype=float)
        except Exception:
            continue
    grbs.close()
    if lats is None or not lvl["TMP"] or not lvl["HGT"]:
        return None

    lons = np.where(lons > 180, lons - 360.0, lons)
    box = ((lats >= DOMAIN["lat_min"]) & (lats <= DOMAIN["lat_max"]) &
           (lons >= DOMAIN["lon_min"]) & (lons <= DOMAIN["lon_max"]))
    ys, xs = np.where(box)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = lambda a: a[y0:y1, x0:x1]

    levels = sorted(set(lvl["TMP"]) & set(lvl["HGT"]), reverse=True)   # surface -> top
    if len(levels) < 6:
        return None
    shape = crop(lvl["TMP"][levels[0]]).shape

    def stack(v):
        """Missing levels are physically zero for the condensate fields, so fill rather
        than drop - dropping would put holes in the vertical coordinate."""
        return np.stack([crop(lvl[v][L]) if L in lvl[v] else np.zeros(shape) for L in levels])

    out = {"tmpc": stack("TMP") - 273.15,
           "hgt_kft": stack("HGT") / 304.8,
           "qliq": stack("CLMR"),
           "qice": stack("CIMIXR"),
           "qsnow": stack("SNMR"),
           "qgrpl": stack("GRLE"),
           "u": stack("UGRD"), "v": stack("VGRD"),
           "refc": crop(refc) if refc is not None else np.zeros(shape),
           "lats": crop(lats), "lons": crop(lons), "levels": levels}
    return out


# --------------------------------------------------------------------------------------
# Classify
# --------------------------------------------------------------------------------------
def _at(arr, ind):
    """Sample a (nlev, ny, nx) stack at a per-column level index."""
    n = arr.shape[0]
    return np.take_along_axis(arr, np.clip(ind, 0, n - 1).astype(int)[None], axis=0)[0]


def _grow(cloud, start, direction):
    """Walk from `start` while the column stays cloudy, giving the far edge of the
    contiguous layer that `start` belongs to. direction -1 = downward, +1 = upward."""
    nlev = cloud.shape[0]
    cur = np.clip(start, 0, nlev - 1).astype(int)
    for _ in range(nlev):
        nxt = cur + direction
        inside = (nxt >= 0) & (nxt <= nlev - 1)
        ok = inside & _at(cloud, np.clip(nxt, 0, nlev - 1))
        cur = np.where(ok, np.clip(nxt, 0, nlev - 1), cur)
    return cur


def classify(f):
    """One cloud type per grid column. Returns (class_grid, diagnostics)."""
    p = np.asarray(f["levels"], dtype=float)
    nlev = len(p)
    ny, nx = f["refc"].shape

    # Condensate PATH per layer, g/m^2. dp/g converts a mixing ratio into the mass actually
    # sitting in the layer, which is what sets opacity - a mixing-ratio threshold silently
    # demands ~4x more condensate at 200 mb than at 900 mb to trip.
    dp = np.abs(np.gradient(p)) * 100.0                    # Pa
    w = (dp / 9.80665 * 1000.0)[:, None, None]             # kg/kg -> g/m^2
    liq = f["qliq"] * w
    ice = (f["qice"] + f["qsnow"]) * w                     # detrained ice + snow = the anvil
    gra = f["qgrpl"] * w

    # Graupel joins the mask only for "is there cloud here" - a core is not empty sky - but
    # is kept out of the ice path so it can't inflate an anvil's opacity with precipitation.
    cloud = (liq + ice + gra) > LAYER_PATH_MIN
    has_cloud = cloud.any(axis=0)
    kidx = np.arange(nlev).reshape(-1, 1, 1) * np.ones((1, ny, nx), dtype=int)

    top_i = np.where(cloud, kidx, -1).max(axis=0)          # highest cloudy level
    bot_i = np.where(cloud, kidx, nlev).min(axis=0)        # lowest cloudy level
    hi_base = _grow(cloud, top_i, -1)                      # base of the TOP layer
    lo_top = _grow(cloud, bot_i, +1)                       # top of the LOWEST layer

    hi_lay = cloud & (kidx >= hi_base[None]) & (kidx <= top_i[None]) & has_cloud[None]
    lo_lay = cloud & (kidx >= bot_i[None]) & (kidx <= lo_top[None]) & has_cloud[None]

    iwp_hi = (ice * hi_lay).sum(axis=0)
    lwp_hi = (liq * hi_lay).sum(axis=0)
    ice_frac_hi = np.where(iwp_hi + lwp_hi > 0, iwp_hi / np.maximum(iwp_hi + lwp_hi, 1e-9), 0.0)
    iwp_lo = (ice * lo_lay).sum(axis=0)
    lwp_lo = (liq * lo_lay).sum(axis=0)
    liq_frac_lo = np.where(iwp_lo + lwp_lo > 0, lwp_lo / np.maximum(iwp_lo + lwp_lo, 1e-9), 0.0)
    gcol = gra.sum(axis=0)

    z = f["hgt_kft"]
    top_kft = np.where(has_cloud, _at(z, top_i), np.nan)
    top_c = np.where(has_cloud, _at(f["tmpc"], top_i), np.nan)
    lo_top_kft, lo_base_kft = _at(z, lo_top), _at(z, bot_i)
    lo_top_c = _at(f["tmpc"], lo_top)
    depth_lo = np.where(has_cloud, lo_top_kft - lo_base_kft, 0.0)

    # Lumpiness of the lowest layer's top over ~15 km. A stratus deck is flat; a cumulus
    # field is not, and that texture separates them where a depth threshold alone can't.
    # Only cloudy neighbours count: including clear sky would read the flat edge of any
    # deck as violently lumpy and promote every stratus boundary to cumulus.
    n_tex = max(3, int(round(15.0 / DX_KM)) | 1)
    zt = np.where(has_cloud & np.isfinite(lo_top_kft), lo_top_kft, 0.0)
    m = (has_cloud & np.isfinite(lo_top_kft)).astype(float)
    den = np.maximum(uniform_filter(m, n_tex), 1e-6)
    mu = uniform_filter(zt, n_tex) / den
    mu2 = uniform_filter(zt ** 2, n_tex) / den
    tex = np.where(uniform_filter(m, n_tex) > 0.15, np.sqrt(np.maximum(mu2 - mu ** 2, 0)), 0.0)

    # --- convective cores ---------------------------------------------------------------
    core = (f["refc"] >= CONV_DBZ) | (gcol >= GRAUPEL_CONV)
    # Widen by one cell before tracing: a core can be a single grid column, and a trajectory
    # sampled with linear interpolation will otherwise walk straight through it.
    core_f = maximum_filter(core.astype(float), size=3)
    r_px = max(1, int(round(ATTACH_NM * 1.852 / DX_KM)))
    near_core = maximum_filter(core.astype(float), size=2 * r_px + 1) > 0.5

    # --- trace the ice layer back along ITS OWN wind ------------------------------------
    # v1 steered everything with a fixed 300-150 mb mean and searched a fixed 100 nm. The
    # outflow level varies with the EL, and how far debris can get is speed x lifetime, so
    # both are derived per column instead.
    wt = ice * hi_lay
    wsum = wt.sum(axis=0)
    fallback = hi_lay.sum(axis=0)
    u_lay = np.where(wsum > 0, (f["u"] * wt).sum(axis=0) / np.maximum(wsum, 1e-9),
                     (f["u"] * hi_lay).sum(axis=0) / np.maximum(fallback, 1))
    v_lay = np.where(wsum > 0, (f["v"] * wt).sum(axis=0) / np.maximum(wsum, 1e-9),
                     (f["v"] * hi_lay).sum(axis=0) / np.maximum(fallback, 1))
    spd = np.hypot(u_lay, v_lay)
    ux = np.where(spd > 0.5, u_lay / np.maximum(spd, 1e-6), 0.0)
    uy = np.where(spd > 0.5, v_lay / np.maximum(spd, 1e-6), 0.0)

    yy, xx = np.mgrid[0:ny, 0:nx]
    upwind = np.zeros((ny, nx))
    reach_km = np.minimum(spd * 3.6 * ANVIL_TAU_H, ANVIL_MAX_NM * 1.852)
    # Step in GRID units, not fractions of each column's reach: a core is often one or two
    # cells wide, and a fractional walk long enough to be useful strides right past it.
    max_px = int(np.ceil(reach_km.max() / DX_KM)) if reach_km.size else 0
    for s_px in np.arange(1.0, max_px + 1.0, 1.0):
        samp = map_coordinates(core_f, [yy - uy * s_px, xx - ux * s_px],
                               order=1, mode="nearest")
        upwind = np.maximum(upwind, np.where(s_px * DX_KM <= reach_km, samp, 0.0))
    sourced = upwind > 0.3

    # --- decision -----------------------------------------------------------------------
    # A glaciated layer whose TOP reached homogeneous freezing came out of the top of
    # something. Base temperature is deliberately not used: an attached or thick anvil
    # commonly has a base warmer than -20 C, and v1 discarded exactly those.
    ice_aloft = has_cloud & (ice_frac_hi >= ICE_FRAC) & (top_c <= GLACIATED_C)
    anvil = ice_aloft & (iwp_hi >= ANVIL_IWP) & (near_core | sourced)
    debris = ice_aloft & ~anvil & (iwp_hi >= DEBRIS_IWP) & sourced
    cirrus = ice_aloft & ~anvil & ~debris

    liq_base = has_cloud & (liq_frac_lo > 0.5)
    tcu = liq_base & (lo_top_c <= TCU_TOP_C)
    cu = liq_base & ~tcu & ((depth_lo >= CU_DEPTH_KFT) | (tex >= CU_TEX_KFT))
    st = liq_base & ~tcu & ~cu

    out = np.full((ny, nx), CLEAR, dtype=np.uint8)
    for mask, cid in ((cirrus, CIRRUS), (st, STRATIFORM), (cu, CUMULUS), (debris, DEBRIS),
                      (tcu, TCU), (anvil & ~near_core, ANVIL_DET), (anvil & near_core, ANVIL_ATT),
                      (core, CONVECTIVE)):
        out[mask] = cid                                    # ascending operational significance

    diag = {"top_kft": top_kft, "top_c": top_c, "iwp": iwp_hi, "lwp": lwp_lo,
            "depth_lo": depth_lo, "graupel": gcol, "refc": f["refc"]}
    return out, diag


# --------------------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------------------
def _figsize(height_in=6.8):
    """Match the figure's aspect to true ground distance so the PNG needs no letterboxing
    in the page - the old square figure was the reason the layout blew up."""
    pc = ccrs.PlateCarree()
    proj = ccrs.Mercator(central_longitude=0.5 * (DOMAIN["lon_min"] + DOMAIN["lon_max"]))
    x0, y0 = proj.transform_point(DOMAIN["lon_min"], DOMAIN["lat_min"], pc)
    x1, y1 = proj.transform_point(DOMAIN["lon_max"], DOMAIN["lat_max"], pc)
    return proj, (height_in * (x1 - x0) / (y1 - y0), height_in)


def render(cls, f, valid, cycle, path):
    proj, size = _figsize()
    cmap = mcolors.ListedColormap(PALETTE)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(PALETTE) + 0.5, 1), len(PALETTE))
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=size, dpi=140, facecolor="#0D1117")
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.set_extent([DOMAIN["lon_min"], DOMAIN["lon_max"],
                   DOMAIN["lat_min"], DOMAIN["lat_max"]], crs=pc)
    ax.set_facecolor("#0D1117")
    ax.pcolormesh(f["lons"], f["lats"], cls, cmap=cmap, norm=norm,
                  shading="nearest", transform=pc, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#5A6B7D",
                   linewidth=0.8, zorder=4)
    ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines",
                                                "10m", facecolor="none"),
                   edgecolor="#3C4A59", linewidth=0.5, zorder=4)
    for name, (la, lo) in SITES.items():
        ax.plot(lo, la, marker="+", markersize=6, markeredgewidth=1.2,
                color="#F2F4F0", transform=pc, zorder=6)
        ax.text(lo + 0.05, la + 0.03, name, fontsize=6.0, color="#F2F4F0",
                family="monospace", transform=pc, zorder=6)
    ax.text(0.015, 0.014, f"HRRR {cycle}Z  valid {valid:%d %b %H%MZ}", transform=ax.transAxes,
            fontsize=6.4, color="#7C8C9C", family="monospace", zorder=7)
    fig.savefig(path, facecolor="#0D1117")
    plt.close(fig)


# --------------------------------------------------------------------------------------
def site_indices(f):
    idx = {}
    for name, (la, lo) in SITES.items():
        d = (f["lats"] - la) ** 2 + (f["lons"] - lo) ** 2
        idx[name] = np.unravel_index(np.argmin(d), d.shape)
    return idx


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
            render(cls, f, valid, cycle, os.path.join(OUT_DIR, png))

            sites = {}
            for name, (jy, jx) in site_indices(f).items():
                sites[name] = {
                    "key": KEY_BY_ID[int(cls[jy, jx])],
                    "top_kft": None if not np.isfinite(diag["top_kft"][jy, jx])
                    else round(float(diag["top_kft"][jy, jx]), 1),
                    "top_c": None if not np.isfinite(diag["top_c"][jy, jx])
                    else round(float(diag["top_c"][jy, jx]), 1),
                    "iwp": round(float(diag["iwp"][jy, jx]), 1),
                    "dbz": round(float(diag["refc"][jy, jx]), 1),
                }
            counts = {c["key"]: int((cls == c["id"]).sum()) for c in CLASSES}
            frames.append({"fh": fh, "valid": valid.strftime("%Y-%m-%dT%H:%MZ"),
                           "valid_short": valid.strftime("%d/%HZ"), "image": png,
                           "counts": counts, "sites": sites})
            logging.info(f"f{fh:02d} valid {valid:%d/%HZ}: " +
                         " ".join(f"{k}={v}" for k, v in counts.items() if v and k != "clear"))
        finally:
            if os.path.exists(path):
                os.remove(path)

    if not frames:
        logging.error("No frames rendered; manifest not rewritten.")
        return

    keep = {os.path.basename(fr["image"]) for fr in frames}
    for fn in os.listdir(MAP_DIR):
        if fn.endswith(".png") and fn not in keep:
            os.remove(os.path.join(MAP_DIR, fn))

    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "model": "HRRR", "cycle": f"{date_str} {cycle}Z",
        "domain": DOMAIN, "classes": CLASSES, "sites": list(SITES),
        "frames": frames,
        "thresholds": {"layer_path_min_gm2": LAYER_PATH_MIN, "glaciated_c": GLACIATED_C,
                       "ice_fraction": ICE_FRAC, "anvil_iwp_gm2": ANVIL_IWP,
                       "debris_iwp_gm2": DEBRIS_IWP, "convective_dbz": CONV_DBZ,
                       "graupel_gm2": GRAUPEL_CONV, "attach_nm": ATTACH_NM,
                       "anvil_tau_h": ANVIL_TAU_H, "anvil_max_nm": ANVIL_MAX_NM,
                       "tcu_top_c": TCU_TOP_C, "cu_depth_kft": CU_DEPTH_KFT},
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh_:
        json.dump(manifest, fh_, indent=1)
    logging.info(f"{len(frames)} frames, {bytes_total/1024/1024:.0f} MB transferred.")


if __name__ == "__main__":
    main()
