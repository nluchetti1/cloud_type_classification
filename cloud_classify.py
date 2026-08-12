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
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
HRRR_ROOT = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
OUT_DIR = "docs"                     # GitHub Pages serves from /docs
MAP_DIR = os.path.join(OUT_DIR, "maps")
DATA_DIR = os.path.join(OUT_DIR, "data")   # packed grids the viewer queries on click
CACHE_DIR = "_cache"

DOMAIN = {"lat_min": 26.5, "lat_max": 30.5, "lon_min": -82.5, "lon_max": -79.0}

# Active pads plus the two reference sites. KXMR is the LLCC observing site and the anchor
# of the XMR climatology; KTTS is the SLF. KMLB/KDAB are gone - they are not launch sites.
SITES = {
    "LC-39A":  (28.6084, -80.6043),   # Falcon 9 / Falcon Heavy / Starship (planned)
    "LC-39B":  (28.6272, -80.6208),   # SLS
    "SLC-41":  (28.5833, -80.5834),   # Vulcan / Atlas V
    "SLC-40":  (28.5619, -80.5772),   # Falcon 9
    "SLC-37B": (28.5317, -80.5657),   # New Glenn
    "SLC-20":  (28.5085, -80.5546),
    "LZ-1":    (28.4857, -80.5444),   # booster landing zone
    "SLC-36":  (28.4707, -80.5379),   # New Glenn
    "SLC-46":  (28.4584, -80.5271),
    "KTTS":    (28.6150, -80.6944),   # Shuttle Landing Facility
    "KXMR":    (28.4675, -80.5664),   # Cape Canaveral SFS, LLCC observing site
}

# Every pad above sits inside ~20 km, which is ~30 px on the domain map - an unreadable
# blob. The inset re-renders that patch at scale so the pads can actually be told apart.
# Extended east past the shoreline on purpose: the empty ocean strip is where the label
# column goes, so leaders never have to cross a pad marker.
CAPE_BOX = {"lat_min": 28.43, "lat_max": 28.67, "lon_min": -80.73, "lon_max": -80.44}

BG = "#FFFFFF"                                # map background
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

# The PNG carries colour, not data, so a queryable copy of each hour is written alongside it.
# It is resampled onto a regular lat/lon mesh because HRRR is Lambert Conformal: at 80 W the
# grid is rotated ~10 deg off north, so a click could not be turned into an index by arithmetic
# on the native grid. 0.03 deg is ~3.2 km, near enough to HRRR's own spacing to lose nothing.
QUERY_DEG = 0.03
IWP_MAX = 3000.0        # ceiling of the logarithmic ice-path packing, g/m^2

# HRRR runs out to 48 h on the synoptic cycles and 18 h on the rest. Asking for f19+ on an
# off-hour cycle just 404s, so the length is derived from the cycle rather than fixed.
EXTENDED_CYCLES = {0, 6, 12, 18}
SHORT_RUN_H, EXTENDED_RUN_H = 18, 48

# Start one cycle back. A run is adopted as soon as its f01 index is posted, but HRRR takes
# ~50-60 min to finish an 18 h run (~100 min for a 48 h synoptic cycle), so reaching for the
# freshest cycle bought a map that was 40 minutes newer and four forecast hours shorter. The
# N-1 cycle is essentially complete on arrival; the top-up pass then fills the rest.
CYCLE_LAG_H = 1
MAX_CYCLE_LOOKBACK_H = 6

# dprog/dt: how many cycles stay on disk. Each run only ever processes the NEWEST cycle -
# the older ones are already here from the runs that made them, so run-to-run comparison
# costs nothing extra in bandwidth. Recomputing them hourly would be 4x the downloads for
# an identical answer.
KEEP_CYCLES = 4

# How many older, still-incomplete cycles to top up on one pass, on top of the newest. Bounds
# the runtime when several runs were picked up early.
MAX_TOPUP_CYCLES = 2

# Bump whenever the rendering or the classification changes. A cycle that is already
# published is normally skipped, but a version mismatch means the PNGs on disk were made by
# older code and have to be rebuilt - otherwise pushing a render change appears to do
# nothing until the next cycle lands. Setting CLOUDSCOPE_FORCE=1 forces the same rebuild.
RENDER_VERSION = "2026.08.12-inset"

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
    {"id": CLEAR,      "key": "clear",      "name": "Clear",           "color": "#FFFFFF"},
    {"id": STRATIFORM, "key": "stratiform", "name": "Stratiform",      "color": "#5C7A99"},
    {"id": CUMULUS,    "key": "cumulus",    "name": "Cumulus",         "color": "#E0A83C"},
    {"id": TCU,        "key": "tcu",        "name": "Towering cumulus","color": "#B0700F"},
    {"id": CONVECTIVE, "key": "convective", "name": "Convective",      "color": "#A11D33"},
    {"id": ANVIL_ATT,  "key": "anvil_att",  "name": "Anvil, attached", "color": "#E2703A"},
    {"id": ANVIL_DET,  "key": "anvil_det",  "name": "Anvil, detached", "color": "#F0A87E"},
    {"id": DEBRIS,     "key": "debris",     "name": "Debris",          "color": "#C7B49E"},
    {"id": CIRRUS,     "key": "cirrus",     "name": "Cirrus",          "color": "#9EC0DC"},
]
# Tuned for a WHITE map. The pale ice colours darkened to survive on paper, and debris moved
# off blue-grey to a faded tan so it reads as aged anvil, not a third variety of cirrus.
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


def run_hours(cycle):
    """Forecast hours to attempt for this cycle."""
    n = EXTENDED_RUN_H if int(cycle) in EXTENDED_CYCLES else SHORT_RUN_H
    return list(range(1, n + 1))


def find_cycle(sess):
    """Newest cycle at least CYCLE_LAG_H old whose f01 wrfprs index is posted."""
    now = datetime.datetime.now(datetime.timezone.utc)
    for back in range(CYCLE_LAG_H, MAX_CYCLE_LOOKBACK_H + 1):
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
    # Anything cloudy that matched nothing above - typically a mid-level glaciated deck
    # topping warmer than -38 C, ice-dominated so not liquid-based either - was silently
    # falling through to CLEAR. Layered cloud is the honest default for it.
    unclaimed = has_cloud & ~ice_aloft & ~liq_base
    st = (liq_base & ~tcu & ~cu) | unclaimed

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
def _mproj():
    return ccrs.Mercator(central_longitude=0.5 * (DOMAIN["lon_min"] + DOMAIN["lon_max"]))


def _aspect(proj, box):
    """Width/height of a lat-lon box in projected units."""
    pc = ccrs.PlateCarree()
    x0, y0 = proj.transform_point(box["lon_min"], box["lat_min"], pc)
    x1, y1 = proj.transform_point(box["lon_max"], box["lat_max"], pc)
    return (x1 - x0) / (y1 - y0)


def _figsize(height_in=6.8):
    """Match the figure's aspect to true ground distance so the PNG needs no letterboxing
    in the page - a square figure was the reason the layout blew up."""
    proj = _mproj()
    return proj, (height_in * _aspect(proj, DOMAIN), height_in)


def _basemap(ax, box, coast_lw=0.9):
    pc = ccrs.PlateCarree()
    ax.set_extent([box["lon_min"], box["lon_max"], box["lat_min"], box["lat_max"]], crs=pc)
    ax.set_facecolor(BG)
    # On white, clear sky and clear water are the same colour, so the coastline is the only
    # thing carrying geography - it gets the weight it used to borrow from the dark panel.
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#46525C",
                   linewidth=coast_lw, zorder=4)


def _label_stack(pts, min_gap):
    """Push overlapping label positions apart along y, keeping their order. Without this the
    pad labels inside the inset land on top of each other - SLC-40 and SLC-41 are 2 km apart."""
    order = sorted(range(len(pts)), key=lambda k: -pts[k])
    out = list(pts)
    for n, k in enumerate(order):
        if n and out[k] > out[order[n - 1]] - min_gap:
            out[k] = out[order[n - 1]] - min_gap
    lo, hi = min(out), max(out)
    if hi > 0.965:                     # ran off the top; slide the whole stack down
        out = [v - (hi - 0.965) for v in out]
        lo = min(out)
    if lo < 0.035:                     # ran off the bottom; slide it back up
        out = [v + (0.035 - lo) for v in out]
    return out


def _cape_inset(fig, ax, proj, size, cls, f, mesh):
    """Re-render the pad cluster at scale. Every pad sits inside ~20 km, which is ~30 px on
    the domain map - a blob. Here they are tens of pixels apart and can be labelled."""
    pc = ccrs.PlateCarree()
    hf = 0.30
    wf = (hf * size[1] * _aspect(proj, CAPE_BOX)) / size[0]
    m = 0.012
    iax = fig.add_axes([1 - wf - m, 1 - hf - m, wf, hf], projection=proj, zorder=8)
    _basemap(iax, CAPE_BOX, coast_lw=1.1)
    iax.pcolormesh(f["lons"], f["lats"], cls, **mesh)
    try:
        iax.spines["geo"].set(edgecolor="#14181B", linewidth=1.2)
    except Exception:
        pass

    # Where the inset is looking, drawn on the domain map.
    ax.plot([CAPE_BOX["lon_min"], CAPE_BOX["lon_max"], CAPE_BOX["lon_max"],
             CAPE_BOX["lon_min"], CAPE_BOX["lon_min"]],
            [CAPE_BOX["lat_min"], CAPE_BOX["lat_min"], CAPE_BOX["lat_max"],
             CAPE_BOX["lat_max"], CAPE_BOX["lat_min"]],
            color="#14181B", linewidth=0.8, transform=pc, zorder=7)

    # Transforms are only trustworthy once the axes has been through a draw.
    fig.canvas.draw()
    inv = iax.transAxes.inverted()
    fx, fy, names = [], [], []
    for name, (la, lo) in SITES.items():
        px, py = proj.transform_point(lo, la, pc)
        u, v = inv.transform(iax.transData.transform((px, py)))
        if -0.02 <= u <= 1.02 and -0.02 <= v <= 1.02:
            fx.append(u); fy.append(v); names.append(name)
    if not names:
        logging.warning("Cape inset: no pads fell inside CAPE_BOX.")
        return

    # One column, not two. Splitting left/right decluttered each side independently and then
    # let the two sides collide in the middle - KTTS and LC-39A landed on top of each other.
    LX = 0.985
    # Keep the column clear of the map content: stack inside the top/bottom margins so the
    # first and last labels are not clipped by the inset frame.
    ly = _label_stack(fy, 0.078)
    for k, name in enumerate(names):
        iax.plot([fx[k], LX - 0.005], [fy[k], ly[k]], color="#14181B", linewidth=0.5,
                 alpha=0.55, transform=iax.transAxes, zorder=9)
        iax.plot(fx[k], fy[k], marker="+", markersize=5, markeredgewidth=1.2,
                 color="#14181B", transform=iax.transAxes, zorder=10)
        iax.text(LX, ly[k], name, fontsize=5.6, color="#14181B", family="monospace",
                 ha="right", va="center", transform=iax.transAxes, zorder=10,
                 path_effects=[pe.withStroke(linewidth=2.2, foreground=BG)])
    iax.text(0.02, 0.975, "CAPE DETAIL", fontsize=5.2, color="#7A858E", family="monospace",
             va="top", transform=iax.transAxes, zorder=10,
             path_effects=[pe.withStroke(linewidth=2.0, foreground=BG)])


def render(cls, f, valid, cycle, path):
    proj, size = _figsize()
    pc = ccrs.PlateCarree()
    cmap = mcolors.ListedColormap(PALETTE)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(PALETTE) + 0.5, 1), len(PALETTE))
    mesh = dict(cmap=cmap, norm=norm, shading="nearest", transform=pc, zorder=2)

    fig = plt.figure(figsize=size, dpi=140, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    _basemap(ax, DOMAIN)
    ax.pcolormesh(f["lons"], f["lats"], cls, **mesh)
    ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines",
                                                "10m", facecolor="none"),
                   edgecolor="#AEB6BD", linewidth=0.6, zorder=4)
    for la, lo in SITES.values():
        ax.plot(lo, la, marker=".", markersize=2.0, color="#14181B", transform=pc, zorder=6)

    # --- Cape inset ---------------------------------------------------------------------
    # Wrapped: a labelling bug in here must not take the whole map down with it. A frame with
    # no inset is still a usable frame; a frame that raised is a hole in the loop.
    try:
        _cape_inset(fig, ax, proj, size, cls, f, mesh)
    except Exception as e:
        logging.warning(f"Cape inset skipped: {type(e).__name__}: {e}")

    ax.text(0.012, 0.012, f"HRRR {cycle}Z  valid {valid:%d %b %H%MZ}", transform=ax.transAxes,
            fontsize=6.4, color="#7A858E", family="monospace", zorder=7)
    try:
        ax.spines["geo"].set_edgecolor("#C3C8BC")
    except Exception:
        pass
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Queryable export
# --------------------------------------------------------------------------------------
_QGRID = {}   # the crop is identical every hour, so the resampling is solved once per run

PLANES = ["class", "top_kft_x4", "top_c_p100", "iwp_log", "depth_kft_x4", "dbz"]


def query_grid(f):
    """Nearest-neighbour map from the model's Lambert crop onto a regular lat/lon mesh."""
    key = (f["lats"].shape, round(float(f["lats"][0, 0]), 4), round(float(f["lons"][0, 0]), 4))
    if key in _QGRID:
        return _QGRID[key]
    qlat = np.arange(DOMAIN["lat_min"], DOMAIN["lat_max"] + 1e-9, QUERY_DEG)
    qlon = np.arange(DOMAIN["lon_min"], DOMAIN["lon_max"] + 1e-9, QUERY_DEG)
    LA, LO = np.meshgrid(qlat, qlon, indexing="ij")          # row 0 = south edge
    cos0 = np.cos(np.radians(0.5 * (DOMAIN["lat_min"] + DOMAIN["lat_max"])))
    tree = cKDTree(np.column_stack([f["lats"].ravel(), f["lons"].ravel() * cos0]))
    _, idx = tree.query(np.column_stack([LA.ravel(), LO.ravel() * cos0]))
    _QGRID[key] = (idx.reshape(LA.shape), qlat, qlon)
    return _QGRID[key]


def pack_query(cls, diag, f):
    """Six uint8 planes, concatenated. Quantisation is chosen so the decode error is under
    the precision anyone would act on: 0.25 kft of height, 1 C, ~2% of ice path, 1 dBZ."""
    idx, qlat, qlon = query_grid(f)
    take = lambda a: np.asarray(a).ravel()[idx.ravel()]
    q = lambda a, lo=0, hi=255: np.clip(np.round(a), lo, hi).astype(np.uint8)
    iwp = np.maximum(take(diag["iwp"]), 0.0)
    planes = [
        take(cls).astype(np.uint8),
        q(np.nan_to_num(take(diag["top_kft"]), nan=0.0) * 4.0),
        q(np.nan_to_num(take(diag["top_c"]), nan=-100.0) + 100.0),
        q(255.0 * np.log10(1.0 + iwp) / np.log10(1.0 + IWP_MAX)),
        q(np.nan_to_num(take(diag["depth_lo"]), nan=0.0) * 4.0),
        q(take(diag["refc"]), 0, 80),
    ]
    return b"".join(p.tobytes() for p in planes), len(qlon), len(qlat), qlat[0], qlon[0]


# --------------------------------------------------------------------------------------
def site_indices(f):
    idx = {}
    for name, (la, lo) in SITES.items():
        d = (f["lats"] - la) ** 2 + (f["lons"] - lo) ** 2
        idx[name] = np.unravel_index(np.argmin(d), d.shape)
    return idx


def load_manifest():
    try:
        with open(os.path.join(OUT_DIR, "manifest.json")) as fp:
            return json.load(fp)
    except Exception:
        return {}


def build_cycle(sess, date_str, cycle, cyc_dt, cid, existing):
    """Render whatever hours of this cycle are on S3 and are not already built.

    HRRR posts a run hour by hour over roughly 50-90 minutes, so a cycle picked up shortly
    after f01 appears is genuinely incomplete - the tail 404s. Rather than waiting for the
    whole run (which would put the page an hour behind) or accepting the truncation
    permanently (which is what silently happened before), each pass fills in the hours that
    have shown up since. Returns (frames, bytes, qmeta, n_new).
    """
    have = {int(fr["fh"]): fr for fr in existing}
    hours = run_hours(cycle)
    todo = [h for h in hours if h not in have]
    if not todo:
        return sorted(have.values(), key=lambda fr: fr["fh"]), 0, None, 0

    total, qmeta, made = 0, None, 0
    for fh in todo:
        path, n = fetch_hour(sess, date_str, cycle, fh)
        total += n
        if not path or n == 0:
            logging.info(f"f{fh:02d}: not posted yet")
            continue
        try:
            f = read_fields(path)
            if f is None:
                logging.warning(f"f{fh:02d}: fields missing")
                continue
            cls, diag = classify(f)
            valid = cyc_dt + datetime.timedelta(hours=fh)
            png = f"maps/cloudtype_{cid}z_f{fh:02d}.png"
            render(cls, f, valid, cycle, os.path.join(OUT_DIR, png))
            blob, qnx, qny, qlat0, qlon0 = pack_query(cls, diag, f)
            qmeta = {"nx": qnx, "ny": qny, "lat0": round(float(qlat0), 6),
                     "lon0": round(float(qlon0), 6), "dlat": QUERY_DEG, "dlon": QUERY_DEG,
                     "iwp_max": IWP_MAX, "planes": PLANES}
            binrel = f"data/cols_{cid}z_f{fh:02d}.bin"
            with open(os.path.join(OUT_DIR, binrel), "wb") as bf:
                bf.write(blob)

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
            # "%d/%HZ" put day-of-month first, so F12 of the 10Z run read as "12/22Z" and
            # looked like a 12Z cycle. Hour only here; valid_label carries the date.
            have[fh] = {"fh": fh, "valid": valid.strftime("%Y-%m-%dT%H:%MZ"),
                        "valid_short": valid.strftime("%HZ"),
                        "valid_label": valid.strftime("%HZ %a %d %b"),
                        "image": png, "data": binrel, "counts": counts, "sites": sites}
            made += 1
            logging.info(f"f{fh:02d} valid {valid:%d %b %HZ}: " +
                         " ".join(f"{k}={v}" for k, v in counts.items() if v and k != "clear"))
        finally:
            if os.path.exists(path):
                os.remove(path)

    return sorted(have.values(), key=lambda fr: fr["fh"]), total, qmeta, made


def main():
    os.makedirs(MAP_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    sess = _session()
    date_str, cycle, cyc_dt = find_cycle(sess)
    if not cycle:
        logging.error("No HRRR cycle available; leaving the previous run in place.")
        return

    prev = load_manifest()
    force = os.environ.get("CLOUDSCOPE_FORCE", "").strip().lower() in ("1", "true", "yes")
    stale = prev.get("render_version") != RENDER_VERSION
    if stale and prev:
        logging.info(f"Render version changed ({prev.get('render_version', 'pre-versioning')}"
                     f" -> {RENDER_VERSION}); rebuilding from scratch.")
    elif force:
        logging.info("CLOUDSCOPE_FORCE set; rebuilding from scratch.")
    # A version bump makes the PNGs on disk inconsistent with the new ones, so the old cycles
    # are dropped rather than mixed. dprog/dt rebuilds itself over the next few hours.
    prior = [] if (stale or force) else list(prev.get("cycles", []))

    cid = f"{date_str}{cycle}"
    cyc_dts = {cid: cyc_dt}
    entries, bytes_total, qmeta, built = [], 0, None, 0

    # Newest cycle first, then top up any retained cycle still missing hours - a run that was
    # picked up early is finished on a later pass instead of staying truncated forever.
    todo = [(cid, date_str, cycle)]
    for c in prior:
        if c["id"] != cid and len(c["frames"]) < len(run_hours(c["hour"])):
            todo.append((c["id"], c["date"], c["hour"]))
            cyc_dts[c["id"]] = datetime.datetime.strptime(
                c["init"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=datetime.timezone.utc)
        if len(todo) >= MAX_TOPUP_CYCLES + 1:
            break

    for tid, tdate, thour in todo:
        existing = next((c["frames"] for c in prior if c["id"] == tid), [])
        frames, nbytes, qm, made = build_cycle(sess, tdate, thour, cyc_dts[tid], tid, existing)
        bytes_total += nbytes
        qmeta = qm or qmeta
        built += made
        if not frames:
            continue
        want = run_hours(thour)
        entries.append({"id": tid, "label": f"{thour}Z", "date": tdate, "hour": thour,
                        "init": cyc_dts[tid].strftime("%Y-%m-%dT%H:%MZ"),
                        "render_version": RENDER_VERSION,
                        "run_h": len(frames), "run_h_expected": len(want),
                        "complete": len(frames) >= len(want), "frames": frames})
        if made:
            logging.info(f"{tid}Z: {len(frames)}/{len(want)} h "
                         f"({'complete' if len(frames) >= len(want) else 'partial'}), "
                         f"+{made} new this pass.")

    if not entries and not prior:
        logging.error("No frames rendered; manifest not rewritten.")
        return

    by_id = {e["id"]: e for e in entries}
    cycles = [by_id.get(c["id"], c) for c in prior if c["id"] != cid]
    cycles = ([by_id[cid]] if cid in by_id else []) + cycles
    cycles = cycles[:KEEP_CYCLES]
    if not cycles:
        logging.error("Nothing to publish; manifest not rewritten.")
        return

    live = set()
    for c in cycles:
        for fr in c["frames"]:
            live.add(os.path.basename(fr["image"]))
            live.add(os.path.basename(fr["data"]))
    dropped = 0
    for d, ext in ((MAP_DIR, ".png"), (DATA_DIR, ".bin")):
        for fn in os.listdir(d):
            if fn.endswith(ext) and fn not in live:
                os.remove(os.path.join(d, fn))
                dropped += 1

    newest = cycles[0]
    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "render_version": RENDER_VERSION,
        "model": "HRRR", "cycle": f"{newest['date']} {newest['hour']}Z",
        "domain": DOMAIN, "classes": CLASSES, "sites": list(SITES),
        "query": qmeta or prev.get("query"),
        "cycles": cycles,
        "frames": newest["frames"],   # so an older viewer still works
        "thresholds": {"layer_path_min_gm2": LAYER_PATH_MIN, "glaciated_c": GLACIATED_C,
                       "ice_fraction": ICE_FRAC, "anvil_iwp_gm2": ANVIL_IWP,
                       "debris_iwp_gm2": DEBRIS_IWP, "convective_dbz": CONV_DBZ,
                       "graupel_gm2": GRAUPEL_CONV, "attach_nm": ATTACH_NM,
                       "anvil_tau_h": ANVIL_TAU_H, "anvil_max_nm": ANVIL_MAX_NM,
                       "tcu_top_c": TCU_TOP_C, "cu_depth_kft": CU_DEPTH_KFT},
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=1)
    logging.info(f"+{built} frames, {bytes_total/1024/1024:.0f} MB transferred; holding "
                 + ", ".join(f"{c['id']}({c['run_h']}/{c['run_h_expected']}h)" for c in cycles)
                 + f"; pruned {dropped} files.")


if __name__ == "__main__":
    main()
