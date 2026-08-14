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
       cirrus split fall out of ice water path (anvil 50+, thin cirrus below).

    3. ANVIL BY DETRAINMENT PHYSICS, NOT BASE TEMPERATURE. An anvil's base is set by where
       the tower detrains, and a thick or attached anvil routinely has a base warmer than
       -20 C, so v1's ANVIL_BASE_C threw those away. v2 keys on what an anvil actually is:
       an optically substantial, glaciated layer whose TOP sits at the homogeneous-freezing
       level or above (<= -38 C, i.e. it came out of the top of a tower, not a mid-level
       deck), traced back along its own layer-mean wind to a convective source within a
       physical advection time rather than a fixed distance in nm.

    4. ATTACHED vs DETACHED. The LLCC treats those differently, so the classifier does too:
       attached = core inside 10 nm; detached = joined to a core through continuous shield,
       or off one within the 3 h clock.

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
RRFS_ROOT = "https://noaa-rrfs-pds.s3.amazonaws.com"
OUT_DIR = "docs"                     # GitHub Pages serves from /docs
MAP_DIR = os.path.join(OUT_DIR, "maps")
DATA_DIR = os.path.join(OUT_DIR, "data")   # packed grids the viewer queries on click
STATE_DIR = os.path.join(OUT_DIR, "state") # outflow age carried hour to hour
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
# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
# Why more than one model, and why these two in particular.
#
# REFS publishes ensemble PRODUCTS only - mean, spread, probability-matched mean, prob, eas -
# and no member files, so there is nothing per-member to classify. But SCN 26-48 spells the
# membership out: REFS combines the current and 6 h old cycles of the RRFS deterministic and
# ensemble systems, and over CONUS the HRRR contributes two more members from the current and
# 6 h old cycles. Four of those - HRRR at t and t-6, RRFS deterministic at t and t-6 - ARE
# published individually. So instead of reading somebody's precomputed probability of a
# reflectivity threshold, CloudScope classifies those four itself and gets a REFS-shaped
# ensemble it can run every LLCC rule against. The five true RRFS ensemble members do not
# appear in the SCN's output listing and stay out of reach.
#
# RRFS is marked unverified until probe_models.py confirms the prslev files carry the
# hydrometeor fields; the classifier needs condensate, not relative humidity, and an FV3
# pressure-level product is not guaranteed to carry all four species.
MODELS = {
    "hrrr": {
        "name": "HRRR", "dx_km": 3.0, "hourly": True,
        "short_run_h": 18, "extended_run_h": 48,
        "files": lambda d, c, fh: [
            (f"{HRRR_ROOT}/hrrr.{d}/conus/hrrr.t{c}z.wrfprsf{fh:02d}.grib2", "levels"),
            (f"{HRRR_ROOT}/hrrr.{d}/conus/hrrr.t{c}z.wrfsfcf{fh:02d}.grib2", "refc"),
        ],
        "probe": "CLMR", "verified": True,
    },
    "rrfs": {
        # SCN 26-48: rrfs.YYYYMMDD/CC/rrfs.tCCz.prslev.3km.fFFF.conus.grib2, hourly to 18 h
        # and to 84 h on 00/06/12/18Z. One file - prslev is expected to carry REFC too.
        "name": "RRFS", "dx_km": 3.0, "hourly": True,
        "short_run_h": 18, "extended_run_h": 60,
        "files": lambda d, c, fh: [
            (f"{RRFS_ROOT}/rrfs_public/rrfs.{d}/{c}/rrfs.t{c}z.prslev.3km.f{fh:03d}.conus.grib2",
             "levels+refc"),
        ],
        "probe": "CLMR", "verified": False,
    },
}
MODEL = os.environ.get("CLOUDSCOPE_MODEL", "hrrr").strip().lower()
if MODEL not in MODELS:
    MODEL = "hrrr"

DX_KM = MODELS[MODEL]["dx_km"]

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

# Runs go long on the synoptic cycles and short on the rest. Asking for hours past the end
# just 404s, so the length comes from the cycle and the model rather than being fixed.
EXTENDED_CYCLES = {0, 6, 12, 18}
SHORT_RUN_H = MODELS[MODEL]["short_run_h"]
EXTENDED_RUN_H = MODELS[MODEL]["extended_run_h"]

# Start one cycle back. A run is adopted as soon as its f01 index is posted, but HRRR takes
# ~50-60 min to finish an 18 h run (~100 min for a 48 h synoptic cycle), so reaching for the
# freshest cycle bought a map that was 40 minutes newer and four forecast hours shorter. The
# N-1 cycle is essentially complete on arrival; the top-up pass then fills the rest.
CYCLE_LAG_H = 1
MAX_CYCLE_LOOKBACK_H = 6

# How many cycles stay on disk. Each run only ever processes the NEWEST one - the older ones
# are already here from the runs that built them - so this is the size of the time-lagged
# ensemble AND the depth of the dprog/dt strip, and it costs nothing extra to download.
#
# Six because four members make a coarse probability: the only values a 4-member POV can take
# are 0, 25, 50, 75, 100. Six gives 17% steps, which is finer than the forecast justifies but
# at least stops the map looking quantised. Going much beyond that is self-defeating - the
# oldest member is then six hours stale and is not really voting on the same forecast.
KEEP_CYCLES = 6

# How many older, still-incomplete cycles to top up on one pass, on top of the newest. Bounds
# the runtime when several runs were picked up early.
MAX_TOPUP_CYCLES = 2

# Below this many members a probability is not a probability, it is a deterministic flag.
POV_MIN_MEMBERS = 2

# Members older than this are dropped from the POV. A run from eight hours ago has seen a
# genuinely different atmosphere and drags the probability toward its own stale solution.
POV_MAX_AGE_H = 6

# Bump whenever the rendering or the classification changes. A cycle that is already
# published is normally skipped, but a version mismatch means the PNGs on disk were made by
# older code and have to be rebuilt - otherwise pushing a render change appears to do
# nothing until the next cycle lands. Setting CLOUDSCOPE_FORCE=1 forces the same rebuild.
# Two versions, because they have very different costs.
#
# DATA_VERSION covers the packed .bin layout and the classification itself. When it moves,
# older cycles are genuinely unreadable and have to go - which takes the time-lagged ensemble
# and the dprog/dt strip down with them for several hours.
#
# RENDER_VERSION covers only what the PNGs look like. Bumping that used to drop every
# retained cycle too, so a palette tweak cost six hours of ensemble. Now it just rebuilds the
# newest cycle's images; older runs keep their old-style PNGs, which is a small visual
# inconsistency in the run selector and a much better trade than losing the POV.
DATA_VERSION = "2026.08.14-tle6"
RENDER_VERSION = "2026.08.14-tle6"

# ---- classification thresholds (all tunable; see README) ----
LAYER_PATH_MIN = 0.20   # g/m^2 of condensate in one layer to call it cloudy
GLACIATED_C    = -38.0  # homogeneous freezing: a top this cold is cirriform by construction
ICE_FRAC       = 0.80   # ice share WITHIN the layer for it to count as glaciated
ANVIL_IWP      = 50.0   # g/m^2 - optically substantial ice; anvils run 100-1000
CONV_DBZ       = 40.0   # composite reflectivity marking a convective core
GRAUPEL_CONV   = 200.0  # g/m^2 of graupel: riming implies an updraft holding supercooled water
ATTACH_NM      = 10.0   # core this close and the anvil is attached, not advected
# Two different clocks, because the LLCC uses two. A shield still joined to its parent is an
# attached anvil no matter how long the ice has been streaming - the rule is written about
# distance from it, not its age. The 3-hour clock belongs to anvil that has SEPARATED and to
# detached anvil. Applying the 3 h cap to everything aged a live, connected shield to cirrus
# while its cores were still firing.
ANVIL_TAU_H    = 3.0    # detached anvil: hours since it left the core
CONN_MAX_H     = 6.0    # sanity cap on tracing a continuous shield, not an age limit
AGE_MAX_H      = 25.0   # ceiling of the stored age field, hours
ANVIL_MAX_NM   = 150.0  # cap, so a 90 kt jet doesn't sweep the whole domain
TCU_TOP_C      = -10.0  # liquid-based layer glaciating at its top
CU_DEPTH_KFT   = 3.0    # depth separating cumuliform from a layered deck
CU_TEX_KFT     = 1.2    # or lumpiness: sigma of cloud-top height over ~15 km

# Debris is gone. In model land the difference between thinning anvil ice and cirrus is a
# guess about optical depth that HRRR's microphysics does not really support, so sourced ice
# is either optically substantial enough to be an anvil or it is cirrus.
(CLEAR, STRATIFORM, CUMULUS, TCU, CONVECTIVE,
 ANVIL_ATT, ANVIL_DET, CIRRUS) = range(8)

CLASSES = [
    {"id": CLEAR,      "key": "clear",      "name": "Clear",           "color": "#FFFFFF"},
    {"id": STRATIFORM, "key": "stratiform", "name": "Stratiform",      "color": "#5C7A99"},
    {"id": CUMULUS,    "key": "cumulus",    "name": "Cumulus",         "color": "#E0A83C"},
    {"id": TCU,        "key": "tcu",        "name": "Towering cumulus","color": "#B0700F"},
    {"id": CONVECTIVE, "key": "convective", "name": "Convective",      "color": "#A11D33"},
    {"id": ANVIL_ATT,  "key": "anvil_att",  "name": "Anvil, attached", "color": "#E2703A"},
    {"id": ANVIL_DET,  "key": "anvil_det",  "name": "Anvil, detached", "color": "#F0A87E"},
    {"id": CIRRUS,     "key": "cirrus",     "name": "Cirrus",          "color": "#9EC0DC"},
]
# Tuned for a WHITE map: the pale ice colours darkened to survive on paper.
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


def model_files(date_str, cycle, fh, model=None):
    """(url, role) for one forecast hour. role says what to pull out of that file."""
    return MODELS[model or MODEL]["files"](date_str, cycle, fh)


def run_hours(cycle):
    """Forecast hours to attempt for this cycle."""
    n = EXTENDED_RUN_H if int(cycle) in EXTENDED_CYCLES else SHORT_RUN_H
    return list(range(1, n + 1))


def find_cycle(sess):
    """Newest cycle at least CYCLE_LAG_H old whose f01 index is posted."""
    now = datetime.datetime.now(datetime.timezone.utc)
    probe = MODELS[MODEL]["probe"]
    for back in range(CYCLE_LAG_H, MAX_CYCLE_LOOKBACK_H + 1):
        t = now - datetime.timedelta(hours=back)
        d, cc = t.strftime("%Y%m%d"), t.strftime("%H")
        if not MODELS[MODEL]["hourly"] and int(cc) not in EXTENDED_CYCLES:
            continue
        try:
            url = model_files(d, cc, 1)[0][0]
            r = sess.get(url + ".idx", timeout=15)
            if r.status_code == 200 and probe in r.text:
                return d, cc, t.replace(minute=0, second=0, microsecond=0)
        except Exception:
            pass
    return None, None, None


def _pull(sess, out, url, want_levels, want_refc):
    """Byte-range the wanted messages out of one GRIB file. Returns bytes written."""
    lvl_re = re.compile(r"^(\d+)\s*mb$")
    try:
        r = sess.get(url + ".idx", timeout=20)
    except Exception:
        return 0
    if r.status_code != 200:
        return 0
    want = []
    for e in _parse_idx(r.text):
        if want_levels:
            m = lvl_re.match(e["level"].strip())
            if m:
                lv = LEVEL_SETS.get(e["short"])
                if lv and int(m.group(1)) in lv:
                    want.append(e)
                    continue
        if want_refc and e["short"] == "REFC" and "entire atmosphere" in e["level"]:
            want.append(e)
    if not want:
        return 0
    total = 0
    for s, e in _merge(want):
        rng = f"bytes={s}-{'' if e is None else e}"
        try:
            rr = sess.get(url, headers={"Range": rng}, timeout=90)
        except Exception:
            continue
        if rr.status_code in (200, 206):
            out.write(rr.content)
            total += len(rr.content)
    return total


def fetch_hour(sess, date_str, cycle, fh):
    """Byte-range the fields needed for one forecast hour. Returns a local GRIB path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, f"{MODEL}_{cycle}z_f{fh:02d}.grib2")
    total = 0
    with open(local, "wb") as out:
        for url, role in model_files(date_str, cycle, fh):
            total += _pull(sess, out, url,
                           want_levels="levels" in role, want_refc="refc" in role)
    return (local, total) if total else (None, 0)


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


def classify(f, prior_age=None):
    """One cloud type per grid column. Returns (class_grid, diagnostics).

    `prior_age` is the previous forecast hour's outflow age on this same grid, which makes
    the anvil label survive its parent: classified hour by hour with no memory, a shield
    goes from anvil to cirrus the instant its last 40 dBZ core drops below threshold, even
    though the ice is plainly the same ice. Carrying the age forward starts the LLCC clock
    at the core's death instead, so the shield ages out over three hours the way a real one
    does.
    """
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

    # A glaciated layer whose TOP reached homogeneous freezing came out of the top of
    # something. Base temperature is deliberately not used: an attached or thick anvil
    # commonly has a base warmer than -20 C, and v1 discarded exactly those.
    ice_aloft = has_cloud & (ice_frac_hi >= ICE_FRAC) & (top_c <= GLACIATED_C)

    # --- convective cores ---------------------------------------------------------------
    core = (f["refc"] >= CONV_DBZ) | (gcol >= GRAUPEL_CONV)
    r_px = max(1, int(round(ATTACH_NM * 1.852 / DX_KM)))
    near_core = maximum_filter(core.astype(float), size=2 * r_px + 1) > 0.5

    # --- outflow wind -------------------------------------------------------------------
    # Ice-mass-weighted mean through the top layer, not a fixed 300-150 mb mean: the
    # detrainment level moves with the EL, so the shield is steered by where its own ice is.
    wt = ice * hi_lay
    wsum = wt.sum(axis=0)
    fallback = hi_lay.sum(axis=0)
    u_lay = np.where(wsum > 0, (f["u"] * wt).sum(axis=0) / np.maximum(wsum, 1e-9),
                     (f["u"] * hi_lay).sum(axis=0) / np.maximum(fallback, 1))
    v_lay = np.where(wsum > 0, (f["v"] * wt).sum(axis=0) / np.maximum(wsum, 1e-9),
                     (f["v"] * hi_lay).sum(axis=0) / np.maximum(fallback, 1))
    # Columns with no ice layer - the clear gaps a detached shield has drifted across - would
    # otherwise carry zero steering wind and stop the trace dead at the first clear cell.
    # Fall back to the mean wind through the outflow layer, which exists everywhere.
    out_lv = np.asarray(f["levels"]) <= 300.0
    if out_lv.any():
        u_out, v_out = f["u"][out_lv].mean(axis=0), f["v"][out_lv].mean(axis=0)
    else:
        u_out = v_out = np.zeros_like(u_lay)
    got_ice = wsum > 0
    u_lay = np.where(got_ice, u_lay, u_out)
    v_lay = np.where(got_ice, v_lay, v_out)
    spd = np.hypot(u_lay, v_lay)
    ux = np.where(spd > 0.5, u_lay / np.maximum(spd, 1e-6), 0.0)
    uy = np.where(spd > 0.5, v_lay / np.maximum(spd, 1e-6), 0.0)

    # Spread outflow age DOWNWIND through the ice shield instead of firing a straight ray
    # upwind from every column. The ray version asked "is there a core exactly this many km
    # up-flow of me, in a straight line, using my local wind" - which fails for a wide shield
    # whose flow curves, and for the leading edge of a shield whose core sits 200+ km back.
    # Those columns fell through to CIRRUS even though the ice plainly came out of a tower.
    #
    # Here the source label is seeded at the cores and walked forward one grid cell at a
    # time along the local wind, but only through glaciated cloud. Anvil is a physically
    # continuous ice shield, so connectivity is the right constraint; the flow can bend as
    # much as it likes and the label follows it. Each step adds dx/speed hours, so the age
    # carried is the along-path travel time rather than a straight-line distance cap.
    yy, xx = np.mgrid[0:ny, 0:nx]
    NEVER = 99.0                      # stands in for "unreachable" and keeps the array finite

    # Exact 8-neighbour steps, taken with nearest-neighbour sampling. Interpolating the age
    # field blends labelled cells against the NEVER sentinel at the frontier, so the age
    # creeps up with every iteration and a shield 60 km long reports 38 hours old.
    m = np.maximum(np.abs(ux), np.abs(uy))
    sx = np.where(m > 0, np.round(ux / np.maximum(m, 1e-9)), 0.0)
    sy = np.where(m > 0, np.round(uy / np.maximum(m, 1e-9)), 0.0)
    step_km = np.hypot(sx, sy) * DX_KM                 # dx or dx*sqrt(2) on the diagonals
    dt_h = np.where((spd > 0.5) & (step_km > 0), step_km / np.maximum(spd * 3.6, 1e-6), NEVER)
    n_steps = int(min(np.ceil(ANVIL_MAX_NM * 1.852 / DX_KM), 120))

    # Advect last hour's age one hour downwind and add an hour to it. Where a core is firing
    # the clock resets to zero regardless.
    seed = np.where(core, 0.0, NEVER)
    if prior_age is not None and prior_age.shape == seed.shape:
        shift = np.where(spd > 0.5, spd * 3.6 / DX_KM, 0.0)     # cells travelled in 1 h
        carried = map_coordinates(np.nan_to_num(prior_age, nan=NEVER, posinf=NEVER),
                                  [yy - uy * shift, xx - ux * shift], order=0, mode="nearest")
        seed = np.minimum(seed, np.where(carried < AGE_MAX_H, carried + 1.0, NEVER))

    def _spread(gate):
        """Walk outflow age downwind from the seeds, one grid cell per step, following the
        local wind. `gate` is where the label may travel."""
        age = seed.copy()
        for _ in range(n_steps):
            upstream = map_coordinates(age, [yy - sy, xx - sx], order=0, mode="nearest")
            cand = np.where(gate, upstream + dt_h, NEVER)
            nxt = np.minimum(age, np.minimum(cand, NEVER))
            if np.array_equal(nxt, age):
                break
            age = nxt
        return age

    # CONNECTED: only through glaciated cloud, one cell of slack for a ragged edge. This is
    # "is this column part of an ice shield that reaches back to a core", which is what makes
    # an anvil attached rather than merely downwind of something. No age limit - the LLCC
    # writes the attached-anvil rule about distance, not age.
    passable = maximum_filter(ice_aloft.astype(np.uint8), size=3) > 0
    age_conn = _spread(passable)

    # FREE: through clear air as well, so a shield that has genuinely separated from its
    # parent still gets found. This one is on the 3-hour clock.
    age_free = _spread(np.ones_like(passable))

    live_core = core.any()
    joined = (age_conn <= CONN_MAX_H) & live_core   # continuous ice back to a LIVE core
    recent = age_free <= ANVIL_TAU_H                # left a core within 3 h, detached or not
    sourced = joined | recent
    age_h = np.minimum(age_conn, age_free)

    # --- decision -----------------------------------------------------------------------
    anvil = ice_aloft & (iwp_hi >= ANVIL_IWP) & (near_core | sourced)
    cirrus = ice_aloft & ~anvil

    liq_base = has_cloud & (liq_frac_lo > 0.5)
    tcu = liq_base & (lo_top_c <= TCU_TOP_C)
    cu = liq_base & ~tcu & ((depth_lo >= CU_DEPTH_KFT) | (tex >= CU_TEX_KFT))
    # Anything cloudy that matched nothing above - typically a mid-level glaciated deck
    # topping warmer than -38 C, ice-dominated so not liquid-based either - was silently
    # falling through to CLEAR. Layered cloud is the honest default for it.
    unclaimed = has_cloud & ~ice_aloft & ~liq_base
    st = (liq_base & ~tcu & ~cu) | unclaimed

    out = np.full((ny, nx), CLEAR, dtype=np.uint8)
    for mask, cid in ((cirrus, CIRRUS), (st, STRATIFORM), (cu, CUMULUS),
                      (tcu, TCU), (anvil & ~near_core, ANVIL_DET), (anvil & near_core, ANVIL_ATT),
                      (core, CONVECTIVE)):
        out[mask] = cid                                    # ascending operational significance

    # Depth of cloud sitting in the 0 to -20 C band, which is what the Thick Cloud Layer
    # rule is written about - the mixed-phase region where a vehicle can trigger a strike.
    dz = np.abs(np.gradient(z, axis=0))
    in_band = cloud & (tmpc_band := (f["tmpc"] <= 0.0) & (f["tmpc"] >= -20.0))
    thick_kft = (dz * in_band).sum(axis=0)

    diag = {"top_kft": top_kft, "top_c": top_c, "thick_kft": thick_kft, "iwp": iwp_hi, "lwp": lwp_lo,
            "depth_lo": depth_lo, "graupel": gcol, "refc": f["refc"],
            "age_h": np.where(age_h >= NEVER, np.nan, age_h), "joined": joined}
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

    ax.text(0.012, 0.012, f"{MODELS[MODEL]['name']} {cycle}Z  valid {valid:%d %b %H%MZ}", transform=ax.transAxes,
            fontsize=6.4, color="#7A858E", family="monospace", zorder=7)
    try:
        ax.spines["geo"].set_edgecolor("#C3C8BC")
    except Exception:
        pass
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# LLCC evaluation and probability of violation
# --------------------------------------------------------------------------------------
# EVERY THRESHOLD BELOW IS A PLACEHOLDER TO BE CHECKED AGAINST THE CURRENT LLCC DOCUMENT.
# These are encoded from the commonly published form of the NASA/USSF Launch Commit Criteria
# and are almost certainly not exact - verify each against the controlling document before
# anyone treats the output as decision support. They are gathered here, in one dict, for
# exactly that reason.
#
# What can and cannot be evaluated from a model field is worth stating plainly. Rules that
# reduce to cloud geometry and distance - cumulus, anvil, thick layer, disturbed weather -
# are computable. Rules that depend on observation - the lightning rule, surface electric
# field mill readings, triboelectrification, smoke plumes, the "good visibility" clauses -
# are not, and no amount of model resolution changes that. A POV computed here is a floor,
# not a verdict.
LLCC = {
    # Cumulus rule: (standoff nm, cloud-top temperature threshold C). A cumulus whose top is
    # colder than the threshold puts everything inside that radius NO-GO.
    "cumulus": [(10.0, -20.0), (5.0, -10.0), (3.0, 0.0)],
    "attached_anvil_nm": 10.0,
    "detached_anvil_nm": 3.0,
    # Thick cloud layer: flight path through a layer this deep inside the 0 to -20 C band.
    "thick_layer_kft": 4.5,
    # Disturbed weather: moderate-or-greater precipitation within this radius, under cloud
    # whose top reaches the threshold. Composite reflectivity stands in for the precip.
    "disturbed_nm": 5.0, "disturbed_dbz": 30.0, "disturbed_top_c": -20.0,
}

RULE_KEYS = ["cumulus", "attached_anvil", "detached_anvil", "thick_layer", "disturbed"]


def _disc(nm, lat_deg, dlat, dlon):
    """Boolean footprint of a circle of radius `nm` on the regular lat/lon query mesh. The
    mesh is not square in kilometres, so the disc is an ellipse in index space."""
    km = nm * 1.852
    kl = dlat * 111.32
    ko = dlon * 111.32 * max(np.cos(np.radians(lat_deg)), 0.1)
    rj, ri = max(1, int(np.ceil(km / kl))), max(1, int(np.ceil(km / ko)))
    jj, ii = np.mgrid[-rj:rj + 1, -ri:ri + 1]
    return np.hypot(jj * kl, ii * ko) <= km


def llcc_violation(planes, q):
    """GO / NO-GO at every mesh point, treating that point as the pad. Returns a dict of
    per-rule boolean grids plus the union."""
    lat_mid = q["lat0"] + 0.5 * q["ny"] * q["dlat"]
    near = lambda mask, nm: maximum_filter(
        mask.astype(np.uint8), footprint=_disc(nm, lat_mid, q["dlat"], q["dlon"])) > 0

    cls = planes["class"]
    top_c = planes["top_c"]
    convective_like = np.isin(cls, [CUMULUS, TCU, CONVECTIVE])

    out = {}
    cum = np.zeros(cls.shape, bool)
    for nm, tc in LLCC["cumulus"]:
        cum |= near(convective_like & (top_c <= tc), nm)
    out["cumulus"] = cum
    out["attached_anvil"] = near(cls == ANVIL_ATT, LLCC["attached_anvil_nm"])
    out["detached_anvil"] = near(cls == ANVIL_DET, LLCC["detached_anvil_nm"])
    # Thick layer is a property of the column being flown through, not of a neighbour.
    out["thick_layer"] = planes["thick_kft"] >= LLCC["thick_layer_kft"]
    out["disturbed"] = (near(planes["dbz"] >= LLCC["disturbed_dbz"], LLCC["disturbed_nm"])
                        & (top_c <= LLCC["disturbed_top_c"]))
    out["any"] = np.logical_or.reduce([out[k] for k in RULE_KEYS])
    return out


def unpack_planes(blob, q):
    """Inverse of pack_query, back to physical units on the mesh."""
    a = np.frombuffer(blob, np.uint8)
    n = q["nx"] * q["ny"]
    g = lambda k: a[k * n:(k + 1) * n].reshape(q["ny"], q["nx"]).astype(float)
    age = g(6)
    return {"class": g(0).astype(int), "top_kft": g(1) / 4.0, "top_c": g(2) - 100.0,
            "iwp": 10 ** (g(3) * np.log10(1 + q["iwp_max"]) / 255.0) - 1.0,
            "depth_kft": g(4) / 4.0, "dbz": g(5),
            "age_h": np.where(age >= 255, np.nan, age / 10.0),
            "thick_kft": g(7) / 4.0}


# --------------------------------------------------------------------------------------
# Queryable export
# --------------------------------------------------------------------------------------
_QGRID = {}   # the crop is identical every hour, so the resampling is solved once per run

PLANES = ["class", "top_kft_x4", "top_c_p100", "iwp_log", "depth_kft_x4", "dbz",
          "age_h_x10",      # 255 = no outflow source found
          "thick_kft_x4"]   # cloud depth inside the 0 to -20 C band


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
        np.where(np.isfinite(take(diag["age_h"])),
                 np.clip(np.round(take(diag["age_h"]) * 10.0), 0, 254), 255).astype(np.uint8),
        q(np.nan_to_num(take(diag["thick_kft"]), nan=0.0) * 4.0),
    ]
    return b"".join(p.tobytes() for p in planes), len(qlon), len(qlat), qlat[0], qlon[0]


# --------------------------------------------------------------------------------------
def site_indices(f):
    idx = {}
    for name, (la, lo) in SITES.items():
        d = (f["lats"] - la) ** 2 + (f["lons"] - lo) ** 2
        idx[name] = np.unravel_index(np.argmin(d), d.shape)
    return idx


POV_DIR = os.path.join(OUT_DIR, "pov")

# Sequential, and deliberately not a rainbow: on a white map the eye should read "how dark"
# without decoding a hue.
POV_COLORS = ["#FFFFFF", "#FBE3D4", "#F6C3A4", "#EF9C74", "#E2703A", "#C24A2C", "#A11D33"]
POV_BOUNDS = [0, 5, 20, 40, 60, 80, 95, 100.01]


def render_pov(pov, q, valid, label, path):
    """Probability-of-violation map on the query mesh."""
    proj, size = _figsize()
    pc = ccrs.PlateCarree()
    lats = q["lat0"] + np.arange(q["ny"]) * q["dlat"]
    lons = q["lon0"] + np.arange(q["nx"]) * q["dlon"]
    cmap = mcolors.ListedColormap(POV_COLORS)
    norm = mcolors.BoundaryNorm(POV_BOUNDS, len(POV_COLORS))
    fig = plt.figure(figsize=size, dpi=140, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    _basemap(ax, DOMAIN)
    ax.pcolormesh(lons, lats, pov, cmap=cmap, norm=norm, shading="nearest",
                  transform=pc, zorder=2)
    ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines",
                                                "10m", facecolor="none"),
                   edgecolor="#AEB6BD", linewidth=0.6, zorder=4)
    for name, (la, lo) in SITES.items():
        ax.plot(lo, la, marker="+", markersize=6, markeredgewidth=1.3, color="#14181B",
                transform=pc, zorder=6)
    ax.text(0.012, 0.012, f"LLCC violation probability  valid {valid:%d %b %H%MZ}  {label}",
            transform=ax.transAxes, fontsize=6.4, color="#7A858E", family="monospace", zorder=7)
    try:
        ax.spines["geo"].set_edgecolor("#C3C8BC")
    except Exception:
        pass
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def build_pov(cycles, q, cid):
    """Probability of LLCC violation from a time-lagged ensemble.

    There is no public per-member feed to draw on: the operational REFS distribution carries
    only combined ensemble products, the prototype member feed stopped on 11 Aug 2026, and
    HREF publishes ensprod only and retires in October. So the members here are the retained
    HRRR cycles valid at the same time - a time-lagged ensemble, which is the same trick HREF
    itself uses for its NAM and HRRR members. Each cycle is one member; the spread is genuine
    run-to-run uncertainty, and it costs nothing extra to download.
    """
    if not q:
        return []
    os.makedirs(os.path.join(OUT_DIR, "pov"), exist_ok=True)
    newest = cycles[0]
    by_valid = {}
    for c in cycles:
        for fr in c["frames"]:
            by_valid.setdefault(fr["valid"], []).append((c, fr))

    t0 = datetime.datetime.strptime(newest["init"], "%Y-%m-%dT%H:%MZ")
    out = []
    for fr0 in newest["frames"]:
        members = [(c, fr) for c, fr in by_valid.get(fr0["valid"], [])
                   if (t0 - datetime.datetime.strptime(c["init"], "%Y-%m-%dT%H:%MZ"))
                   .total_seconds() / 3600.0 <= POV_MAX_AGE_H]
        if len(members) < POV_MIN_MEMBERS:
            continue
        stack, labels = [], []
        for c, fr in members:
            try:
                with open(os.path.join(OUT_DIR, fr["data"]), "rb") as fp:
                    p = unpack_planes(fp.read(), q)
            except Exception:
                continue
            stack.append(llcc_violation(p, q)["any"])
            labels.append(c["label"])
        if len(stack) < POV_MIN_MEMBERS:
            continue
        pov = 100.0 * np.mean(np.stack(stack), axis=0)
        valid = datetime.datetime.strptime(fr0["valid"], "%Y-%m-%dT%H:%MZ")
        png = f"pov/pov_{MODEL}_{cid}z_f{fr0['fh']:02d}.png"
        render_pov(pov, q, valid, f"{len(stack)} members", os.path.join(OUT_DIR, png))
        binrel = f"pov/pov_{MODEL}_{cid}z_f{fr0['fh']:02d}.bin"
        with open(os.path.join(OUT_DIR, binrel), "wb") as bf:
            bf.write(np.clip(np.round(pov), 0, 100).astype(np.uint8).tobytes())
        out.append({"fh": fr0["fh"], "valid": fr0["valid"],
                    "valid_short": fr0["valid_short"], "valid_label": fr0["valid_label"],
                    "image": png, "data": binrel, "members": labels,
                    "mean_pov": round(float(pov.mean()), 2)})
    return out


def load_manifest():
    try:
        with open(os.path.join(OUT_DIR, "manifest.json")) as fp:
            return json.load(fp)
    except Exception:
        return {}


def _state_path(cid, fh):
    return os.path.join(STATE_DIR, f"age_{MODEL}_{cid}z_f{fh:02d}.npy")


def save_age(cid, fh, age_h):
    """Persist the outflow age so a later top-up pass can pick the clock back up. Stored on
    the model grid as uint8 tenths of an hour; 255 means unreached."""
    q = np.where(np.isfinite(age_h), np.clip(age_h * 10.0, 0, 254), 255).astype(np.uint8)
    np.save(_state_path(cid, fh), q)


def load_age(cid, fh):
    try:
        q = np.load(_state_path(cid, fh))
    except Exception:
        return None
    return np.where(q >= 255, np.nan, q.astype(float) / 10.0)


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
            # Hour fh-1's age if we have it - from this pass, or from a file a previous
            # pass left behind. Its absence is not fatal; the hour just starts a fresh clock.
            cls, diag = classify(f, prior_age=load_age(cid, fh - 1))
            save_age(cid, fh, diag["age_h"])
            valid = cyc_dt + datetime.timedelta(hours=fh)
            png = f"maps/cloudtype_{MODEL}_{cid}z_f{fh:02d}.png"
            render(cls, f, valid, cycle, os.path.join(OUT_DIR, png))
            blob, qnx, qny, qlat0, qlon0 = pack_query(cls, diag, f)
            qmeta = {"nx": qnx, "ny": qny, "lat0": round(float(qlat0), 6),
                     "lon0": round(float(qlon0), 6), "dlat": QUERY_DEG, "dlon": QUERY_DEG,
                     "iwp_max": IWP_MAX, "planes": PLANES}
            binrel = f"data/cols_{MODEL}_{cid}z_f{fh:02d}.bin"
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
    os.makedirs(STATE_DIR, exist_ok=True)
    sess = _session()
    date_str, cycle, cyc_dt = find_cycle(sess)
    if not cycle:
        logging.error(f"No {MODELS[MODEL]['name']} cycle available; leaving the previous run in place.")
        return

    prev = load_manifest()
    force = os.environ.get("CLOUDSCOPE_FORCE", "").strip().lower() in ("1", "true", "yes")
    data_stale = prev.get("data_version") != DATA_VERSION
    render_stale = prev.get("render_version") != RENDER_VERSION
    stale = data_stale or render_stale
    if data_stale and prev:
        logging.info(f"Data version changed ({prev.get('data_version', 'pre-versioning')}"
                     f" -> {DATA_VERSION}); dropping retained cycles.")
    elif render_stale and prev:
        logging.info(f"Render version changed ({prev.get('render_version', 'pre-versioning')}"
                     f" -> {RENDER_VERSION}); rebuilding images, keeping the ensemble.")
    elif force:
        logging.info("CLOUDSCOPE_FORCE set; rebuilding the newest cycle.")
    # Only a DATA change invalidates what is on disk. A render change leaves the packed grids
    # perfectly readable, so the ensemble and dprog/dt survive it.
    prior = [] if (data_stale or force) else list(prev.get("cycles", []))

    cid = f"{date_str}{cycle}"
    if render_stale and not data_stale:
        # Images are stale but the packed grids are fine: throw away this cycle's frames so
        # they re-render, and leave every older cycle alone.
        prior = [c for c in prior if c["id"] != cid]
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
    keep_cids = {c["id"] for c in cycles}
    for d, ext in ((MAP_DIR, ".png"), (DATA_DIR, ".bin")):
        for fn in os.listdir(d):
            if fn.endswith(ext) and fn not in live:
                os.remove(os.path.join(d, fn))
                dropped += 1
    for fn in os.listdir(STATE_DIR):
        if fn.endswith(".npy") and not any(f"_{c}z_" in fn for c in keep_cids):
            os.remove(os.path.join(STATE_DIR, fn))
            dropped += 1

    newest = cycles[0]
    pov = build_pov(cycles, qmeta or prev.get("query"), newest["id"])
    live_pov = {os.path.basename(p["image"]) for p in pov} | {os.path.basename(p["data"]) for p in pov}
    povdir = os.path.join(OUT_DIR, "pov")
    if os.path.isdir(povdir):
        for fn in os.listdir(povdir):
            if fn not in live_pov:
                os.remove(os.path.join(povdir, fn))
                dropped += 1

    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "render_version": RENDER_VERSION, "data_version": DATA_VERSION,
        "model": MODELS[MODEL]["name"], "model_key": MODEL,
        "cycle": f"{newest['date']} {newest['hour']}Z",
        "domain": DOMAIN, "classes": CLASSES, "sites": list(SITES),
        "query": qmeta or prev.get("query"),
        "cycles": cycles,
        "pov": {"frames": pov, "rules": RULE_KEYS, "thresholds": LLCC,
                "colors": POV_COLORS, "bounds": POV_BOUNDS,
                "source": f"time-lagged {MODELS[MODEL]['name']} cycles"},
        "frames": newest["frames"],   # so an older viewer still works
        "thresholds": {"layer_path_min_gm2": LAYER_PATH_MIN, "glaciated_c": GLACIATED_C,
                       "ice_fraction": ICE_FRAC, "anvil_iwp_gm2": ANVIL_IWP,
                       "convective_dbz": CONV_DBZ,
                       "conn_max_h": CONN_MAX_H,
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
