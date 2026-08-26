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

    4. ATTACHED vs DETACHED. The LLCC treats those differently, so the classifier does too,
       and the test is CONTINUITY, not distance: attached = an unbroken ice shield running
       back to a live convective core, however far downwind that is; detached = ice the
       trajectory can reach within 3 h but with no continuous shield back to a parent.

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
import time
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
RRFS_NOMADS = "https://nomads.ncep.noaa.gov/pub/data/nccf/com"
RRFS_ROOT = "https://noaa-rrfs-pds.s3.amazonaws.com"   # retro imagery only since 26 Aug 2026
HIRESW_ROOT = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod"
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
        "probe": "CLMR", "verified": True, "keep_cycles": 6,
    },
    # HiResW: RULED OUT, and worth recording why so nobody tries again. The .conus.grib2
    # files do publish .idx and are affordable, but they are a 2-D product - probed 26 Aug
    # 2026, the ARW and FV3 CONUS files carry 42 variables and not one of them is condensate
    # on isobaric levels. TMP exists only at 2 m and 80 m, HGT only at cloud base, ceiling
    # and the wet-bulb-zero level, UGRD/VGRD only at 10 m, 80 m and the PBL. The classifier
    # needs CLMR and CIMIXR through the depth of the atmosphere, so HiResW can seed a
    # convective core from REFC and nothing else. NCEP publishes no pressure-level HiResW
    # product on NOMADS. Kept here, disabled, as a record of the dead end.
    "hiresw_arw": {
        "name": "HiResW ARW", "dx_km": 2.5, "hourly": False,
        "short_run_h": 48, "extended_run_h": 48,
        "files": lambda d, c, fh: [
            (f"{HIRESW_ROOT}/hiresw.{d}/hiresw.t{c}z.arw_2p5km.f{fh:02d}.conus.grib2",
             "levels+refc")],
        "probe": "CLMR", "verified": False, "blocked": "2-D product, no isobaric condensate",
    },
    "hiresw_fv3": {
        "name": "HiResW FV3", "dx_km": 2.5, "hourly": False,
        "short_run_h": 48, "extended_run_h": 48,
        "files": lambda d, c, fh: [
            (f"{HIRESW_ROOT}/hiresw.{d}/hiresw.t{c}z.fv3_2p5km.f{fh:02d}.conus.grib2",
             "levels+refc")],
        "probe": "CLMR", "verified": False, "blocked": "2-D product, no isobaric condensate",
    },
    "rrfs": {
        # NOMADS parallel feed, and it DOES publish .idx - byte-ranging works exactly as it
        # does for HRRR on S3. An earlier probe concluded otherwise and that was wrong: the
        # index requests were being throttled, and a 302 was read as "no such file". NOMADS
        # documents a 10 s spacing between fetches and bounces anything faster, so this model
        # carries its own pacing below. Measured in production at ~16 s per forecast hour.
        "name": "RRFS", "dx_km": 3.0, "hourly": True,
        "short_run_h": 18, "extended_run_h": 60,
        "files": lambda d, c, fh: [
            (f"{RRFS_NOMADS}/rrfs/para/rrfs.{d}/{c}/"
             f"rrfs.t{c}z.prslev.3km.f{fh:03d}.conus.grib2", "levels+refc"),
        ],
        "probe": "CLMR", "verified": True,
        # Four cycles, against HRRR's six: an RRFS hour costs ~16 s where an HRRR hour is a
        # couple of seconds off S3, so the marginal member is far more expensive. Four is
        # enough for RRFS to carry real weight in a ten-member pool without the pass running
        # long.
        "keep_cycles": 4,
        # Between files, and between the byte-ranges within one file. The second must stay
        # small: an hour needs ~100 ranges, and charging the file pause to each of them turns
        # a 20 second download into seven minutes of sleeping.
        "pause_s": 4.0, "range_pause_s": 0.15,
        # ~16 s/hour measured in production, so 18 hours needs ~5 minutes; 600 s leaves room
        # for a bad NOMADS day without risking the 45-minute Actions timeout.
        "budget_s": 600,
    },
}
# Models built each pass, in order. The first is the "primary" - the one whose cloud-type
# maps the viewer opens on. All of them contribute members to the POV.
MODEL_KEYS = [m.strip().lower() for m in
              os.environ.get("CLOUDSCOPE_MODELS", "hrrr,rrfs").split(",")
              if m.strip().lower() in MODELS and not MODELS[m.strip().lower()].get("blocked")]
if not MODEL_KEYS:
    MODEL_KEYS = ["hrrr"]

# MODEL is rebound as the pass walks the list. Most of the module reads it - file naming, the
# fetch pacing, run length, grid spacing - so rebinding one global beats threading a model
# argument through fifteen functions.
MODEL = MODEL_KEYS[0]


def set_model(key):
    """Point the module at one model for the duration of its build."""
    global MODEL, DX_KM, SHORT_RUN_H, EXTENDED_RUN_H
    MODEL = key
    DX_KM = MODELS[key]["dx_km"]
    SHORT_RUN_H = MODELS[key]["short_run_h"]
    EXTENDED_RUN_H = MODELS[key]["extended_run_h"]

DX_KM = MODELS[MODEL]["dx_km"]      # default for the selected model; classify() takes it
                                    # explicitly so a run can mix models of different mesh

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
KEEP_CYCLES = 6      # default; a model may override with its own "keep_cycles"


def keep_cycles(key=None):
    return MODELS[key or MODEL].get("keep_cycles", KEEP_CYCLES)

# How many older, still-incomplete cycles to top up on one pass, on top of the newest. Bounds
# the runtime when several runs were picked up early.
MAX_TOPUP_CYCLES = 2

# Below this many members a probability is not a probability, it is a deterministic flag.
POV_MIN_MEMBERS = 2

# Older cycles to build per pass while the ensemble is short-handed. Normally each pass only
# builds the newest cycle and the ensemble grows an hour at a time - fine in steady state,
# useless after a DATA_VERSION bump wipes the retained cycles and leaves two members
# producing a probability that can only read 0, 50 or 100%. Backfill pulls the older cycles
# that are still on S3 so the ensemble refills in a couple of passes instead of six. It only
# runs when short, so steady-state bandwidth is unchanged.
BACKFILL_PER_PASS = 2

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
DATA_VERSION = "2026.08.26-multimodel"
RENDER_VERSION = "2026.08.26-multimodel"

# ---- classification thresholds (all tunable; see README) ----
LAYER_PATH_MIN = 0.20   # g/m^2 of condensate in one layer to call it cloudy
GLACIATED_C    = -38.0  # homogeneous freezing: a top this cold is cirriform by construction
ICE_FRAC       = 0.80   # ice share WITHIN the layer for it to count as glaciated
ANVIL_IWP      = 50.0   # g/m^2 - optically substantial ice; anvils run 100-1000
CONV_DBZ       = 40.0   # composite reflectivity marking a convective core
GRAUPEL_CONV   = 200.0  # g/m^2 of graupel: riming implies an updraft holding supercooled water
ATTACH_NM      = 10.0   # LLCC standoff from an attached anvil; used by the POV rules, and
                        # deliberately NOT used to decide whether an anvil is attached
# Two different clocks, because the LLCC uses two. A shield still joined to its parent is an
# attached anvil no matter how long the ice has been streaming - the rule is written about
# distance from it, not its age. The 3-hour clock belongs to anvil that has SEPARATED and to
# detached anvil. Applying the 3 h cap to everything aged a live, connected shield to cirrus
# while its cores were still firing.
ANVIL_TAU_H    = 3.0    # detached anvil: hours since it left the core
CONN_MAX_H     = 6.0    # sanity cap on tracing a continuous shield, not an age limit
AGE_MAX_H      = 25.0   # ceiling of the stored age field, hours
TCU_TOP_C      = -10.0  # liquid-based layer glaciating at its top
CU_DEPTH_KFT   = 3.0    # depth separating cumuliform from a layered deck
CU_TEX_KFT     = 1.2    # or lumpiness: sigma of cloud-top height over ~15 km

# Debris is gone. In model land the difference between thinning anvil ice and cirrus is a
# guess about optical depth that HRRR's microphysics does not really support, so sourced ice
# is either optically substantial enough to be an anvil or it is cirrus.
# Convective folded into Cumulus. A 40 dBZ core is a cumuliform cloud, and NASA-STD-4010
# 4.1.3 already treats it as one - the cumulus rules key on cloud-top temperature, not on
# whether the cell is raining, so a Cb was tripping exactly the same criteria a deep cumulus
# does. The core mask itself is untouched: it still seeds the anvil trace and still shows up
# as composite reflectivity in the readout, it just no longer gets a colour of its own.
(CLEAR, STRATIFORM, CUMULUS, TCU, ANVIL_ATT, ANVIL_DET, CIRRUS) = range(7)

CLASSES = [
    {"id": CLEAR,      "key": "clear",      "name": "Clear",           "color": "#FFFFFF"},
    {"id": STRATIFORM, "key": "stratiform", "name": "Stratiform",      "color": "#5C7A99"},
    {"id": CUMULUS,    "key": "cumulus",    "name": "Cumulus",         "color": "#E0A83C"},
    {"id": TCU,        "key": "tcu",        "name": "Towering cumulus","color": "#B0700F"},
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


def _pull(sess, out, url, want_levels, want_refc, pause_s=0.0, range_pause_s=0.0):
    """Byte-range the wanted messages out of one GRIB file. Returns bytes written.

    `pause_s` and `range_pause_s` exist for NOMADS, which documents a 10 s spacing between
    fetches and answers anything faster with a 302. The two are separate on purpose: the
    ranges within one file are one logical transfer, so they take the small pause, while the
    gap between files takes the large one.
    """
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
    for n, (s, e) in enumerate(_merge(want)):
        if n and range_pause_s:
            time.sleep(range_pause_s)
        rng = f"bytes={s}-{'' if e is None else e}"
        try:
            rr = sess.get(url, headers={"Range": rng}, timeout=120)
        except Exception:
            continue
        if rr.status_code in (200, 206):
            out.write(rr.content)
            total += len(rr.content)
        elif rr.status_code in (301, 302, 403, 429):
            # Throttled mid-file. Backing off and retrying once is cheaper than losing the
            # hour, and far better than recording a truncated GRIB as a successful fetch.
            time.sleep(max(pause_s, 5.0))
            rr = sess.get(url, headers={"Range": rng}, timeout=120)
            if rr.status_code in (200, 206):
                out.write(rr.content)
                total += len(rr.content)
    return total


def fetch_hour(sess, date_str, cycle, fh):
    """Byte-range the fields needed for one forecast hour. Returns a local GRIB path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, f"{MODEL}_{cycle}z_f{fh:02d}.grib2")
    total = 0
    m = MODELS[MODEL]
    with open(local, "wb") as out:
        for n, (url, role) in enumerate(model_files(date_str, cycle, fh)):
            if n and m.get("pause_s"):
                time.sleep(m["pause_s"])
            total += _pull(sess, out, url,
                           want_levels="levels" in role, want_refc="refc" in role,
                           pause_s=m.get("pause_s", 0.0),
                           range_pause_s=m.get("range_pause_s", 0.0))
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


def classify(f, prior_age=None, dx_km=None):
    """One cloud type per grid column. Returns (class_grid, diagnostics).

    `dx_km` is the model's own grid spacing - HiResW is 2.5 km where HRRR is 3 km, and it
    feeds the texture filter, the core dilation and the trajectory step length, so it cannot
    be a module-level constant once more than one model is in play.

    `prior_age` is the previous forecast hour's outflow age on this same grid, which makes
    the anvil label survive its parent: classified hour by hour with no memory, a shield
    goes from anvil to cirrus the instant its last 40 dBZ core drops below threshold, even
    though the ice is plainly the same ice. Carrying the age forward starts the LLCC clock
    at the core's death instead, so the shield ages out over three hours the way a real one
    does.
    """
    dx_km = float(dx_km or DX_KM)
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
    n_tex = max(3, int(round(15.0 / dx_km)) | 1)
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
    step_km = np.hypot(sx, sy) * dx_km                 # dx or dx*sqrt(2) on the diagonals
    dt_h = np.where((spd > 0.5) & (step_km > 0), step_km / np.maximum(spd * 3.6, 1e-6), NEVER)
    # Enough steps to cross the domain, because the AGE caps are what limit the trace now,
    # not a distance ceiling. Sizing this from a 150 nm reach was a leftover from the old
    # straight-ray version and silently truncated any shield longer than ~280 km, turning
    # the far end of a perfectly continuous anvil into cirrus. Converged runs exit early.
    n_steps = int(min(nx + ny, 400))

    # Advect last hour's age one hour downwind and add an hour to it. Where a core is firing
    # the clock resets to zero regardless.
    seed = np.where(core, 0.0, NEVER)
    if prior_age is not None and prior_age.shape == seed.shape:
        shift = np.where(spd > 0.5, spd * 3.6 / dx_km, 0.0)     # cells travelled in 1 h
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
    # Attached vs detached is about CONTINUITY with the parent, not distance from it. The
    # 10 nm in the LLCC is the standoff you keep from an anvil, not what makes one attached -
    # so an unbroken shield streaming 60 km downwind of a live core is still an attached
    # anvil, and this used to call it detached. `joined` is the continuous-ice trace; `recent`
    # is trajectory reach through clear air, which is precisely ice that has separated.
    anvil = ice_aloft & (iwp_hi >= ANVIL_IWP) & sourced
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
                      (tcu, TCU), (anvil & ~joined, ANVIL_DET), (anvil & joined, ANVIL_ATT),
                      (core, CUMULUS)):
        out[mask] = cid                                    # ascending operational significance

    # 4.1.8 measures a LAYER: base of the bottom to top of the uppermost, where any part of
    # it lies between 0 and -20 C. Summing cloudy depth inside the band instead - which is
    # what this did first - understates a 6 kft deck whose warm half sits below 0 C, and
    # would wrongly clear it. So each contiguous layer is measured whole, and kept only if
    # it intersects the band.
    band = (f["tmpc"] <= 0.0) & (f["tmpc"] >= -20.0)
    thick_kft = np.zeros((ny, nx))
    run_base = np.zeros((ny, nx))
    run_band = np.zeros((ny, nx), bool)
    was = np.zeros((ny, nx), bool)
    for k in range(nlev):
        c = cloud[k]
        start = c & ~was
        run_base = np.where(start, z[k], run_base)
        run_band = np.where(start, band[k], run_band | (c & band[k]))
        if k:
            ended = was & ~c
            thick_kft = np.where(ended & run_band,
                                 np.maximum(thick_kft, z[k - 1] - run_base), thick_kft)
        was = c
    thick_kft = np.where(was & run_band,
                         np.maximum(thick_kft, z[nlev - 1] - run_base), thick_kft)

    # "Located entirely at altitudes where the temperature is colder than 0 C" is a statement
    # about a cloud's BASE, and it is the exception that decides most anvil calls.
    base_c_hi = np.where(has_cloud, _at(f["tmpc"], hi_base), np.nan)

    diag = {"top_kft": top_kft, "top_c": top_c, "thick_kft": thick_kft, "base_c": base_c_hi, "iwp": iwp_hi, "lwp": lwp_lo,
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
# Encoded from NASA-STD-4010 (2017-06-27), "NASA Standard for Lightning Launch Commit
# Criteria for Space Flight". Requirement numbers below are that document's [LLCCR n].
#
# WHAT A MODEL CAN AND CANNOT DECIDE
#   Evaluable here: the criteria that reduce to cloud geometry, cloud-top and cloud-base
#   temperature, layer thickness, and radar reflectivity - Cumulus (4.1.3), Attached Anvil
#   (4.1.4), Detached Anvil (4.1.5), Disturbed Weather (4.1.7), Thick Cloud Layers (4.1.8).
#
#   Not evaluable, at all, ever, from a forecast field: Lightning (4.1.1) and every "wait 30
#   minutes / 3 hours / 4 hours after every lightning discharge" clause hanging off the anvil
#   rules; Surface Electric Fields (4.1.2), which needs field mills; Smoke Plumes (4.1.9);
#   Debris Clouds (4.1.6), which needs an observed detachment or collapse time. Several of
#   the field-mill EXCEPTIONS that would let a launch proceed are equally unevaluable, so
#   this errs toward NO-GO where the standard would allow a mill to buy relief.
#
#   The upshot: a POV computed here is a floor. Rules it cannot see can only make the real
#   answer worse, never better - except where an unevaluable exception would have granted
#   GO. It is a planning aid, not a launch decision.
#
# MRR (4.2.2): the standard defines it over a box +/-3 nmi horizontally, from the 0 C level
# to 20 km. With composite reflectivity only, 4.2.2c applies - the largest composite
# reflectivity within 4 nmi of the evaluation point - which is what is used here.
LLCC = {
    # --- 4.1.3 Cumulus. Excludes cirrocumulus, altocumulus, stratocumulus, and does not
    # apply to an anvil attached to a parent cumulus.
    "cumulus_through_c":   5.0,    # LLCCR 15: through cloud, top at or colder than +5 C
    "cumulus_through_hard_c": -5.0,  # colder than -5 C removes the field-mill exception
    "cumulus_5nm_c":     -10.0,    # LLCCR 16: 0-5 nmi, top at or colder than -10 C
    "cumulus_10nm_c":    -20.0,    # LLCCR 17: 5-10 nmi, top at or colder than -20 C
    "cumulus_5nm":         5.0,
    "cumulus_10nm":       10.0,

    # --- 4.1.4 Attached anvil, parent top at or colder than -10 C at any time.
    "attached_3nm":        3.0,    # LLCCR 18: within 3 nmi, NO-GO unless both exceptions
    "attached_excep_c":    0.0,    # portion within 5 nmi entirely colder than 0 C
    "attached_excep_nm":   5.0,
    "mrr_max_dbz":         7.5,    # MRR < +7.5 dBZ within 1 nmi
    "mrr_nm":              4.0,    # 4.2.2c evaluation radius for MRR itself
    "mrr_eval_nm":         1.0,

    # --- 4.1.5 Detached anvil.
    "detached_3nm":        3.0,    # LLCCR 22, minus its lightning clocks

    # --- 4.1.7 Disturbed weather: through cloud whose tops are colder than 0 C, with
    # moderate-or-greater precipitation within 5 nmi. "Moderate" is 30 dBZ by definition.
    "disturbed_nm":        5.0, "disturbed_dbz": 30.0, "disturbed_top_c": 0.0,

    # --- 4.1.8 Thick cloud layers. Does NOT apply to attached or detached anvil.
    "thick_kft":           4.5,    # 1.4 km
    "thick_band_c":       (0.0, -20.0),
    "thick_connect_nm":    5.0,    # LLCCR 28b: connected thick layer within 5 nmi
    "thick_cirriform_c": -15.0,    # LLCCR 29: cirriform, entirely colder than -15 C, exempt
    # LLCCR 30 exempts a layer with no 0 dBZ within 5 nmi. The packed reflectivity plane
    # clips at 0, so testing ">= 0" made every point echo-bearing and the exemption could
    # never fire. A packed 0 means "no echo", hence the strictly-greater test.
    "thick_min_dbz":       0.5,

    # --- 4.1.10 Triboelectrification: through any cloud colder than -10 C below 910 m/s.
    # Vehicle-dependent (LLCCR 33 exempts treated vehicles), so off by default.
    "tribo_enabled":     False, "tribo_c": -10.0,
}

RULE_KEYS = ["cumulus_through", "cumulus_5nm", "cumulus_10nm", "attached_anvil",
             "detached_anvil", "disturbed", "thick_layer"]
RULE_NAMES = {
    "cumulus_through": "Cumulus, flight through (4.1.3.1)",
    "cumulus_5nm":     "Cumulus within 5 nmi (4.1.3.2)",
    "cumulus_10nm":    "Cumulus 5-10 nmi (4.1.3.3)",
    "attached_anvil":  "Attached anvil within 3 nmi (4.1.4.1)",
    "detached_anvil":  "Detached anvil within 3 nmi (4.1.5.2)",
    "disturbed":       "Disturbed weather (4.1.7)",
    "thick_layer":     "Thick cloud layer (4.1.8)",
    "tribo":           "Triboelectrification (4.1.10)",
}
# Stated in the UI so nobody mistakes the number for a full evaluation.
RULES_NOT_EVALUATED = [
    "Lightning (4.1.1) and every lightning wait clause on the anvil rules",
    "Surface electric fields (4.1.2) - needs field mills",
    "Debris clouds (4.1.6) - needs an observed detachment or collapse time",
    "Smoke plumes (4.1.9)",
    "Field-mill exceptions that would otherwise permit launch",
]


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
    """GO / NO-GO at every mesh point, treating that point as the flight path.

    Returns per-rule boolean grids plus their union. Each rule is applied as written in
    NASA-STD-4010, with the exceptions that a model CAN evaluate honoured - the temperature
    and MRR exceptions on the anvil rules matter a great deal in practice, and ignoring them
    would paint most of a summer afternoon NO-GO.
    """
    L = LLCC
    lat_mid = q["lat0"] + 0.5 * q["ny"] * q["dlat"]
    near = lambda mask, nm: maximum_filter(
        mask.astype(np.uint8), footprint=_disc(nm, lat_mid, q["dlat"], q["dlon"])) > 0
    peak = lambda field, nm: maximum_filter(
        field, footprint=_disc(nm, lat_mid, q["dlat"], q["dlon"]))

    cls, top_c, base_c = planes["class"], planes["top_c"], planes["base_c"]
    dbz, thick = planes["dbz"], planes["thick_kft"]
    out = {}

    # 4.2.2c - MRR as the largest composite reflectivity within 4 nmi, then the rule tests
    # it within 1 nmi of the flight path.
    mrr = peak(dbz, L["mrr_nm"])
    mrr_ok = peak(mrr, L["mrr_eval_nm"]) < L["mrr_max_dbz"]

    # --- 4.1.3 Cumulus ------------------------------------------------------------------
    # Section applies to cumuliform cloud only, and explicitly not to attached anvil.
    cumuliform = np.isin(cls, [CUMULUS, TCU])
    # LLCCR 15: through the cloud. Top at or colder than +5 C is NO-GO; the field-mill
    # exception only exists for tops warmer than -5 C and cannot be evaluated here, so
    # anything at or colder than +5 C is taken as NO-GO.
    out["cumulus_through"] = cumuliform & (top_c <= L["cumulus_through_c"])
    # LLCCR 16 / 17: standoff by cloud-top temperature.
    out["cumulus_5nm"] = near(cumuliform & (top_c <= L["cumulus_5nm_c"]), L["cumulus_5nm"])
    out["cumulus_10nm"] = near(cumuliform & (top_c <= L["cumulus_10nm_c"]), L["cumulus_10nm"])

    # --- 4.1.4 Attached anvil -----------------------------------------------------------
    # LLCCR 18: within 3 nmi is NO-GO unless the anvil within 5 nmi sits entirely colder
    # than 0 C AND MRR < 7.5 dBZ within 1 nmi. Both are evaluable: the base temperature of
    # the ice layer is what "located entirely at altitudes colder than 0 C" means.
    att = cls == ANVIL_ATT
    att_warm = near(att & (base_c >= L["attached_excep_c"]), L["attached_excep_nm"])
    out["attached_anvil"] = near(att, L["attached_3nm"]) & (att_warm | ~mrr_ok)

    # --- 4.1.5 Detached anvil -----------------------------------------------------------
    # LLCCR 22 with the same evaluable exceptions; its 30-minute and 3-hour lightning clocks
    # are not evaluable and are noted rather than applied.
    det = cls == ANVIL_DET
    det_warm = near(det & (base_c >= L["attached_excep_c"]), L["attached_excep_nm"])
    out["detached_anvil"] = near(det, L["detached_3nm"]) & (det_warm | ~mrr_ok)

    # --- 4.1.7 Disturbed weather --------------------------------------------------------
    # Through non-transparent cloud with tops colder than 0 C, with moderate or greater
    # precipitation within 5 nmi. 30 dBZ is the standard's own definition of moderate.
    in_cloud = cls != CLEAR
    out["disturbed"] = (in_cloud & (top_c < L["disturbed_top_c"])
                        & near(dbz >= L["disturbed_dbz"], L["disturbed_nm"]))

    # --- 4.1.8 Thick cloud layers -------------------------------------------------------
    # Not applicable to anvil of either kind (4.1.8 preamble). LLCCR 29 exempts a cirriform
    # layer entirely colder than -15 C with no liquid water; LLCCR 30 exempts a layer with
    # no 0 dBZ within 5 nmi.
    thick_hit = thick >= L["thick_kft"]
    is_anvil = np.isin(cls, [ANVIL_ATT, ANVIL_DET])
    cirriform_exempt = (cls == CIRRUS) & (top_c <= L["thick_cirriform_c"])
    no_echo_exempt = ~near(dbz >= L["thick_min_dbz"], L["thick_connect_nm"])
    out["thick_layer"] = (near(thick_hit, L["thick_connect_nm"]) & in_cloud
                          & ~is_anvil & ~cirriform_exempt & ~no_echo_exempt)

    # --- 4.1.10 Triboelectrification ----------------------------------------------------
    if L["tribo_enabled"]:
        out["tribo"] = in_cloud & (top_c <= L["tribo_c"])

    keys = [k for k in RULE_KEYS if k in out] + (["tribo"] if "tribo" in out else [])
    out["any"] = np.logical_or.reduce([out[k] for k in keys])
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
            "thick_kft": g(7) / 4.0, "base_c": g(8) - 100.0}


# --------------------------------------------------------------------------------------
# Queryable export
# --------------------------------------------------------------------------------------
_QGRID = {}   # the crop is identical every hour, so the resampling is solved once per run

PLANES = ["class", "top_kft_x4", "top_c_p100", "iwp_log", "depth_kft_x4", "dbz",
          "age_h_x10",      # 255 = no outflow source found
          "thick_kft_x4",   # whole-layer depth where the layer meets the 0/-20 C band
          "base_c_p100"]    # base temperature of the upper cloud layer


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
        q(np.nan_to_num(take(diag["base_c"]), nan=-100.0) + 100.0),
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


def build_pov(out_models, q):
    """Probability of LLCC violation, pooled across every model and every retained cycle.

    Members are (model, cycle) pairs valid at the same time. HRRR contributes six hourly
    cycles off S3, RRFS four off NOMADS - so a typical valid time is scored by ten members
    carrying both run-to-run and model-to-model spread, rather than time-lagging alone.

    Pooling across models is only sound because pack_query resamples every model onto the
    SAME regular lat/lon mesh. HRRR at 3 km and RRFS at 3 km land on identical cells, so the
    per-member NO-GO grids are directly comparable without any regridding here.
    """
    if not q:
        return []
    os.makedirs(os.path.join(OUT_DIR, "pov"), exist_ok=True)

    # Newest initialisation anywhere is the reference for member age.
    all_cycles = [(k, c) for k, cyc in out_models.items() for c in cyc]
    if not all_cycles:
        return []
    t0 = max(datetime.datetime.strptime(c["init"], "%Y-%m-%dT%H:%MZ") for _, c in all_cycles)

    by_valid = {}
    for k, c in all_cycles:
        age_h = (t0 - datetime.datetime.strptime(c["init"], "%Y-%m-%dT%H:%MZ")).total_seconds() / 3600.0
        if age_h > POV_MAX_AGE_H:
            continue
        for fr in c["frames"]:
            by_valid.setdefault(fr["valid"], []).append((k, c, fr))

    out = []
    for valid_s in sorted(by_valid):
        members = by_valid[valid_s]
        if len(members) < POV_MIN_MEMBERS:
            continue
        stack, labels, per_model = [], [], {}
        for k, c, fr in members:
            try:
                with open(os.path.join(OUT_DIR, fr["data"]), "rb") as fp:
                    p = unpack_planes(fp.read(), q)
            except Exception:
                continue
            stack.append(llcc_violation(p, q)["any"])
            labels.append(f"{MODELS[k]['name']} {c['label']}")
            per_model[k] = per_model.get(k, 0) + 1
        if len(stack) < POV_MIN_MEMBERS:
            continue
        pov = 100.0 * np.mean(np.stack(stack), axis=0)
        valid = datetime.datetime.strptime(valid_s, "%Y-%m-%dT%H:%MZ")
        stamp = valid.strftime("%Y%m%dT%H%MZ")
        png = f"pov/pov_{stamp}.png"
        render_pov(pov, q, valid, f"{len(stack)} members", os.path.join(OUT_DIR, png))
        binrel = f"pov/pov_{stamp}.bin"
        with open(os.path.join(OUT_DIR, binrel), "wb") as bf:
            bf.write(np.clip(np.round(pov), 0, 100).astype(np.uint8).tobytes())
        out.append({"valid": valid_s, "valid_short": valid.strftime("%HZ"),
                    "valid_label": valid.strftime("%HZ %a %d %b"),
                    "image": png, "data": binrel, "members": labels,
                    "by_model": {MODELS[k]["name"]: n for k, n in per_model.items()},
                    "mean_pov": round(float(pov.mean()), 2)})
    return out


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
    # Wall-clock budget, for the same reason the aviation dashboard has one: a throttled
    # source has no other way to end, and a short column from a finished run beats a hung job
    # that commits nothing.
    budget_s = MODELS[MODEL].get("budget_s")
    t_start = time.monotonic()
    for fh in todo:
        if budget_s and time.monotonic() - t_start > budget_s:
            logging.info(f"{MODELS[MODEL]['name']} budget of {budget_s}s spent after "
                         f"{made} hours; the rest top up on a later pass.")
            break
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
            cls, diag = classify(f, prior_age=load_age(cid, fh - 1),
                                 dx_km=MODELS[MODEL]["dx_km"])
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


def build_model(sess, key, prev_models, force, data_stale, render_stale):
    """Build one model's cycles for this pass. Returns (cycles, bytes, qmeta, n_new)."""
    set_model(key)
    date_str, cycle, cyc_dt = find_cycle(sess)
    name = MODELS[key]["name"]
    if not cycle:
        logging.warning(f"{name}: no cycle available this pass.")
        return list(prev_models.get(key, [])), 0, None, 0

    prior = [] if (data_stale or force) else list(prev_models.get(key, []))
    cid = f"{date_str}{cycle}"
    if render_stale and not data_stale:
        prior = [c for c in prior if c["id"] != cid]
    cyc_dts = {cid: cyc_dt}
    entries, bytes_total, qmeta, built = [], 0, None, 0
    keep = keep_cycles(key)

    todo = [(cid, date_str, cycle)]
    for c in prior:
        if c["id"] != cid and len(c["frames"]) < len(run_hours(c["hour"])):
            todo.append((c["id"], c["date"], c["hour"]))
            cyc_dts[c["id"]] = datetime.datetime.strptime(
                c["init"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=datetime.timezone.utc)
        if len(todo) >= MAX_TOPUP_CYCLES + 1:
            break

    have_ids = {c["id"] for c in prior} | {cid}
    short = keep - len(have_ids)
    if short > 0:
        added = 0
        for back in range(1, keep + MAX_CYCLE_LOOKBACK_H):
            if added >= min(BACKFILL_PER_PASS, short):
                break
            t = cyc_dt - datetime.timedelta(hours=back)
            bid = t.strftime("%Y%m%d%H")
            if bid in have_ids:
                continue
            todo.append((bid, t.strftime("%Y%m%d"), t.strftime("%H")))
            cyc_dts[bid] = t
            have_ids.add(bid)
            added += 1
        if added:
            logging.info(f"{name}: ensemble short ({len(prior) + 1}/{keep}); "
                         f"backfilling {added} older cycle(s).")

    for tid, tdate, thour in todo:
        existing = next((c["frames"] for c in prior if c["id"] == tid), [])
        frames, nbytes, qm, made = build_cycle(sess, tdate, thour, cyc_dts[tid], tid, existing)
        bytes_total += nbytes
        qmeta = qm or qmeta
        built += made
        if not frames:
            continue
        want = run_hours(thour)
        entries.append({"id": tid, "model": key, "label": f"{thour}Z",
                        "date": tdate, "hour": thour,
                        "init": cyc_dts[tid].strftime("%Y-%m-%dT%H:%MZ"),
                        "render_version": RENDER_VERSION,
                        "run_h": len(frames), "run_h_expected": len(want),
                        "complete": len(frames) >= len(want), "frames": frames})
        if made:
            logging.info(f"{name} {tid}Z: {len(frames)}/{len(want)} h, +{made} new.")

    merged = {c["id"]: c for c in prior}
    merged.update({e["id"]: e for e in entries})
    for c in merged.values():
        c.setdefault("model", key)
    cycles = sorted(merged.values(), key=lambda c: c["init"], reverse=True)[:keep]
    return cycles, bytes_total, qmeta, built


def main():
    os.makedirs(MAP_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    sess = _session()

    prev = load_manifest()
    force = os.environ.get("CLOUDSCOPE_FORCE", "").strip().lower() in ("1", "true", "yes")
    data_stale = prev.get("data_version") != DATA_VERSION
    render_stale = prev.get("render_version") != RENDER_VERSION
    if data_stale and prev:
        logging.info(f"Data version changed ({prev.get('data_version', 'pre-versioning')}"
                     f" -> {DATA_VERSION}); dropping retained cycles.")
    elif render_stale and prev:
        logging.info("Render version changed; rebuilding images, keeping the ensemble.")

    # Previous cycles, per model. Old single-model manifests keep working: their flat
    # "cycles" list is read as belonging to whichever model wrote it.
    prev_models = {k: v.get("cycles", []) for k, v in (prev.get("models") or {}).items()}
    if not prev_models and prev.get("cycles"):
        prev_models = {prev.get("model_key", "hrrr"): prev["cycles"]}

    out_models, bytes_total, qmeta, built = {}, 0, prev.get("query"), 0
    for key in MODEL_KEYS:
        cycles, nb, qm, made = build_model(sess, key, prev_models, force,
                                           data_stale, render_stale)
        bytes_total += nb
        qmeta = qm or qmeta
        built += made
        if cycles:
            out_models[key] = cycles

    if not out_models:
        logging.error("No model produced anything; manifest not rewritten.")
        return

    # Prune anything no longer referenced by any model.
    live, keep_cids = set(), set()
    for cycles in out_models.values():
        for c in cycles:
            keep_cids.add(c["id"])
            for fr in c["frames"]:
                live.add(os.path.basename(fr["image"]))
                live.add(os.path.basename(fr["data"]))
    dropped = 0
    for d, ext in ((MAP_DIR, ".png"), (DATA_DIR, ".bin")):
        for fn in os.listdir(d):
            if fn.endswith(ext) and fn not in live:
                os.remove(os.path.join(d, fn))
                dropped += 1
    for fn in os.listdir(STATE_DIR):
        if fn.endswith(".npy") and not any(f"_{c}z_" in fn for c in keep_cids):
            os.remove(os.path.join(STATE_DIR, fn))
            dropped += 1

    pov = build_pov(out_models, qmeta)
    live_pov = ({os.path.basename(p["image"]) for p in pov}
                | {os.path.basename(p["data"]) for p in pov})
    povdir = os.path.join(OUT_DIR, "pov")
    if os.path.isdir(povdir):
        for fn in os.listdir(povdir):
            if fn not in live_pov:
                os.remove(os.path.join(povdir, fn))
                dropped += 1

    primary = MODEL_KEYS[0] if MODEL_KEYS[0] in out_models else next(iter(out_models))
    newest = out_models[primary][0]
    n_members = sum(len(c) for c in out_models.values())
    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "render_version": RENDER_VERSION, "data_version": DATA_VERSION,
        "model": MODELS[primary]["name"], "model_key": primary,
        "models": {k: {"name": MODELS[k]["name"], "dx_km": MODELS[k]["dx_km"],
                       "cycles": v} for k, v in out_models.items()},
        "cycle": f"{newest['date']} {newest['hour']}Z",
        "domain": DOMAIN, "classes": CLASSES, "sites": list(SITES),
        "query": qmeta,
        "pov": {"frames": pov, "rules": RULE_KEYS, "rule_names": RULE_NAMES,
                "not_evaluated": RULES_NOT_EVALUATED,
                "standard": "NASA-STD-4010 (2017-06-27)", "thresholds": LLCC,
                "colors": POV_COLORS, "bounds": POV_BOUNDS,
                "source": " + ".join(f"{len(v)} {MODELS[k]['name']}"
                                     for k, v in out_models.items())},
        "cycles": out_models[primary],     # so an older viewer still works
        "frames": newest["frames"],
        "thresholds": {"layer_path_min_gm2": LAYER_PATH_MIN, "glaciated_c": GLACIATED_C,
                       "ice_fraction": ICE_FRAC, "anvil_iwp_gm2": ANVIL_IWP,
                       "convective_dbz": CONV_DBZ,
                       "graupel_gm2": GRAUPEL_CONV, "attach_nm": ATTACH_NM,
                       "anvil_tau_h": ANVIL_TAU_H, "conn_max_h": CONN_MAX_H,
                       "tcu_top_c": TCU_TOP_C, "cu_depth_kft": CU_DEPTH_KFT},
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=1)
    logging.info(f"+{built} frames, {bytes_total/1024/1024:.0f} MB; "
                 + "; ".join(f"{MODELS[k]['name']} {len(v)} cycles" for k, v in out_models.items())
                 + f"; {n_members} potential members; {len(pov)} POV frames; "
                 f"pruned {dropped} files.")


if __name__ == "__main__":
    main()
