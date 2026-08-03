# CloudScope

Classifies every grid column of an HRRR forecast into one cloud type over the Florida
spaceport corridor, and publishes an hour-by-hour map to GitHub Pages.

**The point:** a reflectivity map tells you where the convection is. This tells you what
*kind* of cloud is overhead — which is the question the launch cloud rules actually ask,
since the Thick Cloud Layer rule doesn't apply to cumulus or anvil, and anvil has rules of
its own.

---

## How a column gets classified

Cloud type is read from the model's **condensate fields** — `CLMR` (cloud water) and
`CIMIXR` (cloud ice) — not from a relative-humidity threshold.

That choice matters. An RH threshold is a proxy for "is there cloud here," and it fails
badly aloft: at cirrus temperatures the dewpoint depression collapses, so any model with a
moist bias near the tropopause paints cloud that isn't there. Condensate is what the model
actually predicts is in the air, and it hands you the liquid/ice split for free.

Each column gets exactly one label. Where a column qualifies as several, the more
operationally significant one is drawn:

| Type | Test | Colour |
|---|---|---|
| **Convective** | composite reflectivity ≥ 40 dBZ | `#A11D33` |
| **Anvil** | glaciated, base colder than −20 °C, point below 40 dBZ, **and** a ≥ 40 dBZ core upwind along the 300–150 mb flow within 100 nm | `#E2703A` |
| **Cumulus** | liquid-based deck ≥ 8 kft deep | `#D9A441` |
| **Stratiform** | liquid-based deck under 8 kft deep | `#5C7A99` |
| **Cirrus** | glaciated, no convective source upwind | `#BFD3E6` |
| **Clear** | no condensate above 1e-6 kg/kg | — |

**Cirrus and anvil are the same column physically** — ice cloud aloft. The only thing
separating them is whether a parent storm can be found upwind along the anvil-level flow.
That test is the interesting part of this tool: reflectivity is sampled every 8 nm from 12
to 100 nm *against* the 300–150 mb wind vector at each grid point, so debris is attributed
to a source rather than guessed at from height alone.

### Tuning

All thresholds are constants at the top of `cloud_classify.py`:

- `DEEP_KFT` (8.0) — the cumulus/stratiform split. Raise it if shallow cu is over-calling.
- `ICE_FRAC` (0.70) — how ice-dominated a column must be to count as glaciated.
- `ANVIL_FAR_NM` (100.0) — how far debris is allowed to have travelled.
- `CONV_DBZ` (40.0) — convective core threshold.
- `QC_MIN` (1e-6 kg/kg) — the "is there cloud" floor. The single most sensitive knob:
  lower it and thin cirrus appears everywhere.

The precedence order is in `classify()`, as the sequence of assignments — later writes win.

---

## Data path

HRRR from **AWS S3** (`noaa-hrrr-bdp-pds`), byte-ranged from each file's `.idx`.

NOMADS' `grib_filter` would be far cheaper in bytes because it crops server-side, but it
rate-limits per IP and starts answering `302 Over Rate Limit` once a runner has been busy.
S3 has no such limit. The cost of that reliability is that a byte-ranged HRRR message is
the whole CONUS grid, cropped to the domain after download.

Per forecast hour: `TMP`, `CLMR`, `CIMIXR` on 18 isobaric levels, `UGRD`/`VGRD` on the four
anvil-layer levels, and `REFC`. Adjacent byte ranges are merged, so ~100 messages become a
handful of requests.

Heights come from the standard atmosphere rather than the model's `HGT` field. Height is
only used for deck *depth* against an 8 kft threshold, where a hypsometric profile changes
nothing — and skipping `HGT` cuts about a fifth of the download.

---

## Deploying

1. Create the repo and push these files.
2. **Settings → Pages → Source: Deploy from a branch**, branch `main`, folder `/docs`.
3. **Actions → Update CloudScope → Run workflow** for a first run (the schedule is every
   3 h at :20).

The `docs/` directory ships with a **synthetic demo** so the page renders before the first
real run. It is labelled `SYNTHETIC DEMO` in the header and is overwritten the first time
the workflow succeeds.

### Notes

- `pygrib` needs the eccodes C library; the workflow installs it via apt.
- Runtime is dominated by transfer — roughly 50 MB per forecast hour, so ~900 MB for the
  default 18 hours. Trim `FORECAST_HOURS` if that's more than you want every 3 hours.
- Each run republishes all PNGs and force-updates the branch rather than rebasing, because
  rebasing regenerated binaries against a concurrent run produces add/add conflicts. The
  `concurrency` group means only one publish runs at a time.
- Committing PNGs hourly grows repo history permanently. If that becomes a problem, publish
  `docs/` to a `gh-pages` branch with `--force` instead so history doesn't accumulate.

---

## Caveats

- **This is a model diagnosis, not an observation.** It inherits HRRR's cloud microphysics
  wholesale, including its known tendency to over-produce thin ice cloud.
- **The anvil test needs a source in the domain.** An anvil streaming in from a storm
  outside the box, or from more than 100 nm away, reads as cirrus.
- **One label per column.** Cirrus over cumulus is common and only the more significant
  one is drawn — the coverage percentages in the key are shares of the *dominant* type, not
  total sky cover.
- Not an official product, and not a substitute for the 45th Weather Squadron.
