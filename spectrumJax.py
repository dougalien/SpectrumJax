"""
Spectrum Jax - present one or more volcano transects against the perfect shapes.

Upload files of distance and elevation along a line across a volcano. For each
one the app

  1. measures the shape of the outline (how bowed the sides are, how wide the
     flat top is),
  2. puts it on a map of perfect shapes (triangle, circle, square ...), and
  3. finds the perfect shape that fits it best and subtracts that from the real
     outline.

Both pictures are made as images, so they can be copied straight into a slide.

Run with:  py -m streamlit run spectrumJax.py

The measuring engine below (PROCESSING ... flank_profiles) is copied unchanged
from volcanoShape.py, the research app, so the numbers agree with it.
validation/validate_spectrum_jax.py checks that they still do.
"""

import base64
import io
import pathlib
import re

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="Spectrum Jax", page_icon="🌋", layout="wide")


# =====================================================================
# THE MEASURING ENGINE
# Copied unchanged from volcanoShape.py. Do not edit here: change it in the
# research app, then copy it again and run the validation script.
# =====================================================================
PROCESSING = {
    # Profile is resampled to this many evenly spaced points before measurement.
    # An even grid is required for the derivative and area steps to be stable.
    "resample_points": 2001,

    # Savitzky-Golay smoothing, as a fraction of profile length. Smoothing
    # stabilises the slope and the power-law fit; too much erases the summit.
    "smooth_fraction": 0.01,
    "smooth_polyorder": 3,

    # Basal boundary: each flank's base is taken as this quantile of that
    # flank's elevations, not its single lowest sample, then the HIGHER of the
    # two is used. A single stray low reading on a surrounding plain would
    # otherwise capture the measurement entirely.
    "base_quantile": 0.05,

    # Points within this fraction of the edifice height of the summit are
    # excluded from the log-log power-law fit: in log space the near-summit
    # points carry enormous leverage and a little noise drags b badly.
    "fit_exclusion": 0.05,

    # Summit plateau is the width of ground within this fraction of the
    # edifice height of the summit, as a fraction of basal width.
    "plateau_level": 0.05,

    # Two-pass measurement. The first pass locates the edifice on the whole
    # track; the track is then cropped to the edifice plus this fraction of its
    # basal width on each side, and resampled and smoothed again. Without the
    # crop, resampling and smoothing are set by the length of the TRACK, so the
    # same volcano reads differently on a short file and on a long one: a 10 km
    # cone on a track 200 times longer read B/H = 34 instead of 10.
    "window_margin": 0.30,

    # High ground separated by less than this fraction of the basal width is one
    # summit (the two rims of a crater or caldera). High ground further apart is
    # a separate peak, and is reported rather than averaged into the summit.
    "summit_merge": 0.25,

    # High ground is also one summit, at any distance, when the dip between the
    # pieces is shallower than this fraction of the relief: the ridges a smoothing
    # filter leaves at the edges of a steep-walled flat top.
    "saddle_depth": 0.15,
}


def _smooth(zr: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing at the fixed fraction of the profile length."""
    n = len(zr)
    w = int(max(5, round(n * PROCESSING["smooth_fraction"])))
    if w % 2 == 0:
        w += 1
    if w <= PROCESSING["smooth_polyorder"]:
        w = PROCESSING["smooth_polyorder"] + 2 + (PROCESSING["smooth_polyorder"] % 2)
    return savgol_filter(zr, w, PROCESSING["smooth_polyorder"]) if w < n else zr.copy()


def prepare_profile(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the whole track to an even grid and smooth. Returns (x, z_smoothed, z_raw)."""
    n = int(PROCESSING["resample_points"])
    xs = np.linspace(float(np.min(x)), float(np.max(x)), n)
    zr = np.interp(xs, x, z)
    return xs, _smooth(zr), zr


def prepare_window(x: np.ndarray, z: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample only the stretch from a to b, from the ORIGINAL samples, and smooth it."""
    n = int(PROCESSING["resample_points"])
    xs = np.linspace(a, b, n)
    zr = np.interp(xs, x, z)
    return xs, _smooth(zr), zr


def _summit_run(mask: np.ndarray, k0: int, x: np.ndarray, merge_gap: float,
                z: np.ndarray | None = None, shallow: float | None = None) -> tuple[int, int]:
    """Index span of the run of True samples containing k0.

    Only CONTIGUOUS high ground around the highest point is the summit: a second
    peak elsewhere on the edifice is not part of it, however high it stands.
    Two exceptions join a neighbouring run of high ground to the summit:

      * it lies within merge_gap (in x units): the two rims of a crater;
      * the ground between the two never drops below `shallow` (an elevation):
        a flat top whose edges ring or bulge. A smoothing filter overshoots at a
        steep wall, which puts a ridge at each edge of a flat-topped mesa, and
        without this the summit would sit on one ridge and the flat top be lost.
        A parasitic cone is not joined, because the valley beside it is deep.
    """
    n = len(mask)
    lo = hi = k0
    while True:
        while lo > 0 and mask[lo - 1]:
            lo -= 1
        while hi < n - 1 and mask[hi + 1]:
            hi += 1
        grew = False
        for step in (-1, 1):
            edge = lo if step < 0 else hi
            j = edge + step
            shallow_ok = z is not None and shallow is not None
            found = False
            while 0 <= j < n:
                if mask[j]:
                    found = True
                    break
                if shallow_ok and z[j] < shallow:
                    shallow_ok = False
                if not shallow_ok and abs(x[j] - x[edge]) > merge_gap:
                    break
                j += step
            if found and (abs(x[j] - x[edge]) <= merge_gap or shallow_ok):
                if step < 0:
                    lo = j
                else:
                    hi = j
                grew = True
        if not grew:
            return lo, hi


def measure_edifice(x: np.ndarray, z: np.ndarray, merge_gap: float = 0.0) -> tuple[dict, list]:
    """Measure a radial transect through a volcanic edifice.

    The profile is handled summit-down: the summit is located, each flank's base
    is estimated robustly, and the lower-lying of the two bases sets a single
    horizontal basal level. Everything is then measured between that level and
    the summit.

    Every returned quantity is size-independent apart from height and width
    themselves, so a 21 km Martian shield and a 2 km terrestrial cone are
    directly comparable.

    merge_gap is in x units: summit runs closer together than this are one summit
    (crater rims). Call analyse_profile rather than this function directly: it
    supplies the gap and the cropped window.
    """
    notes: list = []
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(x) < 32:
        return {}, ["Profile too short to measure."]

    # Centre the summit on the summit PLATEAU, not on the single highest sample.
    # On a flat-topped edifice argmax lands wherever noise happens to peak, and
    # distance-from-summit is then wrong for the opposite flank: a cone with a
    # plateau half its basal width read b = 0.52 instead of 1.0 before this fix.
    #
    # Only the CONTIGUOUS high ground around the highest point is the summit
    # (plus any crater-rim partner within merge_gap). Taking the midpoint of every
    # sample above the band, wherever it lies, let one parasitic cone at 96% of
    # the summit height drag the summit onto the flank: b fell from 1.0 to 0.39.
    z_top = float(np.max(z))
    z_rng = z_top - float(np.min(z))
    band = z_top - PROCESSING["plateau_level"] * z_rng
    lo_s, hi_s = _summit_run(z >= band, int(np.argmax(z)), x, merge_gap,
                             z=z, shallow=z_top - PROCESSING["saddle_depth"] * z_rng)
    t = int(round(0.5 * (lo_s + hi_s)))
    if t < 4 or t > len(z) - 5:
        return {}, [
            "The highest point sits at the end of the transect, so this is not a "
            "complete edifice profile. Extend the transect past the summit on both sides."
        ]

    z_sum = z_top
    left, right = z[: t + 1], z[t:]
    q = PROCESSING["base_quantile"]
    base_l = float(np.quantile(left, q))
    base_r = float(np.quantile(right, q))
    z_base = max(base_l, base_r)

    height = z_sum - z_base
    if height <= 0:
        return {}, ["No relief between summit and base."]

    if abs(base_l - base_r) > 0.25 * height:
        notes.append(
            f"The two flanks reach base level {abs(base_l - base_r):,.0f} apart in elevation. "
            "The higher base is used, so the lower flank is measured only down to that level. "
            "This under-measures basal width on edifices sitting on sloping ground."
        )

    # The transect has to include real ground beyond the foot for the basal level
    # to be found. If the flank that sets the basal level keeps falling well below
    # it, the quantile landed on the flank itself: a convex shield whose transect
    # ended at its foot read height -31% and B/H 13.8 against a true 10.
    side_min = float(np.min(left)) if base_l >= base_r else float(np.min(right))
    if z_base - side_min > 0.10 * height:
        notes.append(
            f"On the flank that sets the basal level the profile keeps falling "
            f"{(z_base - side_min) / height:.0%} of the edifice height below that level. The "
            "transect probably ends on the flank rather than on the ground beyond it, so the "
            "base is set too high: height is under-measured and B/H over-measured. "
            "Extend the transect onto the surrounding ground."
        )

    # Basal width: the first crossing of the basal level going outward from the
    # summit on each side
    def crossing(seg_idx, direction):
        i = t
        while 0 <= i < len(z):
            if z[i] <= z_base:
                break
            i += direction
        if i < 0 or i >= len(z):
            return None
        j = i - direction
        if not (0 <= j < len(z)) or z[j] == z[i]:
            return float(x[i])
        f = (z_base - z[i]) / (z[j] - z[i])
        return float(x[i] + f * (x[j] - x[i]))

    xl = crossing(t, -1)
    xr = crossing(t, +1)
    if xl is None or xr is None or xr <= xl:
        return {}, ["Could not find the edifice base on both sides of the summit."]

    basal_width = xr - xl
    bh = basal_width / height

    # Dense resample of the edifice only
    xs = np.linspace(xl, xr, 2001)
    zs = np.interp(xs, x, z)
    x_sum = float(x[t])

    # --- summit plateau, measured first ----------------------------------
    # b must be fitted on the FLANK, so the plateau has to be located before it.
    lvl = z_sum - PROCESSING["plateau_level"] * height
    above = zs >= lvl
    k0 = int(np.argmin(np.abs(xs - x_sum)))
    if not above[k0]:
        k0 = int(np.argmax(zs))
    kl, kr = _summit_run(above, k0, xs, merge_gap, z=zs,
                         shallow=z_sum - PROCESSING["saddle_depth"] * height)
    if kr > kl:
        p_left, p_right = float(xs[kl]), float(xs[kr])
    else:
        p_left = p_right = x_sum
    plateau = (p_right - p_left) / basal_width
    if plateau > 0.85:
        notes.append(
            f"The flat top covers {plateau:.0%} of the base, so there is almost no sloping flank left "
            "to measure curvature on. Trust the flat-top width and B/H; b is not meaningful for "
            "a shape this close to a rectangle."
        )

    # A second high point that is NOT part of the summit is reported, not averaged in.
    pk, _ = find_peaks(zs, prominence=0.05 * height)
    far = [k for k in pk if (xs[k] < p_left or xs[k] > p_right)
           and zs[k] >= z_base + 0.85 * height]
    if far:
        k = max(far, key=lambda i: zs[i])
        notes.append(
            f"A second high point stands {abs(xs[k] - x_sum) / basal_width:.0%} of the basal width from "
            f"the summit and reaches {(zs[k] - z_base) / height:.0%} of its height. It is treated as a "
            "separate peak, not as part of the summit, so the summit and plateau describe the "
            "highest point only. If both peaks belong to one summit, the transect may cut a "
            "double edifice or a very wide caldera."
        )

    # --- form exponent b -------------------------------------------------
    # Fit drop against distance measured from the EDGE of the summit plateau,
    # not from the summit point. Measuring from the summit makes b rise with
    # plateau width: a straight-flanked cone carrying a caldera half its basal
    # width read b = 3.6 that way, which would call it convex when its flanks
    # are perfectly straight. Fitting the flank alone separates the two, so
    # plateau size and flank curvature become independent readings.
    # b = 1 is a straight cone; b > 1 convex (shield); b < 1 concave (composite).
    # Distance AND drop must share an origin. Measuring distance from the
    # plateau edge while measuring drop from the summit leaves an offset in the
    # fit, and a straight-flanked cone then reads b = 0.70 instead of 1.00.
    # Both are therefore referenced to the plateau edge.
    dx = np.where(xs < p_left, p_left - xs,
                  np.where(xs > p_right, xs - p_right, 0.0))
    dz = lvl - zs
    floor = PROCESSING["fit_exclusion"] * height
    m = (dx > 0) & (dz > floor) & np.isfinite(dx) & np.isfinite(dz)
    b = r2 = np.nan
    if int(m.sum()) >= 16:
        lx, lz = np.log(dx[m]), np.log(dz[m])
        slope, icept = np.polyfit(lx, lz, 1)
        pred = slope * lx + icept
        ss_res = float(np.sum((lz - pred) ** 2))
        ss_tot = float(np.sum((lz - np.mean(lz)) ** 2))
        b = float(slope)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        if np.isfinite(r2) and r2 < 0.80:
            notes.append(
                f"The power-law fit is weak (r2 = {r2:.2f}), so b is poorly constrained for this "
                "profile. That usually means the flanks are irregular or the two sides differ."
            )

    # --- mean flank slope, area-weighted ---------------------------------
    # A DEM-based mean slope averages over the whole edifice surface, where the
    # outer annuli dominate in proportion to area (r dr). Weighting the profile
    # samples by distance from the summit reproduces that; a plain mean does not.
    local = np.degrees(np.arctan(np.abs(np.gradient(zs, xs))))
    rad = np.abs(xs - x_sum)
    fin = np.isfinite(local)
    slope_deg = float(np.sum(local[fin] * rad[fin]) / np.sum(rad[fin])) if np.sum(rad[fin]) > 0 else np.nan
    max_slope = float(np.nanmax(local)) if fin.any() else np.nan

    # --- flank asymmetry ---------------------------------------------------
    depth = np.clip(zs - z_base, 0.0, None)
    lm, rm = xs <= x_sum, xs >= x_sum
    a_l = float(np.trapezoid(depth[lm], xs[lm])) if lm.sum() > 1 else 0.0
    a_r = float(np.trapezoid(depth[rm], xs[rm])) if rm.sum() > 1 else 0.0
    asym = abs(a_r - a_l) / (a_r + a_l) if (a_r + a_l) > 0 else np.nan

    metrics = {
        "height": height,
        "basal_width": basal_width,
        "bh": bh,
        "b": b,
        "fit_r2": r2,
        "slope_deg": slope_deg,
        "max_slope_deg": max_slope,
        "plateau": plateau,
        "asymmetry": asym,
        "summit_elev": z_sum,
        "base_elev": z_base,
        "x_left": xl,
        "x_right": xr,
        "x_summit": x_sum,
    }
    return metrics, notes


def analyse_profile(x: np.ndarray, z: np.ndarray) -> tuple[dict, list, tuple]:
    """The one route every profile takes, real or ideal: locate, crop, re-measure.

    Pass 1 resamples the whole track and finds the edifice. Pass 2 crops to the
    edifice plus a margin, resamples from the ORIGINAL samples and smooths at the
    scale of the edifice, so the answer no longer depends on how much surrounding
    ground the file happens to contain. A second refinement runs only if the
    width moved by more than 5% between passes.

    Returns (metrics, notes, (x_track, z_track)); the track arrays are the smoothed
    whole-track profile, for plotting. Metric positions are in the original x units.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    xs, zs, _ = prepare_profile(x, z)
    m, notes = measure_edifice(xs, zs)
    if not m:
        return m, notes, (xs, zs)

    for _ in range(2):
        w = m["basal_width"]
        pad = PROCESSING["window_margin"] * w
        a = max(float(x.min()), m["x_left"] - pad)
        b = min(float(x.max()), m["x_right"] + pad)
        if b - a <= 0:
            break
        xc, zc, _ = prepare_window(x, z, a, b)
        m2, n2 = measure_edifice(xc, zc, merge_gap=PROCESSING["summit_merge"] * w)
        if not m2:
            break
        settled = abs(m2["basal_width"] - w) < 0.05 * w
        m, notes = m2, n2
        if settled:
            break
    return m, notes, (xs, zs)


FLANK_GRID = np.linspace(0.0, 1.0, 401)


def flank_profiles(x: np.ndarray, z: np.ndarray, m: dict) -> list:
    """The two flanks of one measured transect, each scaled to its own base, height and radius."""
    w = m["basal_width"]
    a = max(float(np.min(x)), m["x_left"] - PROCESSING["window_margin"] * w)
    b = min(float(np.max(x)), m["x_right"] + PROCESSING["window_margin"] * w)
    xc, zc, _ = prepare_window(np.asarray(x, float), np.asarray(z, float), a, b)
    xs = m["x_summit"]
    out = []
    for side, radius, sign in (("R", m["x_right"] - xs, 1.0), ("L", xs - m["x_left"], -1.0)):
        zz = np.interp(xs + sign * FLANK_GRID * radius, xc, zc)
        out.append({"side": side, "radius": float(radius),
                    "g": np.clip((zz - m["base_elev"]) / m["height"], 0.0, None)})
    return out


# =====================================================================
# READING A FILE
# Any text file (comma, semicolon, tab or space separated, with or without a
# header) or an .xlsx sheet. The first two columns of numbers are distance and
# elevation unless the header names them. Everything is converted to metres.
# =====================================================================
DISTANCE_UNITS = {"metres": 1.0, "kilometres": 1000.0, "feet": 0.3048, "miles": 1609.344}
ELEVATION_UNITS = {"metres": 1.0, "feet": 0.3048, "kilometres": 1000.0}

NODATA_BELOW = -9000.0      # -9999 and -32768 are the usual "no data" fills in elevation files
MIN_ROWS = 32

_DIST_NAME = re.compile(r"dist|length|chainage|offset|^x$|^d$|^s$", re.I)
_ELEV_NAME = re.compile(r"elev|alt|height|^z$|^h$|^y$|dem", re.I)


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def _fields(line: str, delim) -> list:
    parts = line.split() if delim is None else line.split(delim)
    return [p.strip().strip('"').strip("'") for p in parts]


def _pick_delimiter(lines: list):
    """The separator that cuts the most lines into the same number (at least two) of fields."""
    sample = lines[:60]
    best, best_score = "none", 0.0
    for delim in ("\t", ";", ",", None):
        counts = [len(_fields(ln, delim)) for ln in sample]
        usable = [c for c in counts if c >= 2]
        if not usable:
            continue
        mode = max(set(usable), key=usable.count)
        score = usable.count(mode) / len(sample)
        if score > best_score + 1e-9:
            best, best_score = delim, score
    return best


def _to_float(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def read_transect(raw: bytes, filename: str = "") -> tuple:
    """Distance and elevation from an uploaded file, as arrays in the file's own units.

    Returns (x, z, info) where info = {"columns": which columns were used, "notes": [...]}.
    Raises ValueError, in plain words, when the file cannot be used.
    """
    if filename.lower().endswith((".xlsx", ".xlsm")):
        try:
            sheet = pd.read_excel(io.BytesIO(raw), header=None)
        except Exception as exc:
            raise ValueError(f"Could not open this Excel file ({exc}).") from None
        text = sheet.to_csv(index=False, header=False)
    else:
        text = _decode(raw)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(("#", "//", "%"))]
    if not lines:
        raise ValueError("This file is empty.")
    delim = _pick_delimiter(lines)
    if delim == "none":
        raise ValueError("Could not find two columns of numbers in this file.")

    rows, header = [], None
    for ln in lines:
        f = _fields(ln, delim)
        if delim != ",":                    # decimal commas: 1234,5 -> 1234.5
            f = [re.sub(r"^([+-]?\d+),(\d+)$", r"\1.\2", p) for p in f]
        vals = [_to_float(p) for p in f]
        if sum(v is not None for v in vals) >= 2:
            rows.append(vals)
        elif header is None and not rows:
            header = f
    if len(rows) < MIN_ROWS:
        raise ValueError(f"Found only {len(rows)} rows of numbers; at least {MIN_ROWS} are needed.")

    width = max(len(r) for r in rows)
    arr = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = [np.nan if v is None else v for v in r]
    numeric = [j for j in range(width) if np.isfinite(arr[:, j]).mean() >= 0.5]
    if len(numeric) < 2:
        raise ValueError("Need two columns of numbers: distance along the transect, and elevation.")

    notes = []
    jd, jz, named = numeric[0], numeric[1], False
    has_names = header is not None and len(header) == width
    if has_names:
        d_cols = [j for j in numeric if _DIST_NAME.search(header[j])]
        z_cols = [j for j in numeric if _ELEV_NAME.search(header[j])]
        if d_cols and z_cols and d_cols[0] != z_cols[0]:
            jd, jz, named = d_cols[0], z_cols[0], True
    if len(numeric) > 2 and not named:
        notes.append("This file has more than two columns of numbers. The first two were used as "
                     "distance and elevation.")
    label = f"{header[jd]} (distance) and {header[jz]} (elevation)" if has_names \
        else f"column {jd + 1} (distance) and column {jz + 1} (elevation)"

    x, z = arr[:, jd], arr[:, jz]
    bad = ~np.isfinite(x) | ~np.isfinite(z) | (z <= NODATA_BELOW) | (z >= 1e30)
    if bad.any():
        notes.append(f"Dropped {int(bad.sum())} rows with missing or no-data values (such as -9999).")
    x, z = x[~bad], z[~bad]
    if len(x) < MIN_ROWS:
        raise ValueError(f"Only {len(x)} usable rows are left; at least {MIN_ROWS} are needed.")
    if np.any(np.diff(x) < 0):
        order = np.argsort(x, kind="stable")
        x, z = x[order], z[order]
        notes.append("The distances were not in order, so the rows were sorted by distance.")
    keep = np.concatenate([[True], np.diff(x) > 0])
    if not keep.all():
        notes.append(f"Removed {int((~keep).sum())} rows that repeated a distance.")
        x, z = x[keep], z[keep]
    if len(x) < MIN_ROWS or x[-1] <= x[0]:
        raise ValueError("The distances do not change along the file.")
    if np.ptp(z) <= 0:
        raise ValueError("Every elevation is the same, so there is no relief to measure.")
    return x, z, {"columns": label, "notes": notes, "rows": int(len(x))}


# =====================================================================
# THE PERFECT SHAPES
#
# Every shape is a height fraction g(u) of its own height, where u is the distance
# from the summit as a share of the distance to the base (0 at the summit, 1 at
# the base). Each is measured by analyse_profile, the same route a real transect
# takes, so the map compares like with like.
# =====================================================================
def _g_concave(u):
    return np.clip(1.0 - np.sqrt(u), 0.0, None) ** 2
def _g_triangle(u):
    return np.clip(1.0 - u, 0.0, None)
def _g_trapezoid(u):
    return np.minimum(np.clip(1.0 - u, 0.0, None), 0.7) / 0.7
def _g_semicircle(u):
    return np.sqrt(np.clip(1.0 - u ** 2, 0.0, None))
def _g_rounded(u):
    return np.clip(1.0 - u ** 4, 0.0, None) ** 0.25
def _g_square(u):
    return np.clip((1.0 - u) / 0.05, 0.0, 1.0)


NAMED_SHAPES = [
    {"key": "concave", "name": "Concave parabola", "g": _g_concave, "colour": "#2a78d6",
     "note": "Sides sag inward: steep at the top, flaring out at the base, like a steep stratovolcano."},
    {"key": "triangle", "name": "Isosceles triangle", "g": _g_triangle, "colour": "#52514e",
     "note": "Straight sides. The dividing line: b = 1."},
    {"key": "trapezoid", "name": "Trapezoid", "g": _g_trapezoid, "colour": "#eb6834",
     "note": "The triangle with its top 30% sliced off: straight sides and a flat top."},
    {"key": "semicircle", "name": "Semicircle", "g": _g_semicircle, "colour": "#1baf7a",
     "note": "Half a circle. Squashed, it becomes a half-ellipse."},
    {"key": "rounded", "name": "Rounded square", "g": _g_rounded, "colour": "#eda100",
     "note": "Between a circle and a square: steep sides with a rounded shoulder."},
    {"key": "square", "name": "Near-square", "g": _g_square, "colour": "#4a3aa7",
     "note": "As close to a rectangle as can be measured (the walls lean over only 5% of the half-width)."},
]
NAMED_BY_KEY = {s["key"]: s for s in NAMED_SHAPES}
NAMED_REFERENCE_BH = 10.0       # b and flat-top do not depend on it; any value gives the same point

# The landmarks along the "bend the sides" path, for describing a best fit in words.
BEND_LANDMARKS = [(0.5, "the concave parabola"), (1.0, "the isosceles triangle"),
                  (2.0, "the semicircle"), (4.0, "the rounded square")]

U = FLANK_GRID


def ideal_flank(u, bend: float, cut: float):
    """A perfect shape: bend 1 is straight, below 1 sags inward, above 1 bulges; cut is the share of height sliced off."""
    e = max(float(bend), 0.05)
    h = np.clip(1.0 - np.asarray(u, float) ** e, 0.0, None) ** (1.0 / e)
    c = float(np.clip(cut, 0.0, 0.95))
    return np.minimum(h, 1.0 - c) / (1.0 - c)


def ideal_profile(bend: float, cut: float, bh: float = NAMED_REFERENCE_BH, half_width: float = 1000.0, n: int = 2001):
    """A full cross-section of a perfect shape, ready for analyse_profile."""
    x = np.linspace(-half_width * 1.25, half_width * 1.25, int(n))
    return x, (2.0 * half_width / bh) * ideal_flank(np.abs(x) / half_width, bend, cut)


def named_profile(g, bh: float = NAMED_REFERENCE_BH, half_width: float = 1000.0, n: int = 2001):
    x = np.linspace(-half_width * 1.25, half_width * 1.25, int(n))
    return x, (2.0 * half_width / float(bh)) * g(np.abs(x) / half_width)


@st.cache_data(show_spinner="Measuring the perfect shapes (first run only)...")
def ideal_map() -> dict:
    """Where the perfect shapes sit on the map, and the paths between them."""
    named = {}
    for s in NAMED_SHAPES:
        x, z = named_profile(s["g"])
        m, _n, _ = analyse_profile(x, z)
        named[s["key"]] = (float(m["b"]), float(m["plateau"]))

    def path(bend, cuts):
        pts = []
        for c in cuts:
            x, z = ideal_profile(bend, c)
            m, _n, _ = analyse_profile(x, z)
            if m and np.isfinite(m["b"]):
                pts.append((float(c), float(m["b"]), float(m["plateau"])))
        return pts

    return {
        "named": named,
        "bend": [(e,) + p[1:] for e in np.round(np.arange(0.5, 4.01, 0.1), 2) for p in path(e, [0.0])],
        "cuts": {"concave": path(0.5, np.round(np.arange(0.0, 0.501, 0.025), 3)),
                 "triangle": path(1.0, np.round(np.arange(0.0, 0.901, 0.05), 3)),
                 "semicircle": path(2.0, np.round(np.arange(0.0, 0.501, 0.025), 3))},
    }


def fit_best_ideal(flanks: list) -> dict:
    """The bend and cut whose perfect shape has the smallest total misfit to both flanks."""
    def resid(p):
        h = ideal_flank(U, p[0], p[1])
        return np.concatenate([h - f["g"] for f in flanks])

    best = None
    for e0 in (0.3, 0.6, 1.0, 2.0):
        r = least_squares(resid, [e0, 0.05], bounds=([0.15, 0.0], [6.0, 0.7]))
        if best is None or r.cost < best.cost:
            best = r
    return {"bend": float(best.x[0]), "cut": float(best.x[1])}


def closest_shape_text(bend: float, cut: float, plateau: float = 0.0) -> str:
    """The best fit in words: which named shapes the bend sits between, and what the top is like.

    A fitted cut on a pointed cone (measured flat-top under 10% of the base) is a blunted summit, not a
    truncated one, and is worded that way; with a real flat top it is a flat top. The smoothing every
    profile gets blunts even a perfect sharp tip (the concave parabola reads a cut of about 13%), so on a
    pointed cone a cut under 15% is not reported.
    """
    ratio = 1.15
    if bend < BEND_LANDMARKS[0][0] / ratio:
        words = "sagging even more than the concave parabola"
    elif bend > BEND_LANDMARKS[-1][0] * ratio:
        words = "squarer than the rounded square"
    else:
        words = None
        for v, name in BEND_LANDMARKS:
            if v / ratio <= bend <= v * ratio:
                words = f"close to {name}"
                break
        if words is None:
            for (v0, n0), (v1, n1) in zip(BEND_LANDMARKS[:-1], BEND_LANDMARKS[1:]):
                if v0 < bend < v1:
                    words = f"between {n0} and {n1}"
                    break
    if plateau >= 0.10 and cut >= 0.04:
        words += f", with a flat top (about {cut:.0%} of the height cut off the top)"
    elif plateau < 0.10 and cut >= 0.15:
        words += f", with a blunted summit (top cut back about {cut:.0%} of the height)"
    return words


# =====================================================================
# MEASURING A TRANSECT AND COMPARING IT WITH A PERFECT SHAPE
# =====================================================================
TRANSECT_COLOURS = ["#c92a2a", "#0b7285", "#862e9c", "#2b8a3e", "#a61e4d", "#364fc7", "#6d4c41", "#d9480f"]

REFERENCES = [("best", "The best-fitting perfect shape")] + [(s["key"], s["name"]) for s in NAMED_SHAPES]

# The summit must have a real flank on both sides. The engine only refuses a summit within a few samples of the
# end of the transect; a line that starts at the summit can still come out as a "volcano" of zero width with
# nonsense numbers. Across the 60 real transects the shorter side is never under 0.23 of the longer.
MIN_SIDE_RATIO = 0.10


def measure_transect(name: str, x_m, z_m) -> dict:
    """Measure one transect (distance and elevation in metres). Never raises: a failure is reported in the item."""
    item = {"name": name, "ok": False, "on_map": False, "notes": [], "problem": ""}
    x_m, z_m = np.asarray(x_m, float), np.asarray(z_m, float)
    m, notes, _track = analyse_profile(x_m, z_m)
    item["notes"] = list(notes)
    if not m:
        item["problem"] = notes[0] if notes else "This transect could not be measured."
        item["notes"] = []
        return item
    left, right = m["x_summit"] - m["x_left"], m["x_right"] - m["x_summit"]
    if max(left, right) <= 0 or min(left, right) < MIN_SIDE_RATIO * max(left, right):
        item["problem"] = ("The highest point is at, or very near, one end of the transect, so the line does not "
                           "cross the volcano. Use a line that passes through the summit and runs out onto flat "
                           "ground on both sides.")
        item["notes"] = []
        return item
    flanks = flank_profiles(x_m, z_m, m)
    item.update(ok=True, m=m, flanks=flanks, fit=fit_best_ideal(flanks), x=x_m, z=z_m)
    item["on_map"] = bool(np.isfinite(m["b"]) and np.isfinite(m["plateau"]))
    if not item["on_map"]:
        item["notes"].append("b could not be measured (too few points on the sides), so this transect cannot be "
                             "placed on the map. The difference plot still works.")
    if m["bh"] < 1.0 or m["bh"] > 80.0:
        item["notes"].append(
            f"The base is {m['bh']:.1f} times as wide as the volcano is tall, which is unusual. If that is "
            "not what you expect, check that distance and elevation are in the units you chose.")
    return item


def run_all(inputs: list, dist_unit: str, elev_unit: str) -> list:
    """Read and measure every input. A file that cannot be used becomes an item with a plain-language problem."""
    items, seen, n_ok = [], {}, 0
    for name, raw in inputs:
        base = pathlib.Path(name).stem
        seen[base] = seen.get(base, 0) + 1
        label = base if seen[base] == 1 else f"{base} ({seen[base]})"
        try:
            x, z, info = read_transect(raw, name)
        except ValueError as exc:
            items.append({"name": label, "ok": False, "problem": str(exc), "notes": []})
            continue
        item = measure_transect(label, x * DISTANCE_UNITS[dist_unit], z * ELEVATION_UNITS[elev_unit])
        item["read"] = info
        item["notes"] = info["notes"] + item["notes"]
        if item["ok"]:
            n_ok += 1
            item["no"] = n_ok
            item["colour"] = TRANSECT_COLOURS[(n_ok - 1) % len(TRANSECT_COLOURS)]
        items.append(item)
    return items


def difference_for(item: dict, ref_key: str) -> dict:
    """The perfect shape to compare with, and real minus perfect for each side, in shares of the height."""
    if ref_key == "best":
        bend, cut = item["fit"]["bend"], item["fit"]["cut"]
        ideal = ideal_flank(U, bend, cut)
        label = closest_shape_text(bend, cut, item["m"]["plateau"])
        short = "best-fitting perfect shape"
    else:
        s = NAMED_BY_KEY[ref_key]
        ideal = s["g"](U)
        label = s["name"]
        short = s["name"]
    res = {f["side"]: f["g"] - ideal for f in item["flanks"]}
    both = np.concatenate([res["L"], res["R"]])
    mean_res = 0.5 * (res["L"] + res["R"])
    zones = {"summit": float(mean_res[U < 0.15].mean()),
             "middle": float(mean_res[(U >= 0.15) & (U < 0.6)].mean()),
             "base": float(mean_res[U >= 0.6].mean())}
    return {"ideal": ideal, "res": res, "label": label, "short": short, "ref_key": ref_key,
            "rms": float(np.sqrt(np.mean(both ** 2))), "zones": zones}


def difference_words(zones: dict) -> str:
    """Where the real outline has more or less material than the perfect shape."""
    where = {"summit": "near the summit", "middle": "on the middle of the sides", "base": "low on the sides"}
    big = sorted(((k, v) for k, v in zones.items() if abs(v) >= 0.02), key=lambda kv: -abs(kv[1]))
    if not big:
        return "It stays within 2% of its height of the perfect shape everywhere."
    parts = [f"{'more' if v > 0 else 'less'} material {where[k]} ({v * 100:+.0f}% of its height)" for k, v in big]
    text = "Compared with the perfect shape it has " + ", ".join(parts) + "."
    if zones["summit"] <= -0.05:
        text += " Missing material at the summit is what a crater or caldera looks like."
    return text


def describe_transect(item: dict) -> str:
    m = item["m"]
    if np.isfinite(m["b"]):
        sides = ("sides sag inward" if m["b"] < 0.95 else "nearly straight sides" if m["b"] <= 1.05
                 else "sides bulge outward")
        shape = f"{sides} (b = {m['b']:.2f})"
    else:
        shape = "sides that could not be measured"
    top = (f"a flat top {m['plateau']:.0%} as wide as the base" if m["plateau"] >= 0.10
           else "a pointed or narrow top")
    fit = item["fit"]
    return (f"**{item['no']}. {item['name']}**: {shape}, {top}, B/H = {m['bh']:.1f}. "
            f"Best-fitting perfect shape: {closest_shape_text(fit['bend'], fit['cut'], m['plateau'])}. "
            f"{difference_words(difference_for(item, 'best')['zones'])}")


def results_table(items: list) -> pd.DataFrame:
    rows = []
    for it in items:
        m = it["m"]
        d = difference_for(it, "best")
        rows.append({
            "#": it["no"], "Transect": it["name"],
            "Height (m)": round(m["height"]), "Base width (km)": round(m["basal_width"] / 1000.0, 2),
            "B/H": round(m["bh"], 2),
            "b (how bowed the sides are)": round(m["b"], 2) if np.isfinite(m["b"]) else np.nan,
            "Flat-top (share of base)": round(m["plateau"], 3),
            "Best-fitting perfect shape": closest_shape_text(it["fit"]["bend"], it["fit"]["cut"], m["plateau"]),
            "Typical misfit (% of height)": round(100 * d["rms"], 1),
            "Left/right area difference (%)": round(100 * m["asymmetry"], 1),
            "Warnings": len(it["notes"])})
    return pd.DataFrame(rows)


# =====================================================================
# THE PICTURES
# Drawn with matplotlib on a white background and saved as PNG, so they can be
# pasted into a slide or document and look the same there.
# =====================================================================
INK = "#1f2328"
GREY = "#868e96"
EXTRA = "#f59f00"        # the volcano has more material than the perfect shape
MISSING = "#228be6"      # the volcano has less


@st.cache_resource(show_spinner=False)
def _font() -> str:
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Arial", "Helvetica", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


def _rc() -> dict:
    return {"font.family": _font(), "font.size": 13, "axes.titlesize": 17, "axes.titleweight": "bold",
            "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
            "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#495057",
            "axes.linewidth": 1.2, "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white"}


def _png(fig, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    return buf.getvalue()


def _rgba(hex_colour: str, alpha: float) -> tuple:
    h = hex_colour.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, alpha)


MAP_VIEWS = {"both": "Whole map and zoom, side by side", "whole": "Whole map only", "zoom": "Zoom only"}


def _zoom_limits(pts: list) -> tuple:
    """The corner where real volcanoes fall, widened if a transect lies outside it."""
    x0, x1, y0, y1 = 0.15, 1.35, -0.06, 0.40
    for it in pts:
        b, f = it["m"]["b"], it["m"]["plateau"]
        x0, x1 = min(x0, b - 0.08), max(x1, b + 0.1)
        y0, y1 = min(y0, f - 0.05), max(y1, f + 0.06)
    return x0, x1, y0, y1


def _whole_limits(pts: list) -> tuple:
    x0, x1, y0, y1 = 0.2, 2.55, -0.05, 1.14
    for it in pts:
        b, f = it["m"]["b"], it["m"]["plateau"]
        x0, x1 = min(x0, b - 0.08), max(x1, b + 0.08)
        y0, y1 = min(y0, f - 0.04), max(y1, f + 0.05)
    return x0, x1, y0, y1


def _draw_map(ax, mp: dict, pts: list, limits: tuple, zoom: bool, numbered: bool, size: float) -> None:
    """One panel of the shape map: the perfect shapes and the paths between them, then the transects."""
    x0, x1, y0, y1 = limits
    span = x1 - x0
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.grid(True, color="#e9ecef", lw=1)
    ax.set_axisbelow(True)
    ax.axvline(1.0, color="#ced4da", lw=1.6, ls=":", zorder=1)

    bend = np.array([(b, f) for _e, b, f in mp["bend"]])
    ax.plot(bend[:, 0], bend[:, 1], color=INK, lw=2.8, zorder=2, solid_capstyle="round")
    for key, pts_c in mp["cuts"].items():
        arr = np.array(pts_c)
        colour = NAMED_BY_KEY[key]["colour"]
        ax.plot(arr[:, 1], arr[:, 2], color=colour, lw=2.2, ls=(0, (5, 3)), zorder=2)
        if key in ("triangle", "semicircle"):
            # the triangle's 30% cut is the trapezoid, which carries its own label
            ticks = (0.1, 0.5, 0.7, 0.9) if key == "triangle" else (0.1, 0.3, 0.5)
            for c, b, f in pts_c:
                if round(c, 2) in ticks and x0 < b < x1 and y0 < f < y1:
                    ax.plot([b], [f], "o", ms=4.5, color=colour, zorder=3)
                    ax.annotate(f"{c:.0%}", (b, f), xytext=(8, 0), textcoords="offset points", ha="left",
                                va="center", fontsize=10.5, color=colour, zorder=3)

    for s in NAMED_SHAPES:
        bx, fy = mp["named"][s["key"]]
        if not (x0 < bx < x1 and y0 < fy < y1):
            continue
        ax.scatter([bx], [fy], s=190, color=s["colour"], edgecolor="white", linewidth=2.2, zorder=4)
        if not zoom:
            u = np.linspace(-1, 1, 81)
            iw, ih, gap = 0.11, 0.075, 0.032
            ax.fill(np.concatenate([bx + iw * u, [bx + iw, bx - iw]]),
                    np.concatenate([fy + gap + ih * s["g"](np.abs(u)), [fy + gap, fy + gap]]),
                    facecolor=_rgba(s["colour"], 0.25), edgecolor=s["colour"], lw=1.6, zorder=3)
        style = dict(fontsize=12.5, fontweight="bold", color=s["colour"], zorder=5,
                     bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
        if s["key"] == "concave" and zoom:
            ax.annotate(s["name"], (bx, fy), xytext=(bx, y0 + 0.012 * (y1 - y0)), ha="center", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=s["colour"], lw=1.2), **style)
        elif s["key"] == "concave":
            ax.text(bx - 0.08, fy + 0.17, s["name"], ha="left", va="center", **style)
        elif bx > x1 - 0.2 * span:
            ax.text(bx - 0.06 * span, fy, s["name"], ha="right", va="center", **style)
        else:
            ax.text(bx + 0.06 * span, fy, s["name"], ha="left", va="center", **style)

    for it in pts:
        b, f = it["m"]["b"], it["m"]["plateau"]
        ax.scatter([b], [f], s=size, marker="D", color=it["colour"], edgecolor=INK, linewidth=1.6, zorder=8)
        if numbered:
            ax.text(b, f, str(it["no"]), ha="center", va="center", color="white",
                    fontsize=11.5 if size > 400 else 10, fontweight="bold", zorder=9)
    ax.set_xlabel("b:  how bowed the sides are   (1 = straight sides)")
    ax.set_ylabel("Flat-top width as a share of the base")
    ax.text(0.012, 0.975, "← sag inward" if zoom else "← sides sag inward", transform=ax.transAxes,
            va="top", color=GREY, fontsize=12)
    ax.text(0.988, 0.975, "bulge outward →" if zoom else "sides bulge outward →",
            transform=ax.transAxes, va="top", ha="right", color=GREY, fontsize=12)


def shape_map_png(items: list, view: str = "both") -> bytes:
    """Figure 1: the perfect shapes on the map of b against flat-top width, with the transects on top.

    view: "both" (whole map with a zoom beside it), "whole" or "zoom".
    """
    mp = ideal_map()
    pts = [it for it in items if it["ok"] and it["on_map"]]
    whole, zoom = _whole_limits(pts), _zoom_limits(pts)
    with matplotlib.rc_context(_rc()):
        if view == "both":
            fig = Figure(figsize=(16, 7))
            axs = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.16})
            _draw_map(axs[0], mp, pts, whole, False, False, 150)
            _draw_map(axs[1], mp, pts, zoom, True, True, 430)
            axs[0].add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2], fill=False,
                                       ec=GREY, lw=1.6, ls=(0, (2, 2)), zorder=1))
            axs[0].set_title("The whole map", loc="left", fontsize=14, fontweight="normal", pad=8)
            axs[1].set_title("Zoom on the dashed box, where volcanoes usually fall", loc="left", fontsize=14,
                             fontweight="normal", pad=8)
            line_ax = axs[0]
        else:
            fig = Figure(figsize=(11, 7) if view == "whole" else (10, 7))
            ax = fig.add_subplot(111)
            if view == "whole":
                _draw_map(ax, mp, pts, whole, False, True, 330)
            else:
                _draw_map(ax, mp, pts, zoom, True, True, 430)
            line_ax = ax
        fig.suptitle("Where each transect falls among the perfect shapes", x=0.125, ha="left", y=0.99,
                     fontsize=17, fontweight="bold")
        wide = view == "both"
        if pts:
            handles = [Line2D([], [], marker="D", ls="", ms=11, mfc=it["colour"], mec=INK, mew=1.4,
                              label=f"{it['no']}   {it['name']}") for it in pts]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.34 if wide else 0.3, 0.0),
                       ncol=min(3, len(handles)), frameon=False, title="Your transects", alignment="left")
        fig.legend(handles=[
            Line2D([], [], color=INK, lw=2.8, label="Bend the sides"),
            Line2D([], [], color=GREY, lw=2.2, ls=(0, (5, 3)), label="Slice the top off (% of height)")],
            loc="upper center", bbox_to_anchor=(0.82 if wide else 0.8, 0.0), frameon=False,
            title="Paths between the shapes", alignment="left")
        return _png(fig)


def _step_fill(ax, xs, y_real, y_ideal):
    ax.fill_between(xs, y_ideal, y_real, where=y_real >= y_ideal, interpolate=True, color=EXTRA, alpha=0.4, lw=0)
    ax.fill_between(xs, y_ideal, y_real, where=y_real < y_ideal, interpolate=True, color=MISSING, alpha=0.4, lw=0)


def _both_sides(res_or_g: dict) -> np.ndarray:
    """Left flank (reversed) then right flank, so a curve runs from the left base through the summit to the right base."""
    return np.concatenate([res_or_g["L"][::-1], res_or_g["R"][1:]])


X_BOTH = np.concatenate([-U[::-1], U[1:]])
X_TICKS = ([-1, -0.5, 0, 0.5, 1], ["base", "halfway", "summit", "halfway", "base"])


def _ylim_pct(values_pct) -> float:
    return max(10.0, 1.2 * float(np.max(np.abs(values_pct))))


def difference_png(item: dict, ref_key: str = "best") -> bytes:
    """Figure 2: the real outline against the perfect shape, and the difference between them."""
    d = difference_for(item, ref_key)
    g = {f["side"]: f["g"] for f in item["flanks"]}
    real = _both_sides(g)
    ideal = np.concatenate([d["ideal"][::-1], d["ideal"][1:]])
    diff = 100.0 * (real - ideal)
    m = item["m"]
    with matplotlib.rc_context(_rc()):
        fig = Figure(figsize=(11, 7.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.1)
        top = fig.add_subplot(gs[0])
        bot = fig.add_subplot(gs[1], sharex=top)

        _step_fill(top, X_BOTH, real, ideal)
        top.plot(X_BOTH, real, color=item["colour"], lw=3.4, label=f"{item['name']} (measured)", zorder=4)
        top.plot(X_BOTH, ideal, color=INK, lw=2.4, ls=(0, (5, 3)), label=f"Perfect shape: {d['label']}"
                 if ref_key != "best" else "Best-fitting perfect shape", zorder=5)
        top.set_ylim(-0.03, 1.2)
        top.set_ylabel("Height as a share of the volcano's height")
        top.set_title(f"{item['name']}: measured outline against a perfect shape", loc="left", pad=46)
        stats = (f"b = {m['b']:.2f}     flat-top = {m['plateau']:.2f}     B/H = {m['bh']:.1f}"
                 if np.isfinite(m["b"]) else f"B/H = {m['bh']:.1f}")
        top.text(0.0, 1.105, stats, transform=top.transAxes, fontsize=12.5, color="#495057", va="bottom")
        top.text(0.0, 1.035, ("Perfect shape: " + d["label"]) if ref_key == "best"
                 else f"Compared with the {d['label'].lower()}", transform=top.transAxes, fontsize=12.5,
                 color="#495057", va="bottom")
        top.legend(loc="upper left", frameon=False)
        top.tick_params(labelbottom=False)

        _step_fill(bot, X_BOTH, diff, np.zeros_like(diff))
        bot.axhline(0, color=INK, lw=1.6, zorder=3)
        bot.plot(X_BOTH, diff, color=item["colour"], lw=3, zorder=4)
        lim = _ylim_pct(diff)
        bot.set_ylim(-lim, lim)
        bot.set_ylabel("Measured minus perfect\n(% of the height)")
        bot.set_xticks(X_TICKS[0])
        bot.set_xticklabels(X_TICKS[1])
        bot.set_xlim(-1.02, 1.02)
        bot.set_xlabel("Distance from the summit as a share of the distance to the base   "
                       "(left side of the transect | right side)")
        bot.text(0.01, 0.95, "more material than the perfect shape", transform=bot.transAxes, va="top",
                 color="#e67700", fontsize=12, fontweight="bold")
        bot.text(0.01, 0.05, "less material", transform=bot.transAxes, va="bottom",
                 color="#1971c2", fontsize=12, fontweight="bold")
        for a in (top, bot):
            a.axvline(0, color="#ced4da", lw=1.2, ls=":", zorder=1)
            a.grid(True, axis="y", color="#e9ecef", lw=1)
        return _png(fig)


def differences_together_png(items: list, ref_key: str = "best") -> bytes:
    """Figure 2 for several transects at once: only the differences, one line each."""
    with matplotlib.rc_context(_rc()):
        fig = Figure(figsize=(11, 5.6))
        ax = fig.add_subplot(111)
        ax.axhline(0, color=INK, lw=1.6, zorder=3)
        worst = 10.0
        for it in items:
            d = difference_for(it, ref_key)
            diff = 100.0 * _both_sides(d["res"])
            worst = max(worst, _ylim_pct(diff))
            ax.plot(X_BOTH, diff, color=it["colour"], lw=2.8, zorder=4,
                    label=f"{it['no']}   {it['name']}")
        ax.set_ylim(-worst, worst)
        ax.set_xlim(-1.02, 1.02)
        ax.set_xticks(X_TICKS[0])
        ax.set_xticklabels(X_TICKS[1])
        ax.axvline(0, color="#ced4da", lw=1.2, ls=":", zorder=1)
        ax.grid(True, axis="y", color="#e9ecef", lw=1)
        ax.set_ylabel("Measured minus perfect shape\n(% of the height)")
        ax.set_xlabel("Distance from the summit as a share of the distance to the base   "
                      "(left side of the transect | right side)")
        what = "its own best-fitting perfect shape" if ref_key == "best" else NAMED_BY_KEY[ref_key]["name"]
        ax.set_title(f"Difference from {what}", loc="left", pad=12)
        ax.text(0.01, 0.97, "more material than the perfect shape", transform=ax.transAxes, va="top",
                color="#e67700", fontsize=12, fontweight="bold")
        ax.text(0.01, 0.03, "less material", transform=ax.transAxes, va="bottom", color="#1971c2",
                fontsize=12, fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(3, len(items)), frameon=False,
                  title="Your transects", alignment="left")
        return _png(fig)


def check_png(items: list) -> bytes:
    """The uploaded profiles as they are, with the base level, summit and edges the program found."""
    n = len(items)
    cols = 2 if n > 1 else 1
    rows = -(-n // cols)
    with matplotlib.rc_context(_rc()):
        fig = Figure(figsize=(11, 3.7 * rows))
        axes = fig.subplots(rows, cols, squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for it, ax in zip(items, axes.ravel()):
            ax.set_visible(True)
            m = it["m"]
            km = np.ptp(it["x"]) > 3000
            sc = 1000.0 if km else 1.0
            ax.plot(it["x"] / sc, it["z"], color=it["colour"], lw=2.2, zorder=3)
            ax.axhline(m["base_elev"], color=INK, lw=1.4, ls=(0, (5, 3)), zorder=2)
            ax.axvspan(m["x_left"] / sc, m["x_right"] / sc, color="#f1f3f5", zorder=0)
            ax.plot([m["x_left"] / sc, m["x_right"] / sc], [m["base_elev"]] * 2, "o", color=INK, ms=7, zorder=4)
            ax.plot([m["x_summit"] / sc], [m["summit_elev"]], "^", color=INK, ms=10, zorder=4)
            ax.set_title(f"{it['no']}  {it['name']}", loc="left", fontsize=14, pad=8)
            ax.set_title(f"height {m['height']:,.0f} m,  base width {m['basal_width'] / 1000.0:,.1f} km",
                         loc="right", fontsize=11, fontweight="normal", color="#495057", pad=8)
            ax.set_xlabel("Distance (km)" if km else "Distance (m)", fontsize=12)
            ax.set_ylabel("Elevation (m)", fontsize=12)
        fig.tight_layout(h_pad=1.8, w_pad=2.0)
        return _png(fig, dpi=150)


def copy_button_html(png: bytes, label: str = "Copy image") -> str:
    """A button that puts the picture on the clipboard (works in Chrome, Edge, Safari, recent Firefox)."""
    b64 = base64.b64encode(png).decode("ascii")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;font-family:system-ui,sans-serif;font-size:14px;color:#31333f}}
button{{font:inherit;padding:.45rem .9rem;border:1px solid #c9ccd3;border-radius:.5rem;background:#f4f5f7;
color:#31333f;cursor:pointer}}
button:hover{{border-color:#c92a2a;color:#c92a2a}}
#s{{margin-left:.6rem}}
img{{position:absolute;left:-10000px;top:0;width:800px}}
</style></head><body><button id="b" type="button">{label}</button><span id="s"></span>
<img id="pic" alt="" src="data:image/png;base64,{b64}">
<script>
const s = document.getElementById("s");
const pic = document.getElementById("pic");
const done = "Copied. Paste it into your slide.";

// Way 1: the clipboard API. Way 2, if the browser refuses: select the picture and use the old copy command.
function copySelection() {{
  const range = document.createRange();
  range.selectNode(pic);
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
  const ok = document.execCommand("copy");
  sel.removeAllRanges();
  return ok;
}}

document.getElementById("b").addEventListener("click", async () => {{
  let why = "";
  try {{
    const bin = atob(pic.src.split(",")[1]), arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    await navigator.clipboard.write([new ClipboardItem({{"image/png": new Blob([arr], {{type: "image/png"}})}})]);
    s.textContent = done;
    return;
  }} catch (err) {{
    why = err.name;
  }}
  try {{
    if (copySelection()) {{ s.textContent = done; return; }}
  }} catch (err) {{
    why += " / " + err.name;
  }}
  s.textContent = "Copying is blocked in this browser (" + why + "). Right-click the picture and choose Copy image.";
}});
</script></body></html>"""


EXAMPLE_DIR = pathlib.Path(__file__).resolve().parent / "samples" / "jax_examples"


# =====================================================================
# INTERFACE
# =====================================================================
INTRO = """
Upload one or more **transects**: a line of points across a volcano, with the **distance** along the line and the
**elevation** at each point. Spectrum Jax measures the shape of each one, places it on a map of perfect shapes
(triangle, circle, square ...), and shows how far the real outline is from the best-fitting perfect shape.
Both pictures can be copied straight into a presentation.
"""

FILE_HELP = """
**What the file should look like.** Two columns of numbers, one row per point: distance along the line, then
elevation. CSV, TXT or Excel; a header row is optional. The line should cross the whole volcano **through the
summit** and run out onto flat ground on **both sides**; if it stops on the flank, the base cannot be found.
"""

MAP_TEXT = """
**What you are looking at.** Every outline gets two numbers. **b** says how bowed the sides are: 1 is straight sides
(a triangle), below 1 the sides sag inward (steep near the top and flaring out at the base, like a steep
stratovolcano), above 1 they bulge outward (like a dome). **Flat-top** is the width of the flat summit as a share of
the width of the base. The coloured dots are the perfect shapes, measured in exactly the same way as your transects.
The black line shows what happens when a shape's sides are bent, and the dashed lines show what happens when its
top is sliced off. Your transects are the numbered diamonds. The zoom shows the corner where real volcanoes usually
fall; the closer a diamond sits to a perfect shape, the more the volcano looks like it.

Squashing a shape (a circle into an ellipse) does not move it on this map. That only changes **B/H**, the base
width divided by the height, which is in the table below.
"""

DIFF_TEXT = """
**What you are looking at.** The real outline and the perfect shape are scaled to the same height and the same base
width, so only the *shape* is compared, not the size. The top panel shows both. The bottom panel is the difference,
measured minus perfect, as a percentage of the volcano's height. Where the line is **above zero** (orange) the
volcano has more material than the perfect shape; **below zero** (blue) it has less. A line lying flat on zero would
be a perfect match. By default the perfect shape is the one that fits best (found by trying every bend and top-cut
on the map and keeping the smallest total misfit); you can also compare against any one of the named shapes.
"""

HOW_IT_WORKS = """
1. **Find the volcano.** The program finds the summit and then the foot of the volcano on each side. The base is one
   flat level that both sides reach (the higher of the two feet), so left and right are measured down to the same
   elevation. The height is the summit minus that base, and the base width is the distance between the two places
   where the ground crosses that level. The profile is cropped to the volcano before it is measured, so a long
   transect and a short one give the same answer.
2. **Measure two shape numbers.** *b* comes from how quickly the ground drops away from the edge of the flat top: a
   straight cone gives 1. *Flat-top* is the width of the ground within 5% of the height of the summit, as a share of
   the base width.
3. **Place it on the map.** The perfect shapes are put through exactly the same measurement, so a dot's position is
   comparable with theirs.
4. **Subtract the best perfect shape.** Each side of the volcano is scaled to its own height and base distance, the
   best-fitting perfect shape is found, and it is subtracted from the measured outline.
"""

LIMITS = """
* One transect is one slice through a volcano. Lines in different directions can give different dots, and the spread
  between them tells you how much the answer depends on the direction.
* The elevation data set sets what can be seen: with 90 m SRTM data a small crater or a narrow summit is blurred.
* The base is an estimate, and every number that uses it (height, base width, B/H) moves a little if the base moves.
* b and the flat-top width do not depend on the units. Height, base width and B/H do, so the units you pick matter
  for those.
* A deep crater or caldera is not a flat top, even though it can read like one. Check the check picture.
* On a pointed cone the best fit often "cuts" the top. That means the summit is blunter than a perfect point, not
  that the top is flat or sliced off.
* The best-fitting perfect shape bends toward a bump or a dip, so a local feature reads smaller (and a little
  shifted) than it is. In a test, a bulge of 6% of the height read as 3%. Choose one of the named shapes under
  "Compare with" to see the differences at full size.
"""

TALK = """
1. **What I did.** "I took a line of elevations across the volcano and gave the program the distance and the height
   at each point. It finds the summit and the base."
2. **The first picture.** "From the outline it measures two numbers: b, which says whether the sides are straight,
   sagging in or bulging out, and the flat-top width. I measured the perfect shapes the same way, so I can put my
   volcano on the same map and say which shape it is closest to."
3. **The second picture.** "Here I scaled the volcano to the same height and base width as the best-fitting perfect
   shape and subtracted one from the other. Above zero the volcano has more material than the perfect shape, below
   zero it has less."
4. **What it means, and what it does not.** "[Say where your volcano falls and where the biggest differences are.]
   This is one line across one volcano at the resolution of the data, so it shows the shape along that line, not
   why the volcano looks that way."
"""


def example_files() -> list:
    return sorted(EXAMPLE_DIR.glob("*.csv")) if EXAMPLE_DIR.exists() else []


def show_html(html: str, height: int) -> None:
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


def cached_png(state: dict, key: tuple, build) -> bytes:
    if key not in state["png"]:
        state["png"][key] = build()
    return state["png"][key]


def figure_block(png: bytes, filename: str, key: str) -> None:
    st.image(png, width="stretch")
    c1, c2, _ = st.columns([1.5, 1.1, 2.4])
    with c1:
        show_html(copy_button_html(png), height=100)
    with c2:
        st.download_button("Download PNG", png, file_name=filename, mime="image/png", key=f"dl_{key}")


st.title("Spectrum Jax")
st.markdown(INTRO)

with st.container(border=True):
    st.markdown("##### 1. Choose your transects")
    files = st.file_uploader("Drop one or more files here", type=["csv", "txt", "tsv", "dat", "xlsx"],
                             accept_multiple_files=True, key="jax_files")
    u1, u2 = st.columns(2)
    dist_unit = u1.selectbox("Distance is in", list(DISTANCE_UNITS), key="jax_dist_unit")
    elev_unit = u2.selectbox("Elevation is in", list(ELEVATION_UNITS), key="jax_elev_unit")
    examples = example_files()
    use_examples = (st.checkbox(f"Also include the {len(examples)} example transects", key="jax_use_examples")
                    if examples else False)
    st.markdown(FILE_HELP)
    st.markdown("##### 2. Press the button")
    go = st.button("Make my figures", type="primary")

sig = (tuple((f.name, f.size) for f in files), bool(use_examples), dist_unit, elev_unit)

if go:
    inputs = [(f.name, f.getvalue()) for f in files]
    if use_examples:
        inputs += [(p.name, p.read_bytes()) for p in examples]
    if not inputs:
        st.warning("Choose a file first (or tick the example transects), then press the button.")
    else:
        with st.spinner("Measuring..."):
            st.session_state["jax"] = {"items": run_all(inputs, dist_unit, elev_unit), "sig": sig, "png": {}}

state = st.session_state.get("jax")
if state:
    if state["sig"] != sig:
        st.info("The files or units have changed since these figures were made. Press the button to update them.")
    items = state["items"]
    ok = [it for it in items if it["ok"]]
    for it in items:
        if not it["ok"]:
            st.error(f"**{it['name']}** could not be used: {it['problem']}")

    if ok:
        # ---------------------------------------------------------------- figure 1
        st.divider()
        st.subheader("The ideal shapes plot")
        view = st.radio("Show", list(MAP_VIEWS), format_func=MAP_VIEWS.get, horizontal=True, key="jax_view")
        figure_block(cached_png(state, ("map", view), lambda: shape_map_png(ok, view)),
                     "spectrum_jax_ideal_shapes.png", f"map_{view}")
        st.markdown(MAP_TEXT)
        for it in ok:
            if not it["on_map"]:
                st.warning(f"**{it['name']}** is not on the map: b could not be measured for it.")

        # ---------------------------------------------------------------- figure 2
        st.divider()
        st.subheader("The difference from the ideal")
        d1, d2 = st.columns(2)
        names = [f"{it['no']}. {it['name']}" for it in ok]
        which = d1.selectbox("Show", (["All together"] if len(ok) > 1 else []) + names, key="jax_which")
        ref_label = d2.selectbox("Compare with", [label for _k, label in REFERENCES], key="jax_ref")
        ref_key = next(k for k, label in REFERENCES if label == ref_label)
        if which == "All together":
            png = cached_png(state, ("together", ref_key), lambda: differences_together_png(ok, ref_key))
        else:
            item = ok[names.index(which)]
            png = cached_png(state, ("diff", item["no"], ref_key), lambda: difference_png(item, ref_key))
        figure_block(png, "spectrum_jax_difference_from_ideal.png", f"diff_{which}_{ref_key}")
        st.markdown(DIFF_TEXT)

        # ---------------------------------------------------------------- numbers and words
        st.divider()
        st.subheader("The numbers")
        table = results_table(ok)
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download the table (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name="spectrum_jax_results.csv", mime="text/csv", key="dl_table")
        st.markdown("**In words**")
        for it in ok:
            st.markdown("* " + describe_transect(it))
        for it in ok:
            if it["notes"]:
                st.warning(f"**{it['name']}**\n\n" + "\n".join(f"* {n}" for n in it["notes"]))

        with st.expander("Check that the base and summit were found correctly"):
            st.markdown("The triangle is the summit, the two dots are where the ground meets the base level, the dashed "
                        "line is the base level, and the grey band is the part used as the volcano. If these are wrong, "
                        "the numbers are wrong.")
            st.image(cached_png(state, ("check",), lambda: check_png(ok)), width="stretch")
            for it in ok:
                st.caption(f"{it['no']}. {it['name']}: read {it['read']['columns']}, {it['read']['rows']} rows.")

    st.divider()
    with st.expander("What to say in class (about a minute)"):
        st.markdown(TALK)
    with st.expander("What is happening, step by step"):
        st.markdown(HOW_IT_WORKS)
    with st.expander("What this can and cannot tell you"):
        st.markdown(LIMITS)
else:
    with st.expander("What is happening, step by step"):
        st.markdown(HOW_IT_WORKS)
