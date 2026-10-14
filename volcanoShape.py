"""
Volcano Shape - radial transect morphometry for shield vs composite classification.

Load a CSV of distance and elevation along a radial transect through a volcano's
summit. The app measures the edifice, classifies it, and plots it on a reference
diagram of ideal geometric shapes.

Built for transect-only settings: bodies where a gridded DEM does not exist but
altimeter tracks do (Venus Magellan, Titan SARTopo, Mercury MLA), where the
standard DEM-based tools cannot run at all.

Every processing parameter is fixed in PROCESSING below, so every file is
treated identically and results are comparable between students, sessions and
bodies. Nothing is adjustable from the interface by design.

Run with:  py -m streamlit run volcanoShape.py
"""

import io
import json
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.interpolate import LinearNDInterpolator
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import cKDTree

st.set_page_config(page_title="Volcano Shape", page_icon="⛰", layout="wide")


# =====================================================================
# FIXED PROCESSING PARAMETERS
# Identical treatment of every file. Change here, never in the interface.
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

# Published thresholds. See the About tab for full citations.
# Grosse & Kervyn (2018) Geomorphology 304, 1-14, from 158 terrestrial shields:
# an approximate shield/stratovolcano boundary at 12 degrees mean slope and
# H/W_B of 0.10, i.e. B/H of 10.
THRESHOLD_SLOPE_DEG = 12.0
THRESHOLD_BH = 10.0

# A perfect straight-sided cone reads b = 1. Convex flanks (shield-like) read
# above 1; concave flanks (composite-like) read below 1, once the profile is
# inverted for measurement.
CONE_B = 1.0

# A perfect cone is not flat-topped, but the plateau is measured at a finite
# height below the summit, so a cone returns this value rather than zero.
CONE_PLATEAU = PROCESSING["plateau_level"]


# =====================================================================
# LOADING AND PREPARATION
# =====================================================================
def read_profile(uploaded) -> tuple[np.ndarray, np.ndarray, str]:
    """Read a CSV and take the first two numeric columns as distance, elevation."""
    raw = uploaded.read()
    uploaded.seek(0)
    text = raw.decode("utf-8", errors="ignore")

    df = None
    for sep in (None, ",", "\t", ";", r"\s+"):
        try:
            cand = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            if cand.shape[1] >= 2:
                df = cand
                break
        except Exception:
            continue
    if df is None:
        raise ValueError("Could not parse this file. It needs at least two columns.")

    numeric = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= max(16, 0.5 * len(s)):
            numeric.append(col)
    if len(numeric) < 2:
        raise ValueError(
            "Need two numeric columns: distance along the transect, and elevation."
        )

    x = pd.to_numeric(df[numeric[0]], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df[numeric[1]], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(x) & np.isfinite(z)
    x, z = x[good], z[good]
    if len(x) < 32:
        raise ValueError("Need at least 32 usable rows.")

    order = np.argsort(x)
    x, z = x[order], z[order]
    keep = np.concatenate([[True], np.diff(x) > 0])
    return x[keep], z[keep], f"{numeric[0]} vs {numeric[1]}"


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


# =====================================================================
# MEASUREMENT
# =====================================================================
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


def classify(m: dict) -> dict:
    """Shield or composite, from three independent published indicators."""
    votes, detail = [], []

    bh = m.get("bh", np.nan)
    if np.isfinite(bh):
        v = "Shield" if bh >= THRESHOLD_BH else "Composite"
        votes.append(v)
        detail.append((
            "B/H (basal width / height)", f"{bh:.1f}",
            f"shield if >= {THRESHOLD_BH:.0f}", v,
            "Grosse & Kervyn (2018): shields H/W_B mostly 0.01-0.10, boundary at 0.10",
        ))

    sl = m.get("slope_deg", np.nan)
    if np.isfinite(sl):
        v = "Shield" if sl <= THRESHOLD_SLOPE_DEG else "Composite"
        votes.append(v)
        detail.append((
            "Mean flank slope", f"{sl:.1f} deg",
            f"shield if <= {THRESHOLD_SLOPE_DEG:.0f} deg", v,
            "Grosse & Kervyn (2018): shields mostly 1-15 deg, boundary at 12 deg",
        ))

    b = m.get("b", np.nan)
    if np.isfinite(b):
        v = "Shield" if b >= CONE_B else "Composite"
        votes.append(v)
        detail.append((
            "Flank curvature b", f"{b:.3f}",
            f"shield if >= {CONE_B:.2f} (convex)", v,
            "Svensson (1959) form exponent; convex flanks are shield-like, concave composite-like",
        ))

    if not votes:
        return {"verdict": "Not determined", "agreement": 0, "n": 0, "detail": detail,
                "caveats": []}

    caveats = []
    # The 12 degree boundary was calibrated on DEM-derived slopes. A single
    # profile samples only the along-track gradient and misses the cross-track
    # component, so it under-reads a DEM mean slope by roughly 15-25 percent
    # once any surface texture is present. Profile slopes therefore sit low
    # against this threshold, biasing the slope indicator toward "Shield".
    if np.isfinite(sl) and THRESHOLD_SLOPE_DEG * 0.75 <= sl <= THRESHOLD_SLOPE_DEG:
        caveats.append(
            f"Mean slope ({sl:.1f} deg) is just below the {THRESHOLD_SLOPE_DEG:.0f} deg boundary, "
            "inside the band where profile-derived slope is known to under-read a DEM-derived "
            "one. Treat this vote as weak evidence for Shield."
        )
    if np.isfinite(bh) and 0.8 * THRESHOLD_BH <= bh <= 1.25 * THRESHOLD_BH:
        caveats.append(
            f"B/H ({bh:.1f}) sits within 25 percent of the {THRESHOLD_BH:.0f} boundary, so that "
            "vote could go either way on a different azimuth."
        )
    a_ = m.get("asymmetry", np.nan)
    if np.isfinite(a_) and a_ > 0.30:
        caveats.append(
            f"Flank asymmetry is high ({a_:.2f}). One flank dominates, so this transect may not "
            "represent the edifice. Run more azimuths."
        )

    n_shield = votes.count("Shield")
    n = len(votes)
    verdict = "Shield" if n_shield * 2 > n else ("Composite" if n_shield * 2 < n else "Ambiguous")
    agreement = max(n_shield, n - n_shield)
    return {"verdict": verdict, "agreement": agreement, "n": n, "detail": detail,
            "caveats": caveats}


# =====================================================================
# THE IDEAL SHAPE LATTICE
#
# Perfect geometric shapes are generated from two controls applied to a
# parent of known proportions:
#
#   curvature e   z = H (1 - u^e)^(1/e), with u = distance / half-width.
#                 e < 1 is concave (composite-like), e = 1 a straight cone,
#                 e > 1 convex (shield-like).
#   cut depth     the top is sliced off horizontally, removing this fraction of
#                 the height. The flanks and the base are unchanged, so the
#                 summit becomes a flat top: a crater floor or a caldera rim.
#
# Every shape goes through analyse_profile, the SAME route real data takes, so
# the lattice and the samples cannot drift apart.
#
# Two measured facts organise the whole diagram:
#
#   * b and the summit plateau do not depend on the parent's proportions,
#     because stretching a shape vertically changes neither. One reference
#     proportion therefore places every node in the b / plateau plane.
#   * Cutting the top lowers the height but not the width, so B/H rises by
#     roughly 1 / (1 - cut), almost independent of curvature. B/H is therefore
#     the third axis: it separates a broad parent from a cut-down one.
#
# Cutting a straight cone leaves b at 1 however deep the cut, which is why b
# alone cannot see a truncated summit. On curved shapes the plateau already
# grows with curvature (the "backbone" of uncut shapes), and a cut lifts it off
# that backbone, so the plateau reads truncation as its distance from the
# backbone. Three measurements (b, plateau, B/H) recover three controls
# (curvature, cut, parent B/H).
# =====================================================================
LATTICE = {
    # Lines drawn on the diagram.
    "curvature": [0.5, 0.7, 1.0, 1.4, 2.0, 3.0],
    "cut": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    # B/H of the parent each drawn grid sheet is built on (before any cut). In the
    # B/H panels every extra value adds another copy of the grid, shifted up the axis.
    "parent_bh": [12.0],
    # Grid measured to build the lines and to read a sample back as an ideal
    # shape: (start, stop, step). It must contain the values listed above, and its
    # edges are the limits of what counts as "on the lattice". Concave shapes below
    # e = 0.5 and cuts beyond half the height give unstable fits and are not
    # edifices anyone measures; anything out there is reported as off the lattice.
    "grid_curvature": (0.5, 4.0, 0.1),
    "grid_cut": (0.0, 0.50, 0.025),
    # b and plateau are independent of proportion, so one reference B/H is enough.
    "reference_bh": 10.0,
}


def _grid(spec) -> np.ndarray:
    lo, hi, step = spec
    return np.round(np.arange(lo, hi + step / 2.0, step), 4)


def lattice_profile(curvature: float, cut: float, parent_bh: float,
                    half_width: float = 1000.0, n: int = 2001):
    """One ideal edifice: curvature and cut depth, on a parent of the given B/H."""
    H = 2.0 * half_width / max(float(parent_bh), 1e-6)
    x = np.linspace(-half_width * 1.25, half_width * 1.25, int(n))
    u = np.abs(x) / half_width
    e = max(float(curvature), 0.05)
    z = H * np.clip(1.0 - u ** e, 0.0, None) ** (1.0 / e)
    z = np.minimum(z, H * (1.0 - float(np.clip(cut, 0.0, 0.95))))
    return x, z


def lattice_key() -> str:
    """Changes whenever a processing or lattice setting does, so cached shapes never go stale."""
    return json.dumps({"processing": PROCESSING, "lattice": LATTICE}, sort_keys=True, default=str)


@st.cache_data(show_spinner="Measuring the ideal shapes (first run only)...")
def build_lattice(key: str) -> pd.DataFrame:
    """Measure every lattice node with the production code. `key` only drives the cache."""
    ref = float(LATTICE["reference_bh"])
    rows = []
    for e in _grid(LATTICE["grid_curvature"]):
        for d in _grid(LATTICE["grid_cut"]):
            x, z = lattice_profile(e, d, ref)
            m, _notes, _ = analyse_profile(x, z)
            if not m or not np.isfinite(m.get("b", np.nan)):
                continue
            rows.append({"Curvature e": float(e), "Cut": float(d), "b": m["b"],
                         "Plateau": m["plateau"], "B/H": m["bh"], "Fit r2": m["fit_r2"]})
    lat = pd.DataFrame(rows)
    uncut = lat[np.isclose(lat["Cut"], 0.0)].set_index("Curvature e")["B/H"]
    lat["B/H ratio"] = lat["B/H"] / lat["Curvature e"].map(uncut)
    return lat.dropna().reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_inverse(key: str) -> dict:
    """Index the lattice so a (b, plateau) measurement can be read back as (e, cut, B/H ratio)."""
    lat = build_lattice(key)
    pts = np.column_stack([lat["b"], np.log(np.clip(lat["Plateau"], 1e-3, None))])
    scale = pts.std(axis=0)
    tree = cKDTree(pts / scale)
    nn, _ = tree.query(pts / scale, k=2)
    return {
        "interp": LinearNDInterpolator(pts / scale, lat[["Curvature e", "Cut", "B/H ratio"]].to_numpy()),
        "tree": tree,
        "values": lat[["Curvature e", "Cut", "B/H ratio"]].to_numpy(),
        "scale": scale,
        # A measurement further than this from every node is not on the lattice.
        "reach": float(2.0 * np.max(nn[:, 1])),
    }


def estimate_ideal(inv: dict, b: float, plateau: float, bh: float) -> dict | None:
    """The ideal shape that best matches a measurement: curvature, cut depth, parent B/H.

    `on_lattice` is False when the measurement lies outside everything the ideal
    shapes can produce (a b or plateau no cut superellipse reaches). The returned
    numbers are then only the nearest node and should not be quoted as a fit.
    """
    if not (np.isfinite(b) and np.isfinite(plateau) and np.isfinite(bh)):
        return None
    q = np.array([b, np.log(max(float(plateau), 1e-3))]) / inv["scale"]
    dist, i = inv["tree"].query(q)
    v = np.asarray(inv["interp"](q[None, :]))[0]
    on_lattice = bool(np.all(np.isfinite(v))) and float(dist) <= inv["reach"]
    if not np.all(np.isfinite(v)):
        v = inv["values"][i]
    e, cut, ratio = (float(t) for t in v)
    return {"curvature": e, "cut": cut, "parent_bh": float(bh / ratio) if ratio > 0 else np.nan,
            "on_lattice": on_lattice, "distance": float(dist)}


def curvature_class(e: float) -> str:
    return "concave" if e < 0.95 else ("straight" if e <= 1.05 else "convex")


def describe_ideal(ideal: dict) -> str:
    flank = {"concave": "concave flanks (composite-like)",
             "straight": "straight flanks",
             "convex": "convex flanks (shield-like)"}[curvature_class(ideal["curvature"])]
    if ideal["cut"] < 0.04:
        top = "an uncut summit"
    else:
        top = f"a summit cut down {ideal['cut']:.0%} of its height"
    return f"{flank}, {top}, on a parent of B/H about {ideal['parent_bh']:.0f}"


# =====================================================================
# PLOTS
# =====================================================================
SAMPLE_SYMBOLS = ["circle", "square", "triangle-up", "star", "cross",
                  "hexagon", "triangle-down", "diamond-tall", "pentagon", "x-thin"]
SAMPLE_SYMBOLS_3D = ["circle", "square", "diamond", "cross", "x"]

CURVE_COLOURS = {"concave": "#2a78d6", "straight": "#52514e", "convex": "#eb6834"}
CUT_COLOUR = "#8a8981"
LEGEND_CURVE = "Constant curvature e  (blue concave, grey straight, orange convex)"
LEGEND_CUT = "Constant summit cut  (dashed)"

AXIS_CHOICES = {
    "B/H  (proportions)": ("B/H", "bh",
                           "B/H   basal width / height   (higher = broader and flatter)", True),
    "Summit plateau  (flat top / crater)": ("Plateau", "plateau",
                           "Plateau width / basal width   (a cone reads 0.05)", False),
}


def lattice_lines(lat: pd.DataFrame, parents) -> list:
    """The drawn lattice: one line per constant curvature, one per constant cut, per parent B/H."""
    lines = []
    for parent in parents:
        for e in LATTICE["curvature"]:
            s = lat[np.isclose(lat["Curvature e"], e)].sort_values("Cut")
            if len(s) > 1:
                lines.append(dict(kind="curvature", value=float(e), parent=float(parent),
                                  b=s["b"].to_numpy(), plateau=s["Plateau"].to_numpy(),
                                  bh=parent * s["B/H ratio"].to_numpy(),
                                  e=s["Curvature e"].to_numpy(), cut=s["Cut"].to_numpy()))
        for d in LATTICE["cut"]:
            s = lat[np.isclose(lat["Cut"], d)].sort_values("Curvature e")
            if len(s) > 1:
                lines.append(dict(kind="cut", value=float(d), parent=float(parent),
                                  b=s["b"].to_numpy(), plateau=s["Plateau"].to_numpy(),
                                  bh=parent * s["B/H ratio"].to_numpy(),
                                  e=s["Curvature e"].to_numpy(), cut=s["Cut"].to_numpy()))
    return lines


def _add_lattice(fig, lines, xk, yk, uses_bh, legend_seen, row=None, col=None, dims=2):
    """Draw the lattice lines onto a 2-D panel (dims=2) or a 3-D scene (dims=3, with zk in xk,yk,zk)."""
    parents = sorted({ln["parent"] for ln in lines})
    use = parents if uses_bh else parents[:1]
    lab = {"x": [], "y": [], "t": [], "pos": [], "c": []}
    Scatter = go.Scatter if dims == 2 else go.Scatter3d
    for ln in lines:
        if ln["parent"] not in use:
            continue
        kind = ln["kind"]
        colour = CURVE_COLOURS[curvature_class(ln["value"])] if kind == "curvature" else CUT_COLOUR
        name = LEGEND_CURVE if kind == "curvature" else LEGEND_CUT
        hover = [f"e {e_:.2f} | cut {c_:.0%} | parent B/H {ln['parent']:.0f}"
                 for e_, c_ in zip(ln["e"], ln["cut"])]
        kw = dict(mode="lines", name=name, legendgroup=kind, showlegend=kind not in legend_seen,
                  text=hover, hovertemplate="%{text}<extra></extra>")
        if dims == 2:
            kw["line"] = dict(width=1.7 if kind == "curvature" else 1.2, color=colour,
                              dash="solid" if kind == "curvature" else "dash")
            kw["opacity"] = 0.9 if (not uses_bh or ln["parent"] == use[0]) else 0.5
            kw["x"], kw["y"] = ln[xk], ln[yk]
        else:
            kw["line"] = dict(width=4 if kind == "curvature" else 2, color=colour,
                              dash="solid" if kind == "curvature" else "dash")
            kw["x"], kw["y"], kw["z"] = ln[xk[0]], ln[xk[1]], ln[xk[2]]
        legend_seen.add(kind)
        fig.add_trace(Scatter(**kw), **({"row": row, "col": col} if row else {}))
        if dims == 2 and ln["parent"] == use[0]:
            lab["x"].append(float(ln[xk][-1])); lab["y"].append(float(ln[yk][-1]))
            lab["t"].append(f"e={ln['value']:g}" if kind == "curvature"
                            else ("uncut" if ln["value"] == 0 else f"cut {ln['value']:.0%}"))
            lab["pos"].append("top center" if kind == "curvature" else "middle right")
            lab["c"].append(colour)
    if dims == 2 and lab["x"]:
        fig.add_trace(go.Scatter(x=lab["x"], y=lab["y"], mode="text", text=lab["t"],
                                 textposition=lab["pos"], textfont=dict(size=10, color=lab["c"]),
                                 showlegend=False, hoverinfo="skip"),
                      **({"row": row, "col": col} if row else {}))


def sample_hover(s: dict, i: int) -> str:
    out = (f"<b>{i}. {s.get('label', 'sample')}</b><br>b {s['b']:.3f} | plateau {s['plateau']:.3f}"
           f" | B/H {s['bh']:.1f}")
    e = s.get("ideal_e", np.nan)
    if np.isfinite(e):
        out += (f"<br>ideal: e {e:.2f}, cut {s['ideal_cut']:.0%}, parent B/H {s['ideal_bh']:.0f}"
                + ("" if s.get("on_lattice", True) else "  (off the lattice)"))
    return out


def sample_buckets(samples: list, keys: tuple, symbols: list) -> dict:
    """Pinned samples grouped for plotting; each group keeps one marker shape."""
    groups = []
    for s in samples:
        g = str(s.get("group", "") or "")
        if g and g not in groups:
            groups.append(g)
    buckets = {}
    for i, s in enumerate(samples, start=1):
        if not all(np.isfinite(s.get(k, np.nan)) for k in keys):
            continue
        g = str(s.get("group", "") or "ungrouped")
        d = buckets.setdefault(g, {"vals": [[] for _ in keys], "text": [], "hover": [], "symbol": "circle",
                                   "colour": s.get("colour", "#0b0b0b")})
        if g != "ungrouped" and g in groups:
            d["symbol"] = symbols[groups.index(g) % len(symbols)]
        for j, k in enumerate(keys):
            d["vals"][j].append(s[k])
        d["text"].append("" if s.get("no_number") else str(i))
        d["hover"].append(sample_hover(s, i))
    return buckets


def _add_samples(fig, samples, current, keys, legend_seen, row=None, col=None, dims=2):
    """Pinned samples as numbered black markers, and the loaded profile as a red diamond."""
    pos = {"row": row, "col": col} if row else {}
    Scatter = go.Scatter if dims == 2 else go.Scatter3d
    for g, d in sample_buckets(samples, keys, SAMPLE_SYMBOLS if dims == 2 else SAMPLE_SYMBOLS_3D).items():
        kw = dict(mode="markers+text", text=d["text"], textposition="top center",
                  textfont=dict(size=11, color="#0b0b0b"), customdata=d["hover"],
                  hovertemplate="%{customdata}<extra></extra>",
                  name=g if g != "ungrouped" else "Samples", legendgroup="sample:" + g,
                  showlegend=("sample:" + g) not in legend_seen)
        legend_seen.add("sample:" + g)
        if dims == 2:
            kw["x"], kw["y"] = d["vals"][0], d["vals"][1]
            kw["marker"] = dict(size=13, symbol=d["symbol"], color=d["colour"],
                                line=dict(width=2, color="#fcfcfb"))
        else:
            kw["x"], kw["y"], kw["z"] = d["vals"]
            kw["marker"] = dict(size=6, symbol=d["symbol"], color=d["colour"],
                                line=dict(width=1, color="#fcfcfb"))
        fig.add_trace(Scatter(**kw), **pos)
    if current and all(np.isfinite(current.get(k, np.nan)) for k in keys):
        kw = dict(mode="markers", name="Loaded now", legendgroup="now",
                  showlegend="now" not in legend_seen,
                  hovertemplate=f"<b>{current.get('label', 'loaded now')}</b><extra></extra>")
        legend_seen.add("now")
        if dims == 2:
            kw["x"], kw["y"] = [current[keys[0]]], [current[keys[1]]]
            kw["marker"] = dict(size=18, symbol="diamond-open", color="#e34948",
                                line=dict(width=3, color="#e34948"))
        else:
            kw["x"], kw["y"], kw["z"] = [current[keys[0]]], [current[keys[1]]], [current[keys[2]]]
            kw["marker"] = dict(size=9, symbol="diamond", color="#e34948",
                                line=dict(width=2, color="#fcfcfb"))
        fig.add_trace(Scatter(**kw), **pos)


def master_plot(lat: pd.DataFrame, samples: list, current: dict | None = None,
                yaxis: str = "B/H  (proportions)"):
    """One panel: b on x, B/H or plateau on y, the lattice behind, every pinned sample on top."""
    ycol, ykey, ylabel, ylog = AXIS_CHOICES.get(yaxis, AXIS_CHOICES["B/H  (proportions)"])
    fig = go.Figure()
    legend_seen: set = set()
    lines = lattice_lines(lat, LATTICE["parent_bh"])

    if ycol == "B/H":
        # The shading must not set the axis range (a rectangle out to 1e4 squashed the
        # data into a thin band), so the range is taken from what is actually plotted.
        vals = [float(v) for ln in lines for v in ln["bh"]]
        vals += [float(s["bh"]) for s in samples if np.isfinite(s.get("bh", np.nan))]
        if current and np.isfinite(current.get("bh", np.nan)):
            vals.append(float(current["bh"]))
        lo = min(vals + [THRESHOLD_BH]) / 1.4
        hi = max(vals + [THRESHOLD_BH]) * 1.4
        fig.add_hrect(y0=THRESHOLD_BH, y1=hi, fillcolor="rgba(42,120,214,0.06)",
                      line_width=0, layer="below")
        fig.add_hrect(y0=lo, y1=THRESHOLD_BH, fillcolor="rgba(208,59,59,0.06)",
                      line_width=0, layer="below")
        fig.add_hline(y=THRESHOLD_BH, line=dict(color="#8a8981", width=1.5, dash="dash"))
        # Annotations on a log axis take log10 of the value, unlike shapes and traces.
        fig.add_annotation(x=0.015, y=np.log10(THRESHOLD_BH), xref="paper", yanchor="bottom",
                           text=f"B/H = {THRESHOLD_BH:.0f}  shield above, composite below"
                                "  (Grosse & Kervyn 2018, approximate)",
                           showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.8)",
                           font=dict(size=11, color="#52514e"))
    else:
        fig.add_hline(y=CONE_PLATEAU, line=dict(color="#8a8981", width=1.5, dash="dash"))
        fig.add_annotation(x=0.015, y=CONE_PLATEAU, xref="paper", yanchor="bottom",
                           text="a cone with no flat top reads this value", showarrow=False,
                           xanchor="left", bgcolor="rgba(255,255,255,0.8)",
                           font=dict(size=11, color="#52514e"))
    fig.add_vline(x=CONE_B, line=dict(color="#8a8981", width=1.5, dash="dash"))
    fig.add_annotation(x=CONE_B, y=1.0, yref="paper", yshift=14,
                       text="b = 1  straight cone", showarrow=False,
                       font=dict(size=11, color="#52514e"))

    _add_lattice(fig, lines, "b", ykey, uses_bh=(ykey == "bh"), legend_seen=legend_seen)
    _add_samples(fig, samples, current, ("b", ykey), legend_seen)

    fig.update_layout(
        title="Shape lattice - flank curvature against " + ("proportions" if ycol == "B/H" else "summit plateau"),
        xaxis_title="b   flank curvature   (< 1 concave / composite,  1 straight cone,  > 1 convex / shield)",
        yaxis_title=ylabel,
        template="plotly_white", height=680, hovermode="closest",
        margin=dict(l=20, r=30, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
    )
    fig.update_yaxes(type=("log" if ylog else "linear"), gridcolor="#eceae4", zeroline=False)
    if ycol == "B/H":
        fig.update_yaxes(range=[np.log10(lo), np.log10(hi)])
    fig.update_xaxes(gridcolor="#eceae4", zeroline=False)
    return fig


# ---------------------------------------------------------------------
# Three-dimensional and pairwise views.
#
# The three measurements are independent controls of the ideal shapes, so a 3-D
# view is legitimate rather than decorative. Depth is still hard to read on a
# flat screen, so the pairwise panels show exactly the same data with no depth
# to misjudge. Both are offered; they cannot disagree.
# ---------------------------------------------------------------------
def master_plot_3d(lat: pd.DataFrame, samples: list, current: dict | None = None):
    fig = go.Figure()
    legend_seen: set = set()
    lines = lattice_lines(lat, LATTICE["parent_bh"])
    _add_lattice(fig, lines, ("b", "bh", "plateau"), None, uses_bh=True,
                 legend_seen=legend_seen, dims=3)
    _add_samples(fig, samples, current, ("b", "bh", "plateau"), legend_seen, dims=3)
    fig.update_layout(
        title="Shape space - drag to rotate, scroll to zoom",
        template="plotly_white", height=760,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.06, xanchor="left", x=0),
        scene=dict(
            xaxis=dict(title="b  flank curvature", gridcolor="#eceae4",
                       backgroundcolor="#fcfcfb", showbackground=True),
            yaxis=dict(title="B/H  proportions", type="log", gridcolor="#eceae4",
                       backgroundcolor="#fcfcfb", showbackground=True),
            zaxis=dict(title="summit plateau", gridcolor="#eceae4",
                       backgroundcolor="#fcfcfb", showbackground=True),
            camera=dict(eye=dict(x=1.7, y=1.5, z=0.9)),
        ),
    )
    return fig


# x key, y key, x label, y label, x log, y log
PAIRS = [("b", "bh", "b  flank curvature", "B/H  proportions", False, True),
         ("b", "plateau", "b  flank curvature", "summit plateau", False, False),
         ("bh", "plateau", "B/H  proportions", "summit plateau", True, False)]


def master_plot_pairs(lat: pd.DataFrame, samples: list, current: dict | None = None):
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                        subplot_titles=["b vs B/H", "b vs plateau", "B/H vs plateau"])
    legend_seen: set = set()
    lines = lattice_lines(lat, LATTICE["parent_bh"])
    for col, (xk, yk, xlab, ylab, xlog, ylog) in enumerate(PAIRS, start=1):
        _add_lattice(fig, lines, xk, yk, uses_bh=("bh" in (xk, yk)),
                     legend_seen=legend_seen, row=1, col=col)
        _add_samples(fig, samples, current, (xk, yk), legend_seen, row=1, col=col)
        fig.update_xaxes(title_text=xlab, gridcolor="#eceae4", type=("log" if xlog else "linear"),
                         row=1, col=col)
        fig.update_yaxes(title_text=ylab, gridcolor="#eceae4", type=("log" if ylog else "linear"),
                         row=1, col=col)
        if yk == "bh":
            fig.add_hline(y=THRESHOLD_BH, line=dict(color="#8a8981", width=1.5, dash="dot"),
                          row=1, col=col)
        if xk == "bh":
            fig.add_vline(x=THRESHOLD_BH, line=dict(color="#8a8981", width=1.5, dash="dot"),
                          row=1, col=col)
        if xk == "b":
            fig.add_vline(x=CONE_B, line=dict(color="#8a8981", width=1.5, dash="dot"),
                          row=1, col=col)
    fig.update_layout(
        title="The same three measurements, taken two at a time - no depth to misjudge",
        template="plotly_white", height=520,
        margin=dict(l=20, r=20, t=90, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.38, xanchor="left", x=0),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 12
        ann.font.color = "#52514e"
    return fig


# =====================================================================
# THE NAMED PERFECT SHAPES
#
# The shapes everyone knows, drawn on one base and one height so that they nest
# inside the same 2 x 1 rectangle (B/H = 2): a right-angled isosceles triangle,
# the upper half of a circle, and the rectangle itself. Every shape is measured
# by analyse_profile, the same route a real profile takes.
#
# Two things then happen to a shape, and they move it in different ways:
#
#   squashing   lowering the height on the same base (circle -> ellipse,
#               triangle -> flatter triangle). Changes B/H only. b and the flat-top
#               width do not move, so every squashed version of a shape sits on
#               the SAME point of the shape map.
#   cutting     slicing the top off (triangle -> trapezoid). Raises the flat top,
#               leaves b alone on a straight-sided shape, and raises B/H.
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


# g maps u = distance / half-width to a height fraction whose maximum is 1.
NAMED_SHAPES = [
    {"key": "concave", "name": "Concave parabola", "g": _g_concave, "colour": "#2a78d6",
     "note": "Sides sag inward: steep at the top, flaring out at the base, like a steep stratovolcano."},
    {"key": "triangle", "name": "Isosceles triangle", "g": _g_triangle, "colour": "#52514e",
     "note": "Straight sides. At B/H = 2 the apex is a right angle. The dividing line: b = 1 exactly."},
    {"key": "trapezoid", "name": "Trapezoid", "g": _g_trapezoid, "colour": "#eb6834",
     "note": "The triangle with its top 30% cut off: straight sides and a flat top."},
    {"key": "semicircle", "name": "Semicircle", "g": _g_semicircle, "colour": "#1baf7a",
     "note": "The upper half of a perfect circle. Squashed, it becomes a semi-ellipse."},
    {"key": "rounded", "name": "Rounded square", "g": _g_rounded, "colour": "#eda100",
     "note": "Between a circle and a square: steep sides with a rounded shoulder."},
    {"key": "square", "name": "Near-square", "g": _g_square, "colour": "#4a3aa7",
     "note": "A trapezoid whose walls lean over only 5% of the half-width: as close to a rectangle as can "
             "be measured. A true vertical wall has no flank to fit, so b is not meaningful there."},
]
NAMED_BH = [2.0, 5.0, 10.0, 25.0]
NAMED_REFERENCE_BH = 10.0


def named_profile(g, bh: float, half_width: float = 1000.0, n: int = 2001):
    x = np.linspace(-half_width * 1.25, half_width * 1.25, int(n))
    return x, (2.0 * half_width / float(bh)) * g(np.abs(x) / half_width)


@st.cache_data(show_spinner="Measuring the perfect shapes...")
def build_named(key: str):
    """Every named shape measured at several squash levels, plus the triangle cut deeper and deeper."""
    rows = []
    for s in NAMED_SHAPES:
        for bh in NAMED_BH:
            x, z = named_profile(s["g"], bh)
            m, _notes, _ = analyse_profile(x, z)
            if m:
                rows.append({"key": s["key"], "Shape": s["name"], "B/H set": bh, "B/H": m["bh"],
                             "b": m["b"], "Flat-top": m["plateau"], "Slope (deg)": m["slope_deg"],
                             "Fit r2": m["fit_r2"]})
    series = []
    for c in np.round(np.arange(0.0, 0.951, 0.05), 2):
        def g(u, c=c):
            return np.minimum(np.clip(1.0 - u, 0.0, None), 1.0 - c) / (1.0 - c)
        x, z = named_profile(g, NAMED_REFERENCE_BH)
        m, _notes, _ = analyse_profile(x, z)
        if m:
            series.append({"Cut": float(c), "b": m["b"], "Flat-top": m["plateau"]})
    return pd.DataFrame(rows), pd.DataFrame(series)


def named_position(named: pd.DataFrame, key: str) -> tuple:
    """(b, flat-top) of a named shape. Identical at every squash level, so the reference one is used."""
    r = named[(named["key"] == key) & (named["B/H set"] == NAMED_REFERENCE_BH)].iloc[0]
    return float(r["b"]), float(r["Flat-top"])


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def named_overlay_figure(named: pd.DataFrame, height: int = 470):
    """The perfect shapes drawn on one base and one height, nested in the same rectangle."""
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-1, x1=1, y0=0, y1=1,
                  line=dict(color="#a9a7a0", width=1.5, dash="dot"), layer="below")
    fig.add_shape(type="line", x0=-1.15, x1=1.15, y0=0, y1=0, line=dict(color="#0b0b0b", width=1.5))
    fig.add_annotation(x=0, y=1.0, yshift=14, showarrow=False, font=dict(size=11, color="#8a8981"),
                       text="the rectangle every shape fits inside  (width 2, height 1)")
    xs = np.linspace(-1.0, 1.0, 801)
    for s in NAMED_SHAPES:
        r = named[(named["key"] == s["key"]) & (named["B/H set"] == 2.0)].iloc[0]
        fig.add_trace(go.Scatter(
            x=xs, y=s["g"](np.abs(xs)), mode="lines", line=dict(color=s["colour"], width=3.5),
            name=f"{s['name']}  (b {r['b']:.2f}, flat-top {r['Flat-top']:.2f})",
            hovertemplate=f"<b>{s['name']}</b><br>{s['note']}<extra></extra>"))
    fig.update_layout(
        title="The perfect shapes, drawn on one base and one height (B/H = 2)",
        template="plotly_white", height=height, margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0))
    fig.update_xaxes(range=[-1.15, 1.15], visible=False)
    fig.update_yaxes(range=[-0.06, 1.2], scaleanchor="x", scaleratio=1, visible=False)
    return fig


def named_squash_figure():
    """Circle to ellipse, and triangle to flatter triangle: same base, lower and lower height."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05,
                        subplot_titles=["Circle to ellipse", "Triangle to flatter triangle"])
    xs = np.linspace(-1.0, 1.0, 801)
    for col, key in enumerate(("semicircle", "triangle"), start=1):
        s = next(sh for sh in NAMED_SHAPES if sh["key"] == key)
        for bh in NAMED_BH:
            height = 2.0 / bh
            fig.add_trace(go.Scatter(
                x=xs, y=height * s["g"](np.abs(xs)), mode="lines", showlegend=False,
                line=dict(color=s["colour"], width=2.5),
                hovertemplate=f"B/H {bh:g}<extra></extra>"), row=1, col=col)
            fig.add_annotation(x=0, y=height, yshift=11, showarrow=False, row=1, col=col,
                               text=f"B/H {bh:g}", font=dict(size=11, color="#52514e"))
        fig.add_shape(type="line", x0=-1.1, x1=1.1, y0=0, y1=0, line=dict(color="#0b0b0b", width=1.5),
                      row=1, col=col)
        fig.update_xaxes(range=[-1.12, 1.12], visible=False, row=1, col=col)
        fig.update_yaxes(range=[-0.05, 1.18], visible=False, row=1, col=col,
                         scaleanchor="x" if col == 1 else "x2", scaleratio=1)
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=20, r=20, t=50, b=10))
    for ann in fig.layout.annotations[:2]:
        ann.font.size = 13
    return fig


def shape_map_figure(named: pd.DataFrame, series: pd.DataFrame, lat: pd.DataFrame,
                     samples: list, current: dict | None = None, height: int = 700, zoom: bool = False,
                     spokes: list | None = None, trend: dict | None = None):
    """Where each perfect shape sits: b across, flat-top width up, with the paths between them."""
    fig = go.Figure()
    legend_seen: set = set()

    fig.add_vline(x=CONE_B, line=dict(color="#c8c6bf", width=1.5, dash="dot"))
    fig.add_annotation(x=0.215, y=1.10, showarrow=False, xanchor="left", text="< sides sag inward",
                       font=dict(size=11, color="#8a8981"))
    fig.add_annotation(x=2.5, y=1.10, showarrow=False, xanchor="right", text="sides bulge outward >",
                       font=dict(size=11, color="#8a8981"))

    # Bending the sides, with no cut: concave -> triangle -> circle -> rounded square.
    bend = lat[np.isclose(lat["Cut"], 0.0)].sort_values("Curvature e")
    fig.add_trace(go.Scatter(
        x=bend["b"], y=bend["Plateau"], mode="lines", name="Bend the sides (no cut)",
        line=dict(color="#0b0b0b", width=2.5),
        text=[f"bend setting {e:.1f}  (1 = triangle, 2 = circle, 4 = rounded square)" for e in bend["Curvature e"]],
        hovertemplate="%{text}<br>b %{x:.2f}, flat-top %{y:.2f}<extra></extra>"))

    # Cutting the top off: the triangle all the way, the others to half their height. The
    # rounded square is left out: with the flat top already near 1 its fit is unstable.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Cut the top off (dashed)",
                             line=dict(color="#8a8981", width=2, dash="dash")))
    cut_ticks = {0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9}
    for key, e in (("concave", 0.5), ("triangle", 1.0), ("semicircle", 2.0)):
        s = next(sh for sh in NAMED_SHAPES if sh["key"] == key)
        if key == "triangle":
            d = series.rename(columns={"Flat-top": "Plateau"})
        else:
            d = lat[np.isclose(lat["Curvature e"], e) & (lat["Cut"] <= 0.5 + 1e-9)].sort_values("Cut")
        fig.add_trace(go.Scatter(
            x=d["b"], y=d["Plateau"], mode="lines", showlegend=False,
            line=dict(color=s["colour"], width=2, dash="dash"),
            text=[f"{s['name']} with its top cut {c:.0%}" for c in d["Cut"]],
            hovertemplate="%{text}<br>b %{x:.2f}, flat-top %{y:.2f}<extra></extra>"))
        tk = d[d["Cut"].round(2).isin(cut_ticks)]
        if key in ("triangle", "semicircle"):
            fig.add_trace(go.Scatter(x=tk["b"], y=tk["Plateau"], mode="markers+text", showlegend=False,
                                     text=[f"{c:.0%}" for c in tk["Cut"]], textposition="middle right",
                                     textfont=dict(size=10, color=s["colour"]),
                                     marker=dict(size=5, color=s["colour"]), hoverinfo="skip"))

    # The perfect shapes themselves: a marker, a small drawing and a label each.
    for s in NAMED_SHAPES:
        bx, py = named_position(named, s["key"])
        fig.add_trace(go.Scatter(
            x=[bx], y=[py], mode="markers", showlegend=False,
            marker=dict(size=15, color=s["colour"], line=dict(width=2, color="#fcfcfb")),
            hovertemplate=f"<b>{s['name']}</b><br>b {bx:.2f}, flat-top {py:.2f}<br>{s['note']}<extra></extra>"))
        if not zoom:
            u = np.linspace(-1, 1, 81)
            icon_w, icon_h, gap = 0.11, 0.075, 0.035
            fig.add_trace(go.Scatter(
                x=np.concatenate([bx + icon_w * u, [bx + icon_w, bx - icon_w]]),
                y=np.concatenate([py + gap + icon_h * s["g"](np.abs(u)), [py + gap, py + gap]]),
                mode="lines", fill="toself", fillcolor=_rgba(s["colour"], 0.25), showlegend=False,
                line=dict(color=s["colour"], width=1.5), hoverinfo="skip"))
        left = bx > 1.9     # keep the label inside the plot at the right-hand edge
        fig.add_annotation(x=bx - 0.14 if left else bx + 0.14, y=py, xanchor="right" if left else "left",
                           showarrow=False, bgcolor="rgba(255,255,255,0.75)",
                           text=f"<b>{s['name']}</b>", font=dict(size=11, color=s["colour"]))

    # The fitted curve through the real transects, with the 95% band on the curve itself.
    if trend:
        fig.add_trace(go.Scatter(
            x=np.concatenate([trend["grid"], trend["grid"][::-1]]),
            y=np.concatenate([trend["hi"], trend["lo"][::-1]]), fill="toself", fillcolor="rgba(122,31,162,0.15)",
            line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=trend["grid"], y=trend["curve"], mode="lines", name="Volcano shape evolution curve (fitted)",
            line=dict(color="#7a1fa2", width=3.5),
            hovertemplate="fitted curve<br>b %{x:.2f}, flat-top %{y:.3f}<extra></extra>"))

    # When each volcano is collapsed to one point, its individual azimuths hang off it as spokes.
    for sp in spokes or []:
        for px, py, az in sp["points"]:
            fig.add_trace(go.Scatter(x=[sp["cx"], px], y=[sp["cy"], py], mode="lines", showlegend=False,
                                     line=dict(width=1, color=_rgba(sp["colour"], 0.45)), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[p[0] for p in sp["points"]], y=[p[1] for p in sp["points"]], mode="markers", showlegend=False,
            marker=dict(size=7, color=_rgba(sp["colour"], 0.55)), customdata=[p[2] for p in sp["points"]],
            hovertemplate=f"{sp['name']} %{{customdata}} deg<br>b %{{x:.2f}}, flat-top %{{y:.3f}}<extra></extra>"))

    _add_samples(fig, samples, current, ("b", "plateau"), legend_seen)

    fig.update_layout(
        title="Shape map - where each perfect shape sits",
        xaxis_title="b   how bowed the sides are   (1 = straight sides)",
        yaxis_title="flat-top width as a share of the base",
        template="plotly_white", height=height, hovermode="closest",
        margin=dict(l=20, r=30, t=70, b=110),
        legend=dict(orientation="h", yanchor="top", y=-0.13, xanchor="left", x=0))
    fig.update_xaxes(range=[0.15, 1.1] if zoom else [0.2, 2.55], gridcolor="#eceae4", zeroline=False)
    fig.update_yaxes(range=[-0.02, 0.32] if zoom else [-0.05, 1.14], gridcolor="#eceae4", zeroline=False)
    return fig


# =====================================================================
# REAL TRANSECTS
#
# samples/real holds radial elevation transects through seven volcanoes, four
# azimuths each, fetched from SRTM 90 m by samples/real/fetch_srtm_transects.py.
# They go through the same analyse_profile as any uploaded file.
# =====================================================================
REAL_COLOURS = ["#d03b3b", "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#eb6834", "#0b7a8a",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#9467bd", "#393b79", "#ad494a", "#637939"]


def real_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "samples" / "real"


@st.cache_data(show_spinner="Measuring the real transects...")
def load_real_transects(key: str) -> list:
    """Every transect listed in samples/real/manifest.csv, measured and read back against the lattice."""
    man = real_dir() / "manifest.csv"
    if not man.exists():
        return []
    inv = build_inverse(key)
    out, colours = [], {}
    for _, r in pd.read_csv(man).iterrows():
        f = real_dir() / str(r["file"])
        if not f.exists():
            continue
        df = pd.read_csv(f)
        m, notes, _ = analyse_profile(df["distance_m"].to_numpy(float), df["elevation_m"].to_numpy(float))
        if not m:
            continue
        cls = classify(m)
        ideal = estimate_ideal(inv, m["b"], m["plateau"], m["bh"])
        name = str(r["volcano"])
        colours.setdefault(name, REAL_COLOURS[len(colours) % len(REAL_COLOURS)])
        out.append({"label": f"{name} {int(r['azimuth_deg'])} deg", "group": name, "colour": colours[name],
                    "no_number": True, "b": m["b"], "bh": m["bh"], "plateau": m["plateau"],
                    "slope": m["slope_deg"], "asymmetry": m["asymmetry"], "height": m["height"],
                    "basal_width": m["basal_width"], "fit_r2": m["fit_r2"], "verdict": cls["verdict"],
                    "agreement": f"{cls['agreement']}/{cls['n']}", "notes": len(notes),
                    "ideal_e": ideal["curvature"] if ideal else np.nan,
                    "ideal_cut": ideal["cut"] if ideal else np.nan,
                    "ideal_bh": ideal["parent_bh"] if ideal else np.nan,
                    "on_lattice": bool(ideal["on_lattice"]) if ideal else False})
    return out


# ---------------------------------------------------------------------
# Collapsing the azimuths, and asymmetry
#
# Several transects through one volcano disagree for two different reasons: the
# edifice really is lopsided, or a transect missed the summit or ran into the sea
# or a neighbour. To compare them, every flank (two per transect) is first scaled
# by its OWN base, height and radius, so that only differences of shape remain:
#
#   shape spread    mean absolute difference between the flanks' scaled profiles, as a
#                   share of the mean flank's area. 0 = every flank has the same shape.
#   radius spread   standard deviation of the flank radii over their mean (size).
#   height spread   the same for the transect heights. A transect that misses the
#                   summit, or sits on a different base, shows up here.
#   left vs right   |R_right - R_left| / (R_right + R_left), averaged over transects.
#
# The raw area between the highest and lowest profile is NOT used: in elevation it picks
# up the ground around the volcano (sea on one side, a neighbouring massif on another)
# and the base level, and came out wider than the volcano itself.
# Raw profiles are not averaged either: the mean of eight profiles has a lower base than
# any of them, which inflated B/H by 35-50%. One point per volcano is the median of
# its transects' own measurements.
# ---------------------------------------------------------------------
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


def _half_height_u(g: np.ndarray) -> float:
    """Distance (as a share of the flank radius) at which the scaled flank first falls to half height."""
    below = np.flatnonzero(g <= 0.5)
    if len(below) == 0:
        return np.nan
    i = int(below[0])
    if i == 0:
        return float(FLANK_GRID[0])
    f = (g[i - 1] - 0.5) / (g[i - 1] - g[i]) if g[i - 1] != g[i] else 0.0
    return float(FLANK_GRID[i - 1] + f * (FLANK_GRID[i] - FLANK_GRID[i - 1]))


def summarise_azimuths(name: str, transects: list) -> dict:
    """One volcano from several transects: the median measurement, its range, and the asymmetry terms.

    transects: [{"az": bearing in degrees, "m": metrics from analyse_profile, "flanks": flank_profiles(...)}]
    """
    G = np.array([f["g"] for t in transects for f in t["flanks"]])
    radii = np.array([f["radius"] for t in transects for f in t["flanks"]])
    heights = np.array([t["m"]["height"] for t in transects])
    gmean, gmax, gmin = G.mean(axis=0), G.max(axis=0), G.min(axis=0)
    area = float(np.trapezoid(gmean, FLANK_GRID))
    ms = [t["m"] for t in transects]

    r_right = radii[0::2]
    r_left = radii[1::2]
    half = []
    for k, t in enumerate(transects):
        u_r, u_l = _half_height_u(G[2 * k]), _half_height_u(G[2 * k + 1])
        half.append((radii[2 * k] * u_r + radii[2 * k + 1] * u_l) / t["m"]["height"])

    return {
        "name": name, "n": len(transects), "azimuths": [t["az"] for t in transects],
        "b": float(np.median([m["b"] for m in ms])), "b_min": float(min(m["b"] for m in ms)),
        "b_max": float(max(m["b"] for m in ms)),
        "bh": float(np.median([m["bh"] for m in ms])), "bh_min": float(min(m["bh"] for m in ms)),
        "bh_max": float(max(m["bh"] for m in ms)),
        "plateau": float(np.median([m["plateau"] for m in ms])),
        "slope": float(np.median([m["slope_deg"] for m in ms])),
        "height": float(np.median(heights)), "basal_width": float(np.median([m["basal_width"] for m in ms])),
        "fit_r2": float(np.median([m["fit_r2"] for m in ms])),
        "asymmetry_app": float(np.mean([m["asymmetry"] for m in ms])),
        "fullness": float(np.mean([np.trapezoid(g, FLANK_GRID) for g in G])),
        "bh_half": float(np.nanmedian(half)),
        "shape_spread": float(np.mean([np.trapezoid(np.abs(g - gmean), FLANK_GRID) for g in G]) / area),
        "envelope": float(np.trapezoid(gmax - gmin, FLANK_GRID) / area),
        "radius_spread": float(radii.std() / radii.mean()),
        "height_spread": float(heights.std() / heights.mean()),
        "left_right": float(np.mean(np.abs(r_right - r_left) / (r_right + r_left))),
        "bundle": {"G": G, "gmean": gmean, "gmax": gmax, "gmin": gmin},
        "points": [(float(t["m"]["b"]), float(t["m"]["plateau"]), int(t["az"])) for t in transects],
        "warnings": int(sum(t.get("notes", 0) for t in transects)),
    }


@st.cache_data(show_spinner="Collapsing the azimuths...")
def load_real_volcanoes(key: str) -> list:
    """samples/real, one summary per volcano. Each also carries a plottable 'sample' at its median."""
    man = real_dir() / "manifest.csv"
    if not man.exists():
        return []
    inv = build_inverse(key)
    manifest = pd.read_csv(man)
    out = []
    for idx, name in enumerate(dict.fromkeys(manifest["volcano"])):
        transects = []
        for _, r in manifest[manifest["volcano"] == name].iterrows():
            f = real_dir() / str(r["file"])
            if not f.exists():
                continue
            df = pd.read_csv(f)
            x, z = df["distance_m"].to_numpy(float), df["elevation_m"].to_numpy(float)
            m, notes, _ = analyse_profile(x, z)
            if m:
                transects.append({"az": int(r["azimuth_deg"]), "m": m, "flanks": flank_profiles(x, z, m),
                                  "notes": len(notes)})
        if not transects:
            continue
        summ = summarise_azimuths(name, transects)
        summ["colour"] = REAL_COLOURS[idx % len(REAL_COLOURS)]
        summ["kind"] = str(manifest.loc[manifest["volcano"] == name, "kind"].iloc[0]) if "kind" in manifest else ""
        ideal = estimate_ideal(inv, summ["b"], summ["plateau"], summ["bh"])
        summ["sample"] = {
            "label": f"{name} (median of {summ['n']})", "group": name, "colour": summ["colour"], "no_number": True,
            "b": summ["b"], "bh": summ["bh"], "plateau": summ["plateau"], "slope": summ["slope"],
            "asymmetry": summ["asymmetry_app"], "height": summ["height"], "basal_width": summ["basal_width"],
            "fit_r2": summ["fit_r2"], "verdict": "-", "agreement": "-", "notes": summ["warnings"],
            "ideal_e": ideal["curvature"] if ideal else np.nan, "ideal_cut": ideal["cut"] if ideal else np.nan,
            "ideal_bh": ideal["parent_bh"] if ideal else np.nan,
            "on_lattice": bool(ideal["on_lattice"]) if ideal else False}
        out.append(summ)
    return out


def flank_bundle_figure(summaries: list):
    """Every flank of every volcano, scaled to the same base, height and radius, with the mean and the envelope."""
    n = len(summaries)
    cols = 4
    rows = -(-n // cols)
    fig = make_subplots(rows=rows, cols=cols, vertical_spacing=0.16, horizontal_spacing=0.05,
                        subplot_titles=[f"{s['name'].replace('Piton de la ', '')} ({s['shape_spread']:.2f})"
                                        for s in summaries])
    for i, s in enumerate(summaries):
        r, c = divmod(i, cols)
        b = s["bundle"]
        fig.add_trace(go.Scatter(x=np.concatenate([FLANK_GRID, FLANK_GRID[::-1]]),
                                 y=np.concatenate([b["gmax"], b["gmin"][::-1]]), fill="toself",
                                 fillcolor=_rgba(s["colour"], 0.18), line=dict(width=0), hoverinfo="skip",
                                 showlegend=False), row=r + 1, col=c + 1)
        for g in b["G"]:
            fig.add_trace(go.Scatter(x=FLANK_GRID, y=g, mode="lines",
                                     line=dict(width=0.8, color=_rgba(s["colour"], 0.55)),
                                     hoverinfo="skip", showlegend=False), row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=FLANK_GRID, y=b["gmean"], mode="lines", line=dict(width=3, color=s["colour"]),
                                 showlegend=False,
                                 hovertemplate="mean flank<br>%{x:.2f} of the radius, %{y:.2f} of the height<extra></extra>"),
                      row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=FLANK_GRID, y=1 - FLANK_GRID, mode="lines",
                                 line=dict(width=1, color="#8a8981", dash="dot"), hoverinfo="skip",
                                 showlegend=False), row=r + 1, col=c + 1)
    fig.update_xaxes(range=[0, 1], showgrid=False)
    fig.update_yaxes(range=[0, 1.05], showgrid=False)
    for ann in fig.layout.annotations:
        ann.font.size = 11
    fig.update_layout(template="plotly_white", height=260 * rows + 60, margin=dict(l=20, r=20, t=70, b=30),
                      title="Every flank scaled to its own base, height and radius "
                            "(shape spread in brackets; dotted = straight cone)")
    return fig


def asymmetry_figure(summaries: list):
    names = [s["name"] for s in summaries]
    fig = go.Figure()
    for key, label, colour in (("shape_spread", "shape spread between flanks", "#2a78d6"),
                               ("radius_spread", "radius spread (size)", "#eb6834"),
                               ("left_right", "left vs right width", "#1baf7a"),
                               ("height_spread", "height spread (flags a missed summit)", "#8a8981")):
        fig.add_trace(go.Bar(x=names, y=[s[key] for s in summaries], name=label, marker_color=colour,
                             hovertemplate="%{x}<br>" + label + " %{y:.2f}<extra></extra>"))
    fig.update_layout(template="plotly_white", barmode="group", height=380, margin=dict(l=20, r=20, t=60, b=90),
                      title="Asymmetry terms, each a fraction (0 = perfectly symmetric)", yaxis_title="fraction",
                      legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
    return fig


# ---------------------------------------------------------------------
# A curve through the real volcanoes
#
# A power law of flat-top width against b, fitted to the individual real transects:
#
#     flat-top = A x b^k
#
# Transects whose flat-top is more than 4 robust standard deviations from the median
# are left out (a transect through a caldera floor reads far higher than the rest), and
# the exclusions are reported. Four azimuths of one volcano are not four independent
# volcanoes, so every statement of uncertainty resamples or leaves out WHOLE volcanoes:
#
#   * the interval on k comes from resampling volcanoes with replacement;
#   * skill is the error predicting a volcano that was left out of the fit, against the
#     error of ignoring b altogether.
#
# This is a trend across different volcanoes, not a history of one. Calling it an
# evolution curve is a hypothesis that needs ages or an erosion state for each volcano.
# ---------------------------------------------------------------------
def _trend_keep(b: np.ndarray, y: np.ndarray) -> np.ndarray:
    ok = (b > 0) & (y > 0)
    med = np.median(y[ok])
    mad = 1.4826 * np.median(np.abs(y[ok] - med))
    return ok & (np.abs(y - med) <= 4 * mad) if mad > 0 else ok


def fit_shape_trend(points: list, n_boot: int = 1000, seed: int = 7) -> dict | None:
    """Power-law trend of flat-top width against b through real transects.

    points: [{"group": volcano name, "b": ..., "plateau": ...}, ...]. Returns None when there are too
    few transects or volcanoes to fit anything.
    """
    b = np.array([p["b"] for p in points], float)
    y = np.array([p["plateau"] for p in points], float)
    g = np.array([p["group"] for p in points])
    lab = np.array([p.get("label", p["group"]) for p in points])
    keep = _trend_keep(b, y)
    groups = list(dict.fromkeys(g[keep]))
    if keep.sum() < 8 or len(groups) < 3:
        return None
    k, la = np.polyfit(np.log(b[keep]), np.log(y[keep]), 1)
    sd = float((np.log(y[keep]) - (la + k * np.log(b[keep]))).std(ddof=2))
    grid = np.linspace(float(b[keep].min()), float(b[keep].max()), 60)

    rng = np.random.default_rng(seed)
    by_group = {gr: np.flatnonzero(keep & (g == gr)) for gr in groups}
    ks, curves = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(groups), len(groups))
        idx = np.concatenate([by_group[groups[i]] for i in pick])
        if np.unique(b[idx]).size < 4:
            continue
        kk, aa = np.polyfit(np.log(b[idx]), np.log(y[idx]), 1)
        ks.append(kk)
        curves.append(np.exp(aa + kk * np.log(grid)))
    curves = np.array(curves)

    err_curve, err_const = [], []
    for gr in groups:
        tr = keep & (g != gr)
        te = by_group[gr]
        kk, aa = np.polyfit(np.log(b[tr]), np.log(y[tr]), 1)
        err_curve += list(np.abs(np.exp(aa + kk * np.log(b[te])) - y[te]))
        err_const += list(np.abs(np.median(y[tr]) - y[te]))

    return {
        "A": float(np.exp(la)), "k": float(k), "sd": sd, "n": int(keep.sum()), "n_total": int(len(points)),
        "k_lo": float(np.percentile(ks, 2.5)), "k_hi": float(np.percentile(ks, 97.5)),
        "grid": grid, "curve": np.exp(la + k * np.log(grid)),
        "lo": np.percentile(curves, 2.5, axis=0), "hi": np.percentile(curves, 97.5, axis=0),
        "excluded": [(str(lab[i]), float(b[i]), float(y[i])) for i in np.flatnonzero(~keep)],
        "error_curve": float(np.mean(err_curve)), "error_constant": float(np.mean(err_const)),
        "keep": keep, "groups": groups,
    }


def shield_check(points: list, kinds: dict) -> list:
    """Do the shield volcanoes sit on a curve fitted WITHOUT any shield in it?

    kinds maps volcano name to a description; any containing 'shield' (basaltic or merely shield-shaped) is held out. Each held-out volcano
    gets the curve's flat-top at its median b, what it actually reads, and how many of its transects
    fall within 2 residual standard deviations of the curve.
    """
    shields = {n for n, kd in kinds.items() if "shield" in str(kd).lower()}
    # The same transects are left out as in the main fit (the rule is applied to ALL the points),
    # so the curve differs from it only by the shields being held back.
    b_all = np.array([p["b"] for p in points], float)
    y_all = np.array([p["plateau"] for p in points], float)
    use = _trend_keep(b_all, y_all) & np.array([p["group"] not in shields for p in points])
    if use.sum() < 8 or not shields:
        return []
    b, y, keep = b_all[use], y_all[use], np.ones(int(use.sum()), bool)
    k, la = np.polyfit(np.log(b[keep]), np.log(y[keep]), 1)
    sd = float((np.log(y[keep]) - (la + k * np.log(b[keep]))).std(ddof=2))
    out = []
    for name in dict.fromkeys(p["group"] for p in points):
        if name not in shields:
            continue
        mine = [p for p in points if p["group"] == name]
        bm = float(np.median([p["b"] for p in mine]))
        pred = float(np.exp(la + k * np.log(bm)))
        inside = sum(abs(np.log(p["plateau"]) - (la + k * np.log(p["b"]))) <= 2 * sd for p in mine)
        out.append({"name": name, "kind": kinds.get(name, ""), "b": bm,
                    "observed": float(np.median([p["plateau"] for p in mine])), "predicted": pred,
                    "inside": int(inside), "n": len(mine), "extrapolated": bool(bm > b[keep].max())})
    return out


def profile_plot(x, z, m: dict, label: str = "Profile"):
    """The measured transect with the summit, base and basal width marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=z, mode="lines", name=label,
                             line=dict(width=3, color="#0b0b0b"),
                             hovertemplate="%{x:,.0f}, %{y:,.1f}<extra></extra>"))
    if m:
        fig.add_hline(y=m["base_elev"], line=dict(color="#e34948", width=2, dash="dash"))
        fig.add_annotation(x=0.01, y=m["base_elev"], xref="paper", xanchor="left",
                           yanchor="bottom", text="basal level", showarrow=False,
                           font=dict(size=11, color="#e34948"))
        fig.add_trace(go.Scatter(
            x=[m["x_left"], m["x_right"]], y=[m["base_elev"]] * 2,
            mode="markers", name="edifice base",
            marker=dict(size=11, color="#e34948", symbol="line-ns-open",
                        line=dict(width=3, color="#e34948")),
            hovertemplate="base %{x:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[m["x_summit"]], y=[m["summit_elev"]], mode="markers", name="summit",
            marker=dict(size=13, color="#2a78d6", symbol="triangle-up",
                        line=dict(width=2, color="#fcfcfb")),
            hovertemplate="summit %{y:,.1f}<extra></extra>"))
    fig.update_layout(
        title="Radial transect as measured",
        xaxis_title="Distance along transect", yaxis_title="Elevation",
        template="plotly_white", height=430, hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0))
    fig.update_xaxes(gridcolor="#eceae4")
    fig.update_yaxes(gridcolor="#eceae4")
    return fig


# =====================================================================
# INTERFACE
# =====================================================================
for key, default in [("samples", []), ("current", None), ("prepared", None), ("notes", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(
    """
    <div style="padding:0.9rem 1.1rem; border-radius:12px; background:#eef6ff; margin-bottom:0.9rem;">
      <h1 style="margin:0; font-size:1.9rem;">Volcano Shape</h1>
      <p style="margin:0.35rem 0 0 0;">
        Classify a volcanic edifice as <b>shield</b> or <b>composite</b> from a single radial
        transect. Load a CSV of distance and elevation through the summit. Every file is
        processed identically, so results are directly comparable.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_run, tab_map, tab_about = st.tabs(["Analyse a profile", "Master plot", "About & citations"])


with tab_run:
    up = st.file_uploader("Radial transect CSV - distance in one column, elevation in the next",
                          type=["csv", "txt"])
    if up is not None:
        try:
            xr_, zr_, colnames = read_profile(up)
            m, notes, (xs_, zs_) = analyse_profile(xr_, zr_)
            st.session_state["prepared"] = (xs_, zs_)
            st.session_state["notes"] = notes
            if m:
                cls = classify(m)
                ideal = estimate_ideal(build_inverse(lattice_key()), m["b"], m["plateau"], m["bh"])
                st.session_state["current"] = {
                    "label": up.name.rsplit(".", 1)[0], "b": m["b"], "bh": m["bh"],
                    "plateau": m["plateau"], "slope": m["slope_deg"],
                    "asymmetry": m["asymmetry"], "height": m["height"],
                    "basal_width": m["basal_width"], "fit_r2": m["fit_r2"],
                    "verdict": cls["verdict"], "agreement": f"{cls['agreement']}/{cls['n']}",
                    "ideal_e": ideal["curvature"] if ideal else np.nan,
                    "ideal_cut": ideal["cut"] if ideal else np.nan,
                    "ideal_bh": ideal["parent_bh"] if ideal else np.nan,
                    "on_lattice": bool(ideal["on_lattice"]) if ideal else False,
                    "group": "",
                }
                st.caption(f"Read {len(xr_):,} rows from {up.name} - columns used: {colnames}")

                verdict = cls["verdict"]
                if verdict == "Shield":
                    st.success(f"### Shield volcano  ({cls['agreement']} of {cls['n']} indicators agree)")
                elif verdict == "Composite":
                    st.error(f"### Composite volcano  ({cls['agreement']} of {cls['n']} indicators agree)")
                else:
                    st.warning(f"### Ambiguous - the indicators disagree ({cls['agreement']} of {cls['n']})")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("b - flank curvature", f"{m['b']:.3f}",
                          help="Below 1 concave (composite-like), 1 a straight cone, above 1 convex (shield-like).")
                c2.metric("B/H", f"{m['bh']:.1f}",
                          help="Basal width over height. Higher means broader and flatter.")
                c3.metric("Mean flank slope", f"{m['slope_deg']:.1f} deg")
                c4.metric("Summit plateau", f"{m['plateau']:.3f}",
                          help=f"Plateau width as a fraction of basal width. A perfect cone "
                               f"reads {CONE_PLATEAU:.2f}; anything larger is a real flat top, "
                               f"crater or caldera.")

                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Height", f"{m['height']:,.0f}")
                d2.metric("Basal width", f"{m['basal_width']:,.0f}")
                d3.metric("Flank asymmetry", f"{m['asymmetry']:.3f}",
                          help="0 is perfectly symmetric. Above about 0.3, one flank dominates.")
                d4.metric("Power-law fit r2", f"{m['fit_r2']:.3f}")

                st.markdown("**Nearest ideal shape**")
                if ideal:
                    i1, i2, i3 = st.columns(3)
                    i1.metric("Curvature e", f"{ideal['curvature']:.2f}",
                              help="The flank exponent of the ideal shape that matches: below 1 concave, "
                                   "1 a straight cone, above 1 convex. Read back from b and the plateau.")
                    i2.metric("Summit cut", f"{ideal['cut']:.0%}",
                              help="How much of the ideal's height has been sliced off the top to give "
                                   "the measured flat top. 0% is an uncut summit.")
                    i3.metric("Parent B/H", f"{ideal['parent_bh']:.1f}",
                              help="The B/H the ideal would have before the cut. The measured B/H is "
                                   "higher than this by roughly 1 / (1 - cut).")
                    if ideal["on_lattice"]:
                        st.caption(f"Reads as {describe_ideal(ideal)}.")
                    else:
                        st.warning(
                            "This measurement lies off the ideal-shape lattice: no combination of curvature "
                            "and summit cut reproduces this b and plateau together. The numbers above are "
                            "only the nearest node and should not be quoted. Read the position from the "
                            "Master plot instead.")

                st.markdown("**How each indicator voted**")
                st.dataframe(pd.DataFrame(
                    [{"Indicator": a, "Measured": b_, "Shield when": c_, "Vote": d_, "Source": e_}
                     for a, b_, c_, d_, e_ in cls["detail"]],
                ), width="stretch", hide_index=True)

                for c_ in cls.get("caveats", []):
                    st.warning(c_)
                for n_ in notes:
                    st.warning(n_)

                st.plotly_chart(profile_plot(xs_, zs_, m, up.name), width="stretch")

                st.markdown("---")
                p1, p2, p3 = st.columns([1.5, 1.1, 0.9], gap="medium")
                with p1:
                    lbl = st.text_input("Label", value=st.session_state["current"]["label"])
                with p2:
                    grp = st.text_input("Group (optional)",
                                        help="Samples sharing a group get the same marker shape - "
                                             "for example Mars, Earth, Venus.")
                with p3:
                    st.write("")
                    if st.button("Pin to master plot", width="stretch", type="primary"):
                        e = dict(st.session_state["current"])
                        e["label"] = lbl or e["label"]
                        e["group"] = grp or ""
                        st.session_state["samples"] = st.session_state["samples"] + [e]
                        st.success(f"Pinned '{e['label']}'. Open the Master plot tab.")
            else:
                st.session_state["current"] = None
                for n_ in notes:
                    st.error(n_)
        except Exception as exc:
            st.session_state["current"] = None
            st.error(f"Could not read this file: {exc}")
    else:
        st.info(
            "Load a CSV to begin. It needs two numeric columns: distance along a radial "
            "transect that crosses the summit, and elevation. Units do not matter as long as "
            "both columns use the same one - every reported shape number is size-independent."
        )


with tab_map:
    lat = build_lattice(lattice_key())
    named, cut_series = build_named(lattice_key())
    style = st.radio(
        "Plot style",
        ["Named shapes", "Pairwise panels", "3-D", "b vs B/H", "b vs plateau"],
        horizontal=True,
        help="Named shapes is the place to start: the triangle, circle, square and their "
             "relatives, drawn and placed on the map. The other four views plot the same "
             "measurements against a finer grid of ideal shapes.",
    )
    if style == "Named shapes":
        real = load_real_transects(lattice_key()) if real_dir().exists() else []
        vols = load_real_volcanoes(lattice_key()) if real else []
        o1, o2, o3 = st.columns([1.5, 1.2, 1.5])
        show_real = o1.checkbox(
            f"Add the {len(real)} real SRTM transects ({len(vols)} volcanoes, 4 azimuths each)",
            value=False, disabled=not real,
            help="Radial transects through " + (", ".join(v["name"] for v in vols) or "no volcanoes") +
                 ", fetched from SRTM 90 m (samples/real). They are measured by exactly the code an "
                 "uploaded file goes through. One colour per volcano, one marker per azimuth.")
        zoom_map = o2.checkbox("Zoom the map to the bottom-left", value=False,
                               help="Where the real volcanoes sit. The shape drawings are hidden when "
                                    "zoomed because they are drawn at map scale.")
        collapse = o3.checkbox(
            "One point per volcano (median of its azimuths)", value=False, disabled=not (real and show_real),
            help="Collapses each volcano's four azimuths to the median of their own measurements, with "
                 "each azimuth drawn as a spoke so the scatter stays visible. Raw profiles are not "
                 "averaged: that lowers the base and inflates B/H.")
        t1, _t2 = st.columns([1.9, 2.8])
        show_trend = t1.checkbox(
            "Draw the fitted curve through the real transects", value=False, disabled=not (real and show_real),
            help="A power law of flat-top width against b fitted to the individual transects, leaving out "
                 "any transect far above the rest. Its uncertainty comes from resampling whole volcanoes.")
        trend = fit_shape_trend(real) if (show_real and show_trend and real) else None
        collapsed = bool(show_real and collapse and vols)
        if collapsed:
            plotted = st.session_state["samples"] + [v["sample"] for v in vols]
            spokes = [{"name": v["name"], "colour": v["colour"], "cx": v["b"], "cy": v["plateau"],
                       "points": v["points"]} for v in vols]
        else:
            plotted = st.session_state["samples"] + (real if show_real else [])
            spokes = None
        left, right = st.columns([1, 1.1], gap="medium")
        with left:
            st.plotly_chart(named_overlay_figure(named, height=600), width="stretch")
        with right:
            st.plotly_chart(
                shape_map_figure(named, cut_series, lat, plotted, st.session_state["current"],
                                 height=600, zoom=zoom_map, spokes=spokes, trend=trend),
                width="stretch")
        st.plotly_chart(named_squash_figure(), width="stretch")
        if trend:
            st.markdown("#### The fitted curve")
            left_out = "; ".join(f"{lab} (b {bx:.2f}, flat-top {yy:.3f})" for lab, bx, yy in trend["excluded"])
            st.markdown(
                f"**flat-top width = {trend['A']:.3f} x b^{trend['k']:.2f}**, fitted to {trend['n']} of "
                f"{trend['n_total']} real transects" + (f"; left out as far above the rest: {left_out}." if left_out else ".")
            )
            st.markdown(
                f"Resampling whole volcanoes puts the exponent between {trend['k_lo']:.2f} and {trend['k_hi']:.2f} "
                f"(0 would mean no trend). Predicting a volcano that was left out of the fit, the curve is off by "
                f"{trend['error_curve']:.3f} on average, against {trend['error_constant']:.3f} for ignoring b. "
                f"The transects scatter about it by a factor of {np.exp(trend['sd']):.2f} either way, so the "
                "band is wide."
            )
            kinds = {v["name"]: v.get("kind", "") for v in vols}
            sh = shield_check(real, kinds)
            if sh:
                st.markdown("**Do the shield-shaped volcanoes sit on it?** The curve is refitted without them, then "
                            "each is read against it. Only a basaltic shield is a true shield: a volcano with a "
                            "shield's shape can still be a composite (see the Kind column).")
                st.dataframe(pd.DataFrame([{
                    "Volcano": r["name"], "Kind": r["kind"], "Median b": round(r["b"], 2),
                    "Flat-top it reads": round(r["observed"], 3), "Curve without the shields": round(r["predicted"], 3),
                    "Reads / curve": round(r["observed"] / r["predicted"], 2),
                    "Transects within 2 sd": f"{r['inside']} of {r['n']}",
                    "Beyond the b range fitted": "yes" if r["extrapolated"] else "no"} for r in sh]),
                    width="stretch", hide_index=True)
            bend_line = lat[np.isclose(lat["Cut"], 0.0)].sort_values("Curvature e")
            above = trend["grid"] >= float(bend_line["b"].min())
            ratio = trend["curve"][above] / np.interp(trend["grid"][above], bend_line["b"], bend_line["Plateau"]) if above.any() else []
            st.caption(
                "This is a trend across different volcanoes, with different rocks, settings and ages. It shows how the "
                "shape numbers go together, not how one volcano changes with time. Calling it an evolution curve is a "
                "hypothesis that needs an age or an erosion state for each volcano."
                + (f" The curve runs {min(ratio):.1f} to {max(ratio):.1f} times above the black bend line over the data, "
                   "which is the extra summit flattening that craters add to what the sides' concavity alone gives." if len(ratio) else "")
            )
        if show_real and vols:
            st.markdown("#### The real volcanoes, one row each")
            vt = pd.DataFrame([{
                "Volcano": v["name"], "Azimuths": v["n"],
                "b (median)": round(v["b"], 2), "b range": f"{v['b_min']:.2f} - {v['b_max']:.2f}",
                "B/H (median)": round(v["bh"], 1), "B/H range": f"{v['bh_min']:.0f} - {v['bh_max']:.0f}",
                "Flat-top": round(v["plateau"], 3), "Fullness": round(v["fullness"], 2),
                "Half-height width / height": round(v["bh_half"], 1),
                "Shape spread": round(v["shape_spread"], 2), "Radius spread": round(v["radius_spread"], 2),
                "Left vs right width": round(v["left_right"], 2), "Height spread": round(v["height_spread"], 2),
                "Warnings": v["warnings"]} for v in vols])
            st.dataframe(vt, width="stretch", hide_index=True)
            st.markdown(
                "**Fullness** is the area under the profile divided by base times height: 0.17 for the concave "
                "parabola, 0.50 for the triangle, 0.78 for the semicircle. **Half-height width / height** is "
                "B/H measured at mid-height, so the long skirt that makes B/H uncertain does not enter it. "
                "The four **spreads** compare the volcano's eight flanks (two per transect) after each is "
                "scaled by its own base, height and radius: *shape* is how much their profiles differ, *radius* "
                "and *height* how much their sizes differ, *left vs right* how unequal the two sides of a "
                "transect are. 0 means perfectly symmetric."
            )
            st.plotly_chart(flank_bundle_figure(vols), width="stretch")
            st.plotly_chart(asymmetry_figure(vols), width="stretch")
            flagged = [v["name"] for v in vols if v["height_spread"] > 0.08]
            widest = max(vols, key=lambda v: v["b_max"] - v["b_min"])
            st.caption(
                "Every spread is an upper bound on the volcano's real asymmetry. The centres are approximate, "
                "so some transects are chords that miss the summit, and the sea or a neighbouring edifice "
                "lies inside some of them. Height spread is the flag for that: a transect that misses the "
                "summit or sits on a different base makes it large"
                + (f" (here {', '.join(flagged)})" if flagged else "")
                + f". Shape spread runs from {min(v['shape_spread'] for v in vols):.2f} to "
                f"{max(v['shape_spread'] for v in vols):.2f} across these volcanoes. Azimuths of one volcano "
                f"can disagree a lot: {widest['name']}'s b runs from {widest['b_min']:.2f} to {widest['b_max']:.2f}, "
                "so four or more azimuths are needed before a single volcano's b or B/H can be trusted."
            )
            with st.expander("Every transect, as measured"):
                tab = pd.DataFrame([{"Volcano": s["group"], "Azimuth": s["label"].rsplit(" ", 2)[-2],
                                     "Height (m)": round(s["height"]), "Basal width (km)": round(s["basal_width"] / 1000, 1),
                                     "B/H": round(s["bh"], 1), "b": round(s["b"], 3),
                                     "Flat-top": round(s["plateau"], 3), "Mean slope": round(s["slope"], 1),
                                     "Fit r2": round(s["fit_r2"], 2),
                                     "App's verdict": f"{s['verdict']} ({s['agreement']})",
                                     "Warnings": s["notes"]} for s in real])
                st.dataframe(tab, width="stretch", hide_index=True)
                st.caption("Centres are approximate summit or caldera-centre coordinates. Where the sea or a "
                           "neighbouring edifice lies within the transect, the base level is set by that.")

        st.markdown("#### How to read these")
        st.markdown(
            "**Left: the shapes.** All drawn on the same base and the same height, so they nest "
            "inside one rectangle. Each legend entry gives the two numbers that place that shape "
            "on the map: **b**, how bowed the sides are (1 is straight, above 1 bulging outward "
            "like a dome, below 1 sagging inward like a steep stratovolcano), and the **flat-top "
            "width** as a share of the base."
        )
        st.markdown(
            "**Right: the shape map.** Left to right is how bowed the sides are; bottom to top is "
            "how much of the base the flat top covers. Each perfect shape is one point. The "
            "**black line** bends the sides of an uncut shape from concave, through the straight "
            "triangle and the circle, toward a square. The **dashed lines** cut the top off a "
            "shape: for the triangle the trapezoid is a 30% cut and the near-square a 95% cut. "
            "Your samples are added as numbered black markers and stay there as you load more "
            "files, so you build one diagram across a whole study."
        )
        st.markdown(
            "**Bottom: squashing a shape.** Lowering the height on the same base turns a circle "
            "into an ellipse and a triangle into a flatter triangle. This changes **B/H** (base "
            "width divided by height) and nothing else: b and the flat-top width stay exactly "
            "where they were, so a squashed shape does not move on the map. Real volcanoes are "
            "far flatter than the shapes at the top: a median composite has B/H of about 8 and "
            "shields run from 10 to 100 or more (Grosse et al.), so the ones they resemble look "
            "like the thin curves here."
        )
        st.caption(
            "A circle's b is 1.62, not 2: b is fitted over the whole side, and a circle's side "
            "steepens to vertical at the base, which pulls the fitted exponent down. Cutting a "
            "straight-sided shape never changes b, but cutting a curved one lowers it a little. "
            "Near a rectangle (flat top above 0.85) b stops meaning anything, and the app says so."
        )
        with st.expander("The numbers behind the pictures"):
            shown = named[["Shape", "B/H set", "B/H", "b", "Flat-top", "Slope (deg)"]].copy()
            shown["B/H"] = shown["B/H"].round(2)
            shown["b"] = shown["b"].round(3)
            shown["Flat-top"] = shown["Flat-top"].round(3)
            shown["Slope (deg)"] = shown["Slope (deg)"].round(1)
            st.dataframe(shown, width="stretch", hide_index=True)
            st.caption("B/H set is the squash level the shape was built at; B/H is what the app "
                       "measured back from it. b and flat-top are the same at every level.")
    elif style == "3-D":
        st.plotly_chart(
            master_plot_3d(lat, st.session_state["samples"], st.session_state["current"]),
            width="stretch")
        st.caption(
            "Drag to rotate, scroll to zoom. The loaded profile is the red diamond; pinned "
            "samples are numbered black markers. If a position looks ambiguous, switch to the "
            "pairwise panels - depth on a flat screen is genuinely unreliable."
        )
    elif style == "Pairwise panels":
        st.plotly_chart(
            master_plot_pairs(lat, st.session_state["samples"], st.session_state["current"]),
            width="stretch")
        st.caption(
            "Three panels, one per pair of measurements. Dotted lines mark b = 1 (a straight "
            "cone) and B/H = 10 (the approximate shield boundary). In the B/H panels the grid is "
            "drawn for a parent of B/H " + ", ".join(f"{p:g}" for p in LATTICE["parent_bh"]) +
            " before any cut. A parent twice as broad sits twice as high: B/H scales with it "
            "exactly, while b and the plateau do not move."
        )
    else:
        yaxis = ("B/H  (proportions)" if style == "b vs B/H"
                 else "Summit plateau  (flat top / crater)")
        st.plotly_chart(
            master_plot(lat, st.session_state["samples"], st.session_state["current"], yaxis),
            width="stretch")
    if style == "b vs plateau":
        st.caption(
            "This is the shape plane: it does not depend on proportions. Uncut shapes (the "
            "'uncut' line) run from concave to convex and climb as they bend, because a convex "
            "top is already flat at the 5% level. A cut lifts a shape straight up off that line "
            "without changing b on a straight cone, so the height above the uncut line reads "
            "summit truncation."
        )

    if style != "Named shapes":
        k1, k2, k3 = st.columns(3, gap="medium")
        cards = [
            (k1, CURVE_COLOURS["convex"], "Curvature e (solid lines)",
             "Bends the flanks. e = 1 is a straight cone, e = 2 a circle; below 1 the flanks are "
             "concave (composite-like); above 1 they are convex (shield-like)."),
            (k2, CUT_COLOUR, "Summit cut (dashed lines)",
             "Slices the top off, flattening the summit and lowering the height but not the width. "
             "A cut raises B/H, lifts the plateau, and leaves b at 1 on a straight cone."),
            (k3, CURVE_COLOURS["straight"], "Parent B/H",
             "The proportions the shape had before the cut. The B/H panels draw one parent "
             f"(B/H {', '.join(f'{p:g}' for p in LATTICE['parent_bh'])}); a parent twice as broad sits "
             "twice as high, because B/H scales with it exactly."),
        ]
        for col, colour, title, text in cards:
            with col:
                st.markdown(
                    f"<div style='border-left:5px solid {colour}; padding-left:8px;'>"
                    f"<b>{title}</b><br><span style='font-size:0.85em; color:#52514e;'>"
                    f"{text}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state["samples"]:
        rows = []
        for i, s in enumerate(st.session_state["samples"], start=1):
            rows.append({"#": i, "Label": s["label"], "Group": s["group"],
                         "Verdict": s["verdict"], "Agreement": s["agreement"],
                         "b": round(s["b"], 3), "B/H": round(s["bh"], 1),
                         "Slope (deg)": round(s["slope"], 1),
                         "Plateau": round(s["plateau"], 3),
                         "Ideal e": round(s.get("ideal_e", np.nan), 2),
                         "Ideal cut": round(s.get("ideal_cut", np.nan), 3),
                         "Parent B/H": round(s.get("ideal_bh", np.nan), 1),
                         "On lattice": s.get("on_lattice", False),
                         "Asymmetry": round(s["asymmetry"], 3),
                         "Height": round(s["height"], 1),
                         "Basal width": round(s["basal_width"], 1)})
        sdf = pd.DataFrame(rows)
        st.dataframe(sdf, width="stretch", hide_index=True)

        b1, b2, b3 = st.columns([1.4, 1.0, 1.0], gap="medium")
        with b1:
            drop = st.multiselect("Remove samples",
                                  [f"{r['#']} - {r['Label']}" for r in rows])
        with b2:
            st.write("")
            if st.button("Remove selected", width="stretch", disabled=not drop):
                kill = {int(d.split(" - ", 1)[0]) for d in drop}
                st.session_state["samples"] = [
                    s for i, s in enumerate(st.session_state["samples"], start=1) if i not in kill]
                st.rerun()
        with b3:
            st.write("")
            if st.button("Clear the whole plot", width="stretch"):
                st.session_state["samples"] = []
                st.rerun()

        st.download_button("Download this master plot as CSV",
                           data=sdf.to_csv(index=False).encode("utf-8"),
                           file_name="volcano_shape_master_plot.csv", mime="text/csv")
    else:
        st.info("No samples pinned yet. Analyse a profile and press 'Pin to master plot'.")

    with st.expander("The ideal shapes, as measured"):
        st.caption(
            "Every ideal is put through the same measurement code as your data, so the grid "
            "and the samples are always on the same footing. B/H is for a parent of B/H "
            f"{LATTICE['reference_bh']:g}; b and the plateau are the same for any parent."
        )
        st.dataframe(lat, width="stretch", hide_index=True)


with tab_about:
    st.markdown("""
## What this does

You give it one **radial transect** across a volcano - a line of distance and elevation
running through the summit and out onto the surrounding ground on both sides. It returns
a classification, **shield** or **composite**, and places the edifice on a diagram of ideal
geometric shapes.

## Why transects

The standard tools for volcano morphometry need a gridded digital elevation model and a
mapped basal outline. Many planetary bodies have neither. What they do have is **altimeter
tracks** - one-dimensional profiles from Magellan at Venus, SARTopo at Titan, MLA over
Mercury's northern hemisphere. A method that works from a single profile can be applied
where the grid-based tools cannot run at all.

That is the point of this app, and it is the claim the project has to defend.

## How it measures

1. **Resample and smooth, twice.** A first pass puts the whole track on an even grid and locates
   the edifice. The track is then cropped to the edifice plus a margin and resampled and smoothed
   again at the scale of the edifice, so the answer does not depend on how much surrounding
   ground the file happens to contain. Every step is fixed, so every file gets identical treatment.
2. **Find the summit**: the centre of the highest ground. Only high ground connected to the
   highest point counts, and the two rims of a crater or caldera (closer together than a quarter
   of the basal width) count as one summit. A separate high point elsewhere on the edifice is
   reported, not averaged in.
3. **Find the base.** Each flank's base is a low quantile of that flank's elevations, not its
   single lowest point, so one stray reading on the surrounding plain cannot capture the
   measurement. The higher of the two bases sets one horizontal basal level.
4. **Measure.**
   - **b**, flank curvature, from a log-log fit of drop against distance, both measured from the
     edge of the summit plateau so that a flat top does not masquerade as curvature. A straight
     cone gives exactly 1, however deeply its top is cut. Convex flanks give more, concave less.
   - **B/H**, basal width over height.
   - **Mean flank slope**, weighted by distance from the summit so that it reproduces the
     area-weighted mean a DEM-based method returns rather than a plain profile average.
   - **Summit plateau**, the width of near-summit ground as a fraction of basal width.
   - **Flank asymmetry**, the imbalance in cross-sectional area either side of the summit.
5. **Classify** on three independent indicators and report how many agree.

## What the diagram shows

The grid behind the plot is a **lattice of perfect geometric shapes**, built from two controls
applied to a parent triangle:

- **Curvature e** bends the flanks: e = 1 is a straight cone, below 1 concave (composite-like),
  above 1 convex (shield-like).
- **Summit cut** slices the top off horizontally, taking a fraction of the height. The flanks and
  base are unchanged, so the summit becomes a flat top, like a crater floor or caldera rim.

Each ideal is measured by the same code as your profile, so the grid and your samples are always
on the same footing. Three facts organise the diagram:

- **b and the plateau do not depend on proportions**, since stretching a shape vertically changes
  neither. They place a shape in the *shape plane*; B/H is a separate axis.
- **Cutting a straight cone leaves b at 1 however deep the cut.** b alone cannot see a truncated
  summit. On curved shapes the plateau already grows with curvature (the "uncut" line), and a cut
  lifts it off that line, so the height above the uncut line reads truncation.
- **Cutting the top lowers the height but not the width, so B/H rises by roughly 1 / (1 - cut).**
  A B/H of 20 with b = 1 and a wide plateau is a cut-down cone, not a shield. A shield is convex.

Three measurements (b, plateau, B/H) recover the three controls (curvature, cut, parent B/H),
reported on the Analyse tab as the **nearest ideal shape**. On random ideal shapes the median
error is about 0.01 in curvature and 0.003 in cut depth without noise (0.1 and 0.02 with noise
of 2% of the height); strongly convex shapes are less well determined because b saturates. That
is a test of the measurements, not of the idea that real volcanoes are cut superellipses: a
sample that no ideal reproduces is flagged as off the lattice.

## Limits you should state in any write-up

- **One profile is not an edifice.** Volcanoes are not radially symmetric. Sector collapses,
  rift zones and flank vents bias any single azimuth. Run several azimuths per volcano and
  compare them.
- **A track that misses the summit is not a radial profile.** Altimeter tracks are chords. On
  a straight cone, a chord offset from the summit by 0.3 of the radius reads b = 1.20 (it looks
  convex), B/H about 36 percent too high and a plateau about three times too wide, and the
  asymmetry measure does not notice because the chord is itself symmetric. The size of the bias
  depends on the shape: a convex ellipse (e = 2) is unaffected, since its chords are ellipses of
  the same proportions. Report how far each track passes from the summit, and prefer tracks
  that cross it.
- **The thresholds are Earth-calibrated** and are not rescaled for gravity here. Martian
  shields sit at the low end of the terrestrial envelope, so the direction holds for Mars, but
  it is not validated for higher-gravity bodies and is meaningless for ice edifices.
- **The slope threshold is biased against composites here.** The 12 degree boundary was
  calibrated on DEM-derived slopes. A single profile measures only the along-track gradient
  and misses the cross-track component, so it under-reads a DEM mean slope by roughly 15-25
  percent once any surface texture is present. Profile slopes therefore read low against this
  threshold and the slope vote leans toward Shield. Quantifying that offset against gridded
  Mars data is a worthwhile part of the project; do not simply adjust the threshold to taste.
- **The classes genuinely overlap.** Grosse et al. (2013) measured 759 composite volcanoes
  and report that most morphometric parameters show large variation without clear separations.
  Disagreement between the three indicators is information, not a malfunction.
- **Basal delineation dominates the error budget.** van Wees et al. (2024) had seven experts
  draw boundaries on the same 16 volcanoes and found the boundary choice mattered more than
  which DEM was used. A single profile makes that harder, not easier.
- **Erosion flattens composites** toward shield-like slopes, and impact gardening does the
  same on airless bodies. A shape class is a statement about present form, not about
  eruptive style.

## Sources

**Thresholds used for classification**

- Grosse, P. & Kervyn, M. (2018). Morphometry of terrestrial shield volcanoes.
  *Geomorphology* 304, 1-14. *(158 shields; H/W_B mostly 0.01-0.10 and average slopes mostly
  1-15 deg, with an approximate shield/stratovolcano boundary at 12 deg and H/W_B 0.10.)*
- Svensson, H. (1959). Is the cross-section of a glacial valley a parabola?
  *Journal of Glaciology* 3(25), 362-363. *(Origin of the power-law form exponent b.)*

**Context and comparison**

- Grosse, P., Euillades, P.A., Euillades, L.D. & van Wyk de Vries, B. (2012). Systematic
  morphometric characterization of volcanic edifices using digital elevation models.
  *Geomorphology* 136, 114-131. *(MORVOLC, the DEM-based standard.)*
- Grosse, P., Euillades, P.A., Euillades, L.D. & van Wyk de Vries, B. (2013/2014). A global
  database of composite volcano morphometry. *Bulletin of Volcanology* 76, 784.
  *(~750 composite volcanoes: H 100-2500 m median 1020; W_B 2-36 km median 8.8; H/W_B
  0.01-0.28 median 0.12; average slope 3-30 deg median 17.)*
- van Wees, R.M.J., O'Hara, D., Kereszturi, G., Grosse, P., Lahitte, P., Tournigand, P.-Y. &
  Kervyn, M. (2024). Towards more consistent volcano morphometry datasets. *Geomorphology*
  465, 109381. *(Boundary delineation matters more than DEM choice.)*
- Karatson, D., Favalli, M., Tarquini, S., Fornaciai, A. & Worner, G. (2010). The regular
  shape of stratovolcanoes. *J. Volcanol. Geotherm. Res.* 193, 171-181.
  *(C-type and P-type upper-flank forms, correlating with SiO2 and H2O.)*
- Li, Y., Liu, G. & Cui, Z. (2001). Glacial valley cross-profile morphology, Tian Shan
  Mountains, China. *Geomorphology* 38, 153-166. *(b is sensitive to the choice of origin.)*

**Planetary application**

- Plescia, J.B. (2004). Morphometric properties of Martian volcanoes. *J. Geophys. Res.
  Planets* 109, E03003. *(Per-volcano heights, diameters and flank slopes: a ready-made
  comparison set.)*
- Head, J.W. et al. (1998). Characterization of major volcanic edifices on Mars using MOLA
  data. *LPSC XXIX*, abstract 1322. *(Morphometry from individual MOLA tracks - the profile
  approach is not new for Mars, and should not be claimed as such.)*
- Ivanov, M.A. & Head, J.W. (2025). Large volcanoes on Venus: morphology, morphometry and
  stratigraphy. *Icarus* 429, 116404. *(Diameters 74-743 km; aspect ratios far lower than
  terrestrial shields.)*
- Bleacher, J.E. & Greeley, R. (2003). Shield volcano slope distributions. *LPSC XXXIV*,
  abstract 1794. *(Pixel-scale slope distributions - a different quantity from the
  edifice-average slope used here; do not compare them directly.)*

## Fixed processing parameters

These are hardwired so that every file is treated identically. They are listed here for the
methods section of any write-up.
""")
    st.dataframe(
        pd.DataFrame([{"Parameter": k, "Value": v} for k, v in PROCESSING.items()]
                     + [{"Parameter": "threshold_slope_deg", "Value": THRESHOLD_SLOPE_DEG},
                        {"Parameter": "threshold_BH", "Value": THRESHOLD_BH}]),
        width="stretch", hide_index=True)
