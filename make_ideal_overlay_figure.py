"""
Figure: real volcano profiles compared with perfect shapes.

The sand-grain parallel. Wadell (1932, 1935) scored a sand grain against the
sphere of equal volume, so a perfect sphere reads 1 and nature's departure from
it carries the story. Powers (1953) turned that into a visual comparison chart:
a row of reference outlines, and you place your grain against them.

This figure is that chart for volcanoes.

The reference row is one continuous family of perfect shapes, the superellipse

    z / H  =  [ 1 - u^e ] ^ (1/e)        u = horizontal distance / half width

where e alone sets the flank curvature: e < 1 concave (steep summit, flaring
apron), e = 1 a straight-sided cone, e > 1 convex. Four of the named shapes from
the app are members of this one family rather than separate ideas - e = 0.5 is
the concave parabola, e = 1 the isosceles triangle, e = 2 the semicircle, e = 4
the rounded square - which is why the family is used here instead of a set of
unrelated outlines.

Below the reference row, real radial transects. Each perfect shape is pinned the
way Doug specified: apex at the measured summit, base corners at the measured
basal points, so the two flanks may have different widths and the asymmetry stays
visible instead of being averaged away. One exponent e is fitted to both flanks
at once by least squares; the shaded area is the ground that lies above (blue) or
below (red) that perfect shape.

Measurement of base, summit, height, b and B/H is done by volcanoShape.py, so the
numbers here and the numbers in the app are the same numbers.

Run:  py make_ideal_overlay_figure.py
Out:  figures/ideal_overlay.png, .pdf
      figures/ideal_overlay_values.csv          one row per volcano (the figure)
      figures/ideal_overlay_all_azimuths.csv    one row per transect (all 32)
"""

import ast
import glob
import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
APP = HERE / "volcanoShape.py"
REAL = HERE / "samples" / "real"
OUT = HERE / "figures"

# Two inks and a diverging pair. Excess and deficit are a polarity, not two
# categories, so they take the warm/cool poles. Everything else is ink, which
# keeps the figure readable in greyscale and in print - the reference charts
# this imitates are greyscale for exactly that reason.
INK = "#0b0b0b"
MUTED = "#8a8981"
FAINT = "#d8d6ce"
EXCESS = "#2a78d6"
DEFICIT = "#d03b3b"

# The reference row. Named shapes are labelled where the family passes through
# them; they are not separate shapes.
REFERENCE = [
    (0.40, ""),
    (0.50, "concave parabola"),
    (0.70, ""),
    (1.00, "isosceles triangle"),
    (2.00, "semicircle"),
    (4.00, "rounded square"),
]

E_GRID = np.round(np.arange(0.20, 6.001, 0.01), 3)


def load_app():
    """Borrow volcanoShape.py's analysis layer without starting its interface."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.interpolate import LinearNDInterpolator
    from scipy.signal import find_peaks, savgol_filter
    from scipy.spatial import cKDTree

    class _Stub:
        def __getattr__(self, _k):
            def deco(*a, **kw):
                if a and callable(a[0]):
                    return a[0]
                return lambda f: f
            return deco

    ns = {"np": np, "pd": pd, "go": go, "make_subplots": make_subplots,
          "savgol_filter": savgol_filter, "find_peaks": find_peaks,
          "cKDTree": cKDTree, "LinearNDInterpolator": LinearNDInterpolator,
          "st": _Stub(), "io": __import__("io"), "json": __import__("json"),
          "pathlib": pathlib, "__builtins__": __builtins__}
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.ClassDef)):
            try:
                exec(compile(ast.Module([node], []), "<app>", "exec"), ns)
            except Exception:
                pass
    return ns


def superellipse(u, e):
    """The perfect shape. u = 0 at the apex, u = 1 at the base corner."""
    return np.clip(1.0 - np.clip(u, 0.0, 1.0) ** e, 0.0, None) ** (1.0 / e)


def apex_normalise(x, z, m):
    """Crop to the edifice and map it into the unit box, apex-pinned.

    u runs 0 at the summit to 1 at each basal point, so the left and right
    flanks are each stretched onto their OWN half width. An off-centre summit
    is therefore not straightened out - it shows as two flanks of different
    real width, which is what the asymmetry column reports.
    """
    xl, xr, xs = m["x_left"], m["x_right"], m["x_summit"]
    sel = (x >= xl) & (x <= xr)
    xx, zz = x[sel], z[sel]
    v = np.clip((zz - m["base_elev"]) / m["height"], 0.0, 1.0)
    left = max(xs - xl, 1e-9)
    right = max(xr - xs, 1e-9)
    u = np.where(xx <= xs, (xs - xx) / left, (xx - xs) / right)
    # Signed horizontal coordinate for drawing: each flank keeps its real share
    # of the basal width, so the panel is drawn to the measured proportions.
    span = left + right
    s = np.where(xx <= xs, -(xs - xx) / span, (xx - xs) / span)
    return u, v, s, left / span


def fit_exponent(u, v):
    """The member of the family that fits best, with one free exponent.

    No free scale factor is needed: height and width are already normalised by
    the measured base and summit, so the fit is pure shape.
    """
    sst = float(np.sum((v - v.mean()) ** 2))
    if sst <= 0:
        return np.nan, np.nan
    best_e, best_r2 = np.nan, -np.inf
    for e in E_GRID:
        ssr = float(np.sum((v - superellipse(u, e)) ** 2))
        r2 = 1.0 - ssr / sst
        if r2 > best_r2:
            best_e, best_r2 = float(e), r2
    return best_e, best_r2


def measure_all(analyse):
    """Every transect, measured once."""
    rows = []
    for f in sorted(glob.glob(str(REAL / "*_az*.csv"))):
        base = os.path.basename(f)
        key = base.split("_az")[0]
        az = base.split("_az")[1][:3]
        d = pd.read_csv(f)
        x = d.iloc[:, 0].to_numpy(float)
        z = d.iloc[:, 1].to_numpy(float)
        m, notes, _track = analyse(x, z)
        if not m or not np.isfinite(m.get("b", np.nan)):
            continue
        u, v, s, lshare = apex_normalise(x, z, m)
        e, r2 = fit_exponent(u, v)
        rows.append({"key": key, "az": az, "x": x, "z": z, "m": m,
                     "u": u, "v": v, "s": s, "lshare": lshare,
                     "e": e, "e_r2": r2, "b": m["b"], "bh": m["bh"],
                     "plateau": m["plateau"], "fit_r2": m.get("fit_r2", np.nan),
                     "notes": notes})
    return rows


def main():
    ns = load_app()
    analyse = ns["analyse_profile"]

    man = pd.read_csv(REAL / "manifest.csv")
    stem = man["file"].astype(str).str.split("_az").str[0]
    pretty = dict(zip(stem, man["volcano"]))
    kinds = dict(zip(stem, man["kind"]))

    rows = measure_all(analyse)

    # One transect per volcano for the picture: the azimuth whose e is closest
    # to that volcano's median e, so the panel shown is a typical flank and not
    # the best-fitting one. The spread across all four is printed on the panel
    # so the reader never sees a single flank as the whole answer.
    per = {}
    for r in rows:
        per.setdefault(r["key"], []).append(r)

    picked = []
    for key, rs in per.items():
        es = np.array([r["e"] for r in rs], float)
        bs = np.array([r["b"] for r in rs], float)
        med = float(np.median(es))
        chosen = min(rs, key=lambda r: abs(r["e"] - med))
        picked.append({**chosen,
                       "name": pretty.get(key, key), "kind": kinds.get(key, ""),
                       "e_med": med, "e_lo": float(es.min()), "e_hi": float(es.max()),
                       "b_med": float(np.median(bs)),
                       "b_lo": float(bs.min()), "b_hi": float(bs.max()),
                       "bh_med": float(np.median([r["bh"] for r in rs])),
                       "r2_med": float(np.median([r["e_r2"] for r in rs])),
                       "n_az": len(rs)})
    picked.sort(key=lambda p: p["e_med"])

    # ---------------- layout ----------------
    n_ref = len(REFERENCE)
    n_real = len(picked)
    real_cols = 5
    real_rows = int(np.ceil(n_real / real_cols))

    fig = plt.figure(figsize=(20.5, 4.3 + 3.9 * real_rows))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.02 * real_rows],
                                 hspace=0.0)

    uu = np.linspace(0.0, 1.0, 601)
    ss = np.concatenate([-0.5 * uu[::-1], 0.5 * uu[1:]])

    # --- reference row ---
    gs_t = top.add_gridspec(1, n_ref, left=0.035, right=0.985,
                            top=0.60, bottom=0.17, wspace=0.16)
    for i, (e, named) in enumerate(REFERENCE):
        ax = top.add_subplot(gs_t[0, i])
        yy = superellipse(np.abs(np.concatenate([uu[::-1], uu[1:]])), e)
        ax.fill_between(ss, 0, yy, color=FAINT, alpha=0.6, linewidth=0)
        ax.plot(ss, yy, color=INK, linewidth=2.0)
        ax.plot([-0.56, 0.56], [0, 0], color=INK, linewidth=1.2)
        ax.add_patch(plt.Rectangle((-0.5, 0), 1.0, 1.0, fill=False, ec=FAINT,
                                   lw=1.0, ls=(0, (2, 2))))
        label = f"e = {e:.2g}" if e != 0.4 else "e = 0.4"
        ax.set_title(label, fontsize=13, color=INK, pad=6, fontweight="bold")
        if named:
            ax.text(0.0, -0.21, named, fontsize=9.5, color=MUTED, ha="center")
        ax.set_xlim(-0.62, 0.62)
        ax.set_ylim(-0.34, 1.16)
        ax.set_aspect("equal")
        ax.axis("off")

    top.text(0.035, 0.955, "Volcano profiles against perfect shapes",
             fontsize=20, color=INK, fontweight="bold", ha="left", va="top")
    top.text(0.035, 0.845,
             "Radial SRTM transects through the summit, cropped to the measured edifice "
             "and scaled into one box, so only SHAPE is compared – not size.",
             fontsize=11.5, color=MUTED, ha="left", va="top")
    top.text(0.035, 0.705,
             "REFERENCE  —  one family of perfect shapes,  z/H = [1 − uᵉ]^(1/e)",
             fontsize=13, color=INK, fontweight="bold", ha="left", va="top")
    top.text(0.035, 0.085,
             "e < 1  concave flanks (steep summit, flaring apron)        "
             "e = 1  straight-sided cone        e > 1  convex flanks",
             fontsize=11, color=MUTED, ha="left", va="bottom")

    # --- measured panels ---
    gs_b = bottom.add_gridspec(real_rows, real_cols, left=0.035, right=0.985,
                               top=0.90, bottom=0.145, wspace=0.14, hspace=0.46)
    for j, p in enumerate(picked):
        r, c = divmod(j, real_cols)
        ax = bottom.add_subplot(gs_b[r, c])
        s, v, e = p["s"], p["v"], p["e"]
        fit = superellipse(p["u"], e)

        ax.fill_between(s, fit, v, where=(v >= fit), color=EXCESS, alpha=0.32,
                        linewidth=0, interpolate=True)
        ax.fill_between(s, fit, v, where=(v < fit), color=DEFICIT, alpha=0.32,
                        linewidth=0, interpolate=True)
        ax.plot(s, fit, color=MUTED, linewidth=1.7, ls=(0, (4, 2.5)))
        ax.plot(s, v, color=INK, linewidth=2.0)
        ax.plot([-0.56, 0.56], [0, 0], color=INK, linewidth=1.2)
        ax.add_patch(plt.Rectangle((-0.5, 0), 1.0, 1.0, fill=False, ec=FAINT,
                                   lw=1.0, ls=(0, (2, 2))))

        ax.set_title(p["name"], fontsize=13.5, color=INK, pad=7, fontweight="bold")
        ax.text(0.0, 1.10, p["kind"], fontsize=9.2, color=MUTED, ha="center")
        ax.text(0.0, -0.20,
                f"e = {p['e_med']:.2f}   (range {p['e_lo']:.2f}–{p['e_hi']:.2f} "
                f"over {p['n_az']} azimuths)",
                fontsize=10.6, color=INK, ha="center", fontweight="bold")
        ax.text(0.0, -0.30,
                f"transect shown: azimuth {p['az']}°,  fit r² = {p['e_r2']:.2f}",
                fontsize=9.4, color=MUTED, ha="center")
        ax.text(0.0, -0.40,
                f"b = {p['b']:.2f}      B/H = {p['bh']:.0f}",
                fontsize=9.4, color=MUTED, ha="center")
        ax.set_xlim(-0.62, 0.62)
        ax.set_ylim(-0.50, 1.18)
        ax.set_aspect("equal")
        ax.axis("off")

    bottom.text(0.035, 0.965,
                "MEASURED  —  real volcanoes in the same box, ordered by flank curvature",
                fontsize=13, color=INK, fontweight="bold", ha="left", va="top")
    bottom.text(0.985, 0.965,
                "perfect shape pinned apex-to-summit and base-to-base, so an off-centre "
                "summit stays off-centre",
                fontsize=11, color=MUTED, ha="right", va="top")

    handles = [
        Line2D([0], [0], color=INK, lw=2.0, label="measured profile"),
        Line2D([0], [0], color=MUTED, lw=1.7, ls=(0, (4, 2.5)),
               label="best-fitting perfect shape"),
        Patch(facecolor=EXCESS, alpha=0.32, label="ground ABOVE the perfect shape"),
        Patch(facecolor=DEFICIT, alpha=0.32, label="ground BELOW it"),
    ]
    bottom.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.035, 0.052),
                  ncol=4, frameon=False, fontsize=11, labelcolor=INK)
    bottom.text(0.035, 0.010,
                "Base, summit, height, b and B/H measured by volcanoShape.py.  "
                "Elevations: SRTM 90 m via Open Topo Data.  "
                "Comparison-to-an-ideal method after Wadell (1932, 1935) and "
                "Powers (1953).",
                fontsize=9.2, color=MUTED, ha="left", va="bottom")

    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ideal_overlay.{ext}", dpi=220, facecolor="white")
    plt.close(fig)

    # ---------------- tables ----------------
    summary = pd.DataFrame([{
        "Volcano": p["name"], "Type": p["kind"],
        "e (median)": round(p["e_med"], 2),
        "e low": round(p["e_lo"], 2), "e high": round(p["e_hi"], 2),
        "Fit r2 (median)": round(p["r2_med"], 2),
        "b (median)": round(p["b_med"], 3),
        "b low": round(p["b_lo"], 3), "b high": round(p["b_hi"], 3),
        "B/H (median)": round(p["bh_med"], 1),
        "Azimuths": p["n_az"],
        "Panel azimuth": p["az"],
    } for p in picked])
    summary.to_csv(OUT / "ideal_overlay_values.csv", index=False)

    full = pd.DataFrame([{
        "Volcano": pretty.get(r["key"], r["key"]), "Type": kinds.get(r["key"], ""),
        "Azimuth": r["az"],
        "e": round(r["e"], 2), "Fit r2": round(r["e_r2"], 3),
        "b": round(r["b"], 3), "Power-law r2": round(float(r["fit_r2"]), 3),
        "B/H": round(r["bh"], 1), "Plateau": round(r["plateau"], 3),
        "Base elev (m)": round(float(r["m"]["base_elev"]), 0),
        "Height (m)": round(float(r["m"]["height"]), 0),
        "Basal width (m)": round(float(r["m"]["x_right"] - r["m"]["x_left"]), 0),
        "Left flank share": round(r["lshare"], 2),
        "Warnings": " | ".join(r["notes"]),
    } for r in rows])
    full = full.sort_values(["Volcano", "Azimuth"]).reset_index(drop=True)
    full.to_csv(OUT / "ideal_overlay_all_azimuths.csv", index=False)

    print(f"wrote {OUT/'ideal_overlay.png'}")
    print(f"wrote {OUT/'ideal_overlay.pdf'}")
    print(f"wrote {OUT/'ideal_overlay_values.csv'}")
    print(f"wrote {OUT/'ideal_overlay_all_azimuths.csv'}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
