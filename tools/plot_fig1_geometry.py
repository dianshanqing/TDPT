"""Draw Figure 1 in isolation: python plot_fig1_geometry.py --outdir figures/fig1.

The constants reproduce legacy_duct and the base compact_n8_w0002_tau14,
tau19 and tau23 runs. Coordinates are lattice indices. A material layer
at x=i fills the forward interval [i,i+1). In the legacy panel each of
the two accepted transverse indices has a one-pitch footprint centered
on its node. The compact detector is the closed physical acceptance square,
so its outline passes through the outer accepted nodes.
"""

import argparse
from dataclasses import dataclass
from itertools import product
from math import atan2, cos, degrees, radians, sin
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon


@dataclass(frozen=True)
class Geometry:
    title: str
    shape: tuple
    source: tuple
    layers: tuple       # Inclusive material indices on x
    detector_x: int
    roi: tuple          # Inclusive node indices on both y and z
    duct: tuple = ()    # Inclusive node indices on both y and z


AIR = Geometry("Air-duct geometry", (16, 8, 8), (1, 3, 3), (8, 9), 12, (3, 4), (3, 4))
UNIFORM = Geometry("Uniform slab", (17, 9, 9), (2, 4, 4), (4, 11), 14, (3, 5))
INK, MUTED, FRAME = "#282A29", "#575B58", "#C6C8C3"
DUCT, DETECTOR = "#536971", "#865F63"
SLAB_TOP, SLAB_FRONT, SLAB_SIDE = "#FCFBF8", "#E8E5DE", "#D0CCC4"
ELEV, AZIM = map(radians, (27, -72))


def project(p):
    """Equal-scale orthographic projection; axes follow Matplotlib's camera convention."""
    x, y, z = p
    return (-sin(AZIM)*x + cos(AZIM)*y,
            cos(ELEV)*z - sin(ELEV)*(cos(AZIM)*x + sin(AZIM)*y))


def line(ax, points, **style):
    xy = list(map(project, points))
    return ax.plot(*zip(*xy), solid_capstyle="round", **style)[0]


def face(ax, points, color, edge="none", width=0.48, **style):
    patch = Polygon([project(p) for p in points], closed=True, facecolor=color,
                    edgecolor=edge, linewidth=width, joinstyle="round", **style)
    return ax.add_patch(patch)


def yz_square(x, y0, y1, z0, z1):
    return [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]


def outline(ax, points, **style):
    return line(ax, [*points, points[0]], **style)


def domain(ax, g):
    X, Y, Z = (n - 1 for n in g.shape)
    # Light edges bound the lattice. The three indexed axes are emphasized.
    for corner in product((0, X), (0, Y), (0, Z)):
        for dim, upper in enumerate((X, Y, Z)):
            if corner[dim] == 0:
                end = list(corner)
                end[dim] = upper
                line(ax, [corner, end], color=FRAME, lw=0.32, zorder=0)

    for axis, upper, start, offset, text_offset, ha in (
        (0, X, (0, 0, 0), (0, -0.23), (0, -0.36), "center"),
        (1, Y, (X, 0, 0), (0.22, -0.10), (0.38, -0.15), "left"),
        (2, Z, (0, 0, 0), (-0.20, 0), (-0.32, 0), "right"),
    ):
        end = list(start)
        end[axis] = upper
        line(ax, [start, end], color=MUTED, lw=0.65, zorder=1)
        # A single zero labels the common x/z origin; the y origin is at x=X.
        ticks = (0, 4, 8, 12, upper) if axis == 0 else (4, upper)
        for t in ticks:
            pos = list(start)
            pos[axis] = t
            u, v = project(pos)
            ax.plot([u, u+offset[0]], [v, v+offset[1]], color=MUTED, lw=0.5)
            ax.text(u+text_offset[0], v+text_offset[1], str(t), ha=ha,
                    va="top" if axis < 2 else "center", color=MUTED, fontsize=7.2)
        mid = list(start)
        mid[axis] = (6, 0.78*Y, 0.7*Z)[axis]
        u, v = project(mid)
        dx, dy = ((-0.15, -1.17), (1.44, -0.36), (-0.82, 0.1))[axis]
        ax.text(u+dx, v+dy, "xyz"[axis], ha="center", va="center",
                fontsize=8, fontstyle="italic", color=INK)


def slab(ax, g):
    x0, last = g.layers
    x1 = last + 1
    Y, Z = g.shape[1]-1, g.shape[2]-1
    top = [(x0, 0, Z), (x1, 0, Z), (x1, Y, Z), (x0, Y, Z)]
    side = [(x0, 0, 0), (x1, 0, 0), (x1, 0, Z), (x0, 0, Z)]
    face(ax, top, SLAB_TOP, zorder=2)
    face(ax, side, SLAB_SIDE, zorder=2)
    for x in range(x0+1, x1):
        line(ax, [(x, 0, 0), (x, 0, Z), (x, Y, Z)], color="#96938B", lw=0.43, zorder=3)

    front = yz_square(x1, 0, Y, 0, Z)
    silhouette = [(x0, 0, 0), (x1, 0, 0), (x1, Y, 0),
                  (x1, Y, Z), (x0, Y, Z), (x0, 0, Z)]
    outline(ax, silhouette, color=MUTED, lw=1.45, zorder=4)
    for corner in ((x0, 0, Z), (x1, Y, Z), (x1, 0, 0)):
        line(ax, [(x1, 0, Z), corner], color=MUTED, lw=0.88, zorder=4)
    if not g.duct:
        face(ax, front, SLAB_FRONT, zorder=3)
        return

    # Four pieces surround an actual open front face. Clip the far rim and
    # inner walls to the near opening so hidden edges cannot cross the slab.
    lo, hi = g.duct[0]-0.5, g.duct[1]+0.5
    for bounds in ((0, lo, 0, Z), (hi, Y, 0, Z), (lo, hi, 0, lo), (lo, hi, hi, Z)):
        face(ax, yz_square(x1, *bounds), SLAB_FRONT, zorder=3)
    near = yz_square(x1, lo, hi, lo, hi)
    far = yz_square(x0, lo, hi, lo, hi)
    # Dashed hidden edges locate the upstream mouth behind the slab.
    hidden = dict(color="#778488", lw=0.48, ls="--", dash_capstyle="butt", zorder=4)
    outline(ax, far, **hidden)
    for i in (0, 2):
        line(ax, [far[i], near[i]], **hidden)
    aperture = face(ax, near, "white", zorder=4)
    for i, j, shade in ((0, 1, "#E9EEEC"), (1, 2, "#C6D0D0")):
        face(ax, [near[i], near[j], far[j], far[i]], shade, DUCT, 0.40,
             clip_path=aperture, zorder=5)
    outline(ax, near, color=DUCT, lw=1.5, zorder=6)


def detector(ax, g):
    """Project the acceptance plane and its actual lattice nodes."""
    lo, hi = g.roi
    margin = 0.5 if g.duct else 0
    a, b = lo-margin, hi+margin

    def coords(y, z):
        return project((g.detector_x, y, z))

    window = [coords(y, z) for y, z in ((a, a), (b, a), (b, b), (a, b))]
    ax.add_patch(Polygon(window, closed=True, facecolor="#F2E9E6", edgecolor=DETECTOR,
                         linewidth=0.85, joinstyle="round", zorder=7))
    nodes = [coords(y, z)
             for y, z in product(range(lo, hi+1), repeat=2)]
    ax.plot(*zip(*nodes), ls="none", marker="o", ms=2.7, mfc=DETECTOR, mec="white",
            mew=0.30, zorder=8)
    limits = (rf"$y,z\in\{{{lo},{hi}\}}$" if g.duct else rf"${lo}\leq y,z\leq {hi}$")
    label = f"Detector, $x={g.detector_x}$\n{limits}"
    offset = (4, 17) if g.duct else (-9, 18)
    callout(ax, label, coords(b, b), offset, color=DETECTOR)


def callout(ax, text, point, offset, color=INK, align="left", va="bottom"):
    return ax.annotate(text, point, xytext=offset, textcoords="offset points",
                       ha=align, va=va, fontsize=7.6, linespacing=1.12,
                       color=color, zorder=11,
                       bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
                       arrowprops=dict(arrowstyle="-", color=color, lw=0.55,
                                       shrinkA=2, shrinkB=2.5))


def mark_source(ax, g):
    x, y, z = g.source
    start = project(g.source)
    # Limit the arrow to the projected free space before the slab silhouette.
    left_edge = project((g.layers[0], 0, z))[0]
    length = min(2.5, (left_edge-start[0]) / -sin(AZIM) - 0.08)
    tip = project((x+length, y, z))
    ax.annotate("", xy=tip, xytext=start,
                arrowprops=dict(arrowstyle="-|>", mutation_scale=6.5, lw=0.95,
                                color="black", shrinkA=0, shrinkB=0), zorder=9)
    ax.plot(*start, "o", ms=4.1, color="black", mec="white", mew=0.45, zorder=10)
    midpoint = ((start[0]+tip[0])/2, (start[1]+tip[1])/2)
    ax.annotate("$+x$", midpoint, xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=7.0, color=INK, zorder=10)
    callout(ax, f"Source\n{g.source}", start, (3, -10) if g.duct else (-2, -11),
            align="left" if g.duct else "right", va="top")


def dimension(ax, g):
    x0, last = g.layers
    x1, y, z = last+1, 0, 2.3
    start, end = project((x0, y, z)), project((x1, y, z))
    ax.annotate("", xy=end, xytext=start, zorder=6,
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.68,
                                mutation_scale=6.0, shrinkA=0.8, shrinkB=0.8))
    mid = project(((x0+x1)/2, y, z))
    u, v = project((1, 0, 0))
    ax.text(*mid, f"{x1-x0} grid spacings", ha="center", va="center",
            rotation=degrees(atan2(v, u)), fontsize=7.6, color=INK, zorder=7,
            bbox=dict(facecolor=SLAB_SIDE, edgecolor="none", pad=1.0))


def panel(fig, g, position, count):
    ax = fig.add_axes([position/count, 0.014, 1/count, 0.92])
    ax.set_aspect("equal")
    ax.set_axis_off()
    # Leave only enough space for the projected axes and their direct labels.
    u0 = -1.85 if g.duct else -1.45
    v0 = -4.1 if g.duct else -3.55
    ax.set_xlim(u0, u0+20.0)
    ax.set_ylim(v0, v0+14.45)
    domain(ax, g)
    slab(ax, g)
    detector(ax, g)
    mark_source(ax, g)
    if g.duct:
        target = project((g.layers[1]+1, g.duct[0]-0.5, g.duct[1]+0.5))
        callout(ax, "Air duct", target, (-33, 19), color=DUCT, align="right")
    else:
        dimension(ax, g)

    letter = "a" if g.duct else "b"
    title = g.title if g.duct else f"{g.title} in vacuum"
    shape = " × ".join(map(str, g.shape))
    fig.text((position+0.018)/count, 0.988, f"({letter})  {title}  ({shape})",
             va="top", color=INK, size=8.8)


def generate(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    layouts = (("fig1a_air_duct", (AIR,)), ("fig1b_uniform_slab", (UNIFORM,)),
               ("fig1_geometry", (AIR, UNIFORM)))
    # Arial is used for both lettering and variables. DejaVu Sans ships with Matplotlib.
    font = FontProperties(family=["Arial", "Liberation Sans", "DejaVu Sans"]).get_name()
    with mpl.rc_context({"font.family": font, "mathtext.fontset": "custom",
                         "mathtext.rm": font, "mathtext.sf": font, "mathtext.cal": font,
                         "mathtext.fallback": "stixsans",
                         "mathtext.it": f"{font}:italic", "mathtext.bf": f"{font}:bold",
                         "text.usetex": False, "pdf.fonttype": 42, "savefig.facecolor": "white"}):
        for stem, geometries in layouts:
            fig = Figure(figsize=(85/25.4*len(geometries), 2.5), facecolor="white")
            FigureCanvasAgg(fig)
            for i, g in enumerate(geometries):
                panel(fig, g, i, len(geometries))
            for ext in ("pdf", "png"):
                path = outdir / f"{stem}.{ext}"
                fig.savefig(path, dpi=600)
                print(path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the two benchmark geometries and Figure 1.")
    parser.add_argument("--outdir", type=Path, default=Path("figures/fig1"),
                        help="output directory (default: figures/fig1 under the working directory)")
    generate(parser.parse_args().outdir)
