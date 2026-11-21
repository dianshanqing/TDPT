"""Check Figure 1 against the archived models and redraw it in a fresh directory."""
import argparse
import json
from pathlib import Path
import re
import sys

from matplotlib import __version__ as matplotlib_version
from PIL import Image

import plot_fig1_geometry as plot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import material_at
from models import load_model, model_from_dict


def check_model(selector, record, panel):
    cfg = load_model(ROOT / selector)
    with (ROOT / record).open(encoding="utf-8") as stream:
        recorded = model_from_dict(json.load(stream)["config"])
    if cfg != recorded:
        raise ValueError(f"Resolved input differs from archived configuration: {selector}")
    g, src = cfg.geom, cfg.source
    actual = ((g.nx, g.ny, g.nz), (src.x0, src.y0, src.z0), src.dir0,
              (g.slab_x0, g.slab_x1), g.detector_x,
              (g.roi_y0, g.roi_y1), (g.roi_z0, g.roi_z1))
    expected = (panel.shape, panel.source, 0, panel.layers, panel.detector_x,
                panel.roi, panel.roi)
    if actual != expected:
        raise ValueError(f"Drawing differs from {selector}: {actual} != {expected}")
    if not panel.duct and any(cfg.mats.concrete.p_int):
        raise ValueError(f"Expected a vacuum outside the slab in {selector}")
    for x in range(g.nx):
        for y in range(g.ny):
            for z in range(g.nz):
                material = "concrete"
                if panel.layers[0] <= x <= panel.layers[1]:
                    in_duct = (panel.duct and panel.duct[0] <= y <= panel.duct[1]
                               and panel.duct[0] <= z <= panel.duct[1])
                    material = "air" if in_duct else "tungsten"
                if material_at(cfg, x, y, z) != material:
                    raise ValueError(f"Material layout differs at {(x, y, z)} in {selector}")
    print(f"Model, drawing and recorded input agree: {selector}")


def compare_exports(outdir):
    archived = ROOT / "figures/fig1"
    exact = True
    for stem in ("fig1a_air_duct", "fig1b_uniform_slab", "fig1_geometry"):
        png = stem + ".png"
        with Image.open(archived / png) as ref, Image.open(outdir / png) as new:
            if ref.size != new.size or ref.info.get("dpi") != new.info.get("dpi"):
                raise ValueError(f"PNG dimensions or resolution differ: {png}")
            pixels_match = ref.convert("RGBA").tobytes() == new.convert("RGBA").tobytes()
        pdf = stem + ".pdf"
        original = (archived / pdf).read_bytes()
        rendered = (outdir / pdf).read_bytes()
        # Matplotlib writes one MediaBox per page and an export timestamp in the info dictionary.
        boxes = rb"/MediaBox\s*\[([^\]]+)\]"
        ref_boxes = re.findall(boxes, original)
        if len(ref_boxes) != 1 or ref_boxes != re.findall(boxes, rendered):
            raise ValueError(f"PDF page count or size differs: {pdf}")
        timestamp = rb"/CreationDate\s*\(D:[^)]*\)"
        pdf_match = re.sub(timestamp, b"", original) == re.sub(timestamp, b"", rendered)
        print(f"{stem}: PNG pixels {'match' if pixels_match else 'differ'}, "
              f"PDF bytes excluding CreationDate {'match' if pdf_match else 'differ'}")
        exact = exact and pixels_match and pdf_match
    return exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/user_runs/geometry_001"),
                        help="New directory under outputs/user_runs, relative to the project root")
    parser.add_argument("--require-exact", action="store_true",
                        help="Exit with an error on pixel or normalized PDF byte differences")
    args = parser.parse_args()
    outdir = (ROOT / args.outdir).resolve()
    if ROOT / "outputs/user_runs" not in outdir.parents:
        parser.error("Choose a new directory under outputs/user_runs.")
    if outdir.exists():
        parser.error(f"Output already exists. Choose a new directory: {outdir}")

    check_model("configs/models/legacy_duct.json",
                "outputs/air_duct/classical_reference/legacy_duct.json", plot.AIR)
    for tau in (14, 19, 23):
        filename = f"compact_n8_w0002_tau{tau}.json"
        check_model(f"configs/models/compact_deep/{filename}",
                    f"outputs/deep_penetration/tau{tau}/classical/{filename}", plot.UNIFORM)
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"Drawing with Matplotlib {matplotlib_version} in {outdir}")
    plot.generate(outdir)
    if compare_exports(outdir):
        print("All three PNG pixel arrays and normalized PDFs match.")
    elif args.require_exact:
        parser.exit(1, "Exact rendering comparison failed. Check Matplotlib and font versions.\n")
    else:
        print("Model parameters, PNG dimensions and PDF page sizes match. "
              "Rendering differs from the archived export; inspect both images.")


if __name__ == "__main__":
    main()
