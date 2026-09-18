"""Render Figure 1 under a range of palettes so one can be picked by eye.

Each entry is (name, heads+MLP dark, heads+MLP light, neuron dark, neuron light).
The dark tone carries EAP-IG and the light tone RelP, so a palette has to keep
both pairs legible on white and keep the two hues far apart.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "circuit_reuse_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from analysis import figure1, granularity_parity as gp

PALETTES = [
    ("tab20 blue/red (current)", "#1f77b4", "#aec7e8", "#d62728", "#ff9896"),
    ("tab10 blue/orange", "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"),
    ("Okabe-Ito blue/vermillion", "#0072B2", "#7FC1E3", "#D55E00", "#F0A882"),
    ("IBM blue/magenta", "#648FFF", "#A9BFFF", "#DC267F", "#F09CC1"),
    ("Wong blue/orange", "#0173B2", "#7FBEE0", "#DE8F05", "#F0C266"),
    ("ColorBrewer Set1", "#377EB8", "#9CC0DE", "#E41A1C", "#F29B9C"),
    ("deep navy/vivid red", "#00306E", "#5C7FBF", "#E8231F", "#F4867F"),
    ("navy/gold", "#00306E", "#5C7FBF", "#D4A215", "#EBC85A"),
    ("seaborn deep", "#4C72B0", "#A3B9DA", "#C44E52", "#E2A6A8"),
    ("teal/magenta", "#009E9E", "#7FCFCF", "#D81B60", "#EC8DAF"),
    ("indigo/amber", "#3F51B5", "#9FA8DA", "#FF8F00", "#FFC77F"),
    ("slate/coral", "#2F4B7C", "#97A5BD", "#F95D6A", "#FCAEB4"),
    ("NPG cyan/red", "#4DBBD5", "#A6DDEA", "#E64B35", "#F2A59A"),
    ("purple/green", "#7B3294", "#BD98C9", "#008837", "#7FC39B"),
    ("Dark2 teal/orange", "#1B9E77", "#8DCFBB", "#D95F02", "#ECAF80"),
    ("steel blue/crimson", "#1B4F9C", "#8DA7CE", "#C1121F", "#E08890"),
]


def apply(hm, hm_light, nu, nu_light):
    """Rebind the palette everywhere the figure reads it from."""
    gp.GRAN_COLORS = {"head_mlp": hm, "neuron": nu}
    gp.GRAN_COLORS_LIGHT = {"head_mlp": hm_light, "neuron": nu_light}
    gp.CONFIG_COLORS = {(m, g): (gp.GRAN_COLORS if m == "eap_ig"
                                 else gp.GRAN_COLORS_LIGHT)[g]
                        for m, g in gp.CONFIG_ORDER}
    # figure1 bound these names at import time, so patch its namespace too.
    figure1.GRAN_COLORS = gp.GRAN_COLORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="results/granularity_parity_analysis")
    args = ap.parse_args()
    analysis = Path(args.analysis_dir)
    out = analysis / "palettes"
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# Figure 1 palette options", "",
             "Same figure, same data, one palette each. Dark tone is EAP-IG and",
             "light tone is RelP within each granularity.", ""]
    for i, (name, *colors) in enumerate(PALETTES, start=1):
        apply(*colors)
        slug = f"p{i:02d}"
        figure1.make(analysis, out / f"{slug}.png", figure1.K if hasattr(figure1, "K") else 10, 50)
        swatch = "  ".join(f"`{c}`" for c in colors)
        lines += [f"## Option {i}: {name}", "", swatch, "",
                  f"![](palettes/{slug}.png)", "", "---", ""]
    (analysis / "PALETTES.md").write_text("\n".join(lines))
    print(f"wrote {analysis / 'PALETTES.md'}")


if __name__ == "__main__":
    main()
