#!/usr/bin/env python3
"""Compare regenerated figure rasters against manuscript reference rasters where useful.

CONTRACT
--------
Inputs (read-only):
    figures/Figure_{1,2,S1}.png            written by scripts/12..14
    figures/Figure_S2_selectivity.png      written by scripts/15
    docs/published_figures/Figure_{1,2,S1,S2}.png
Output:
    data/derived/figure_tables/figure_pixel_comparison.csv
        figure, published_px, regenerated_px, dimensions_match, mean_abs_pixel_diff,
        pct_pixels_within_8, verdict

WHAT THE NUMBER MEANS - AND DOES NOT
    Mean absolute pixel difference on a common grid is a WEAK similarity measure for
    line art.  A one-pixel horizontal shift of a 2 px black curve on white produces a
    large per-pixel difference while being visually identical, so a figure that is
    correct in every scientific respect can score badly.  A LOW score is strong
    evidence of a match; a high score is a prompt to look at the figure, not a verdict
    that it is wrong.  Every number here is reported with that caveat attached, and the
    per-figure diagnosis in docs/FIGURE_TABLES.md is what should be read alongside.
    Figures without a same-layout reference target are reported as not comparable
    instead of being assigned a misleading pixel-distance score.

METHOD
    Both images are converted to greyscale, the regenerated one is resampled to the
    published one's dimensions (Lanczos), and the mean absolute difference is taken over
    all pixels on a 0-255 scale.  `pct_pixels_within_8` is the fraction of pixels
    agreeing to within 8 grey levels (~3%), which is less sensitive to sub-pixel
    placement than the mean and so tracks "same content" more closely.

INVARIANTS
    * Read-only: this script writes one CSV and no images.
    * Figure 1 is generated from deposited source data, but no same-layout reference
      raster is used for pixel scoring.
    * Figure S2 is compared against a different manuscript figure by design and is
      reported as not comparable rather than scored - see scripts/15_figure_S2.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PUBLISHED = REPO / "docs" / "published_figures"
FIGURES = REPO / "figures"
OUT = REPO / "data" / "derived" / "figure_tables" / "figure_pixel_comparison.csv"

# regenerated file -> published file.  A None target means the generated image is not a
# same-layout rebuild of the manuscript reference image.
PAIRS = [
    (
        "Figure_1.png",
        None,
        "manuscript layout generated from deposited Figure_1_source_data.csv; "
        "no same-layout reference raster is used for pixel scoring",
    ),
    ("Figure_2.png", "Figure_2.png", ""),
    ("Figure_S1.png", "Figure_S1.png", ""),
    (
        "Figure_S2_selectivity.png",
        None,
        "manuscript Figure_S2.png is an IC50 replicate-concordance scatter; this script "
        "generates the distinct pSTAT1-minus-pSTAT3 selectivity atlas",
    ),
]
WITHIN = 8          # grey levels; ~3% of the 0-255 range
GOOD_MAD = 12.0     # below this, "close match" on line art at these rasters
FAIR_MAD = 30.0


def compare(regen: Path, pub: Path) -> dict:
    a = Image.open(pub).convert("L")
    b = Image.open(regen).convert("L")
    same = (a.size == b.size)
    if not same:
        b = b.resize(a.size, Image.LANCZOS)
    x = np.asarray(a, dtype=np.int16)
    y = np.asarray(b, dtype=np.int16)
    d = np.abs(x - y)
    mad = float(d.mean())
    within = float((d <= WITHIN).mean())
    verdict = ("close" if mad < GOOD_MAD else
               "moderate" if mad < FAIR_MAD else "differs")
    return {"published_px": f"{a.size[0]}x{a.size[1]}",
            "regenerated_px": f"{Image.open(regen).size[0]}x"
                              f"{Image.open(regen).size[1]}",
            "dimensions_match": same,
            "mean_abs_pixel_diff": round(mad, 2),
            "pct_pixels_within_8": round(100.0 * within, 2),
            "verdict": verdict}


def main() -> int:
    rows = []
    for regen_name, pub_name, note in PAIRS:
        regen = FIGURES / regen_name
        if not regen.exists():
            rows.append({"figure": regen_name, "verdict": "not rendered",
                         "note": "run scripts/12..15 first"})
            continue
        if pub_name is None:
            size = Image.open(regen).size
            reference_name = "Figure_S2.png" if "S2" in regen_name else regen_name
            reference = PUBLISHED / reference_name
            if reference.exists():
                ref_size = Image.open(reference).size
                published_px = f"{ref_size[0]}x{ref_size[1]}"
            else:
                published_px = ""
            rows.append({
                "figure": regen_name,
                "published_px": published_px,
                "regenerated_px": f"{size[0]}x{size[1]}",
                "dimensions_match": False,
                "mean_abs_pixel_diff": np.nan,
                "pct_pixels_within_8": np.nan,
                "verdict": "not comparable",
                "note": note})
            continue
        res = compare(regen, PUBLISHED / pub_name)
        rows.append({"figure": regen_name, **res, "note": ""})

    df = pd.DataFrame(rows)
    cols = ["figure", "published_px", "regenerated_px", "dimensions_match",
            "mean_abs_pixel_diff", "pct_pixels_within_8", "verdict", "note"]
    df = df.reindex(columns=cols)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.drop(columns=["note"]).to_string(index=False))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    print("mean abs pixel diff is a WEAK measure on line art: a sub-pixel shift of a "
          "thin curve scores high while being visually identical.\nRead it with the "
          "per-figure diagnosis in docs/FIGURE_TABLES.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
