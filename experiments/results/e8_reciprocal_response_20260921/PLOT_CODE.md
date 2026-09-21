# Plot code

The following Python code reconstructs `reciprocal_merged_400_9grid.svg` from `RECIPROCAL_400_CURVE_POINTS.csv`.

```python
import html
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
CSV = ROOT / "RECIPROCAL_400_CURVE_POINTS.csv"
OUT = ROOT / "reciprocal_merged_400_9grid.svg"

PRETTY = {
    "word_sorting": "Word Sorting",
    "spiral_matrix": "Spiral Matrix",
    "mini_sudoku": "Mini Sudoku",
    "maze": "Maze",
    "word_ladder": "Word Ladder",
    "knights_knaves": "Knights & Knaves",
    "graph_color": "Graph Color",
    "wikisql": "WikiSQL",
}
TASKS = [
    "word_sorting",
    "spiral_matrix",
    "mini_sudoku",
    "maze",
    "word_ladder",
    "knights_knaves",
    "graph_color",
    "wikisql",
]

df = pd.read_csv(CSV)
assert len(df) == 400
assert df[["task", "method", "lambda", "seed"]].drop_duplicates().shape[0] == 400
assert set(df["seed"].unique()) == {4000, 5000}

df["xmatch"] = df.apply(
    lambda r: r["lambda"]
    if r["method"] == "reciprocal_linear"
    else math.sqrt(r["lambda"]),
    axis=1,
)
summary = (
    df.groupby(["task", "method", "xmatch"], as_index=False)["late_window_pass8_mean"]
    .mean()
    .sort_values(["task", "method", "xmatch"])
)

W, H = 1200, 1200
pad, title_h, cols, rows = 25, 50, 3, 3
cw = (W - pad * (cols + 1)) / cols
ch = (H - title_h - pad * (rows + 1)) / rows

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<style>text{font-family:Arial,sans-serif;fill:#111}.ttl{font-size:18px;font-weight:700}.axis{font-size:10px}.legend{font-size:10px}.note{font-size:12px}</style>',
    '<text x="25" y="30" class="ttl">E8 Reciprocal curves — merged 400-cell view</text>',
]

for idx, task in enumerate(TASKS):
    r, c = divmod(idx, 3)
    x0 = pad + c * (cw + pad)
    y0 = title_h + pad + r * (ch + pad)
    ml, mt, mr, mb = 48, 34, 15, 42
    pw, ph = cw - ml - mr, ch - mt - mb

    g = summary[summary.task == task]
    xmin = max(g.xmatch.min(), 1e-6)
    xmax = g.xmatch.max()
    lx0, lx1 = math.log10(xmin), math.log10(xmax)
    ymin = max(0, g.late_window_pass8_mean.min() - 0.05)
    ymax = min(1.0, g.late_window_pass8_mean.max() + 0.03)
    if ymax - ymin < 0.08:
        ymin = max(0, ymax - 0.08)

    def X(v):
        if lx1 == lx0:
            return x0 + ml + pw / 2
        return x0 + ml + (math.log10(v) - lx0) / (lx1 - lx0) * pw

    def Y(v):
        return y0 + mt + (ymax - v) / (ymax - ymin) * ph

    parts.extend(
        [
            f'<text x="{x0 + cw / 2:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" class="ttl">{html.escape(PRETTY[task])}</text>',
            f'<line x1="{x0 + ml:.1f}" y1="{y0 + mt + ph:.1f}" x2="{x0 + ml + pw:.1f}" y2="{y0 + mt + ph:.1f}" stroke="#333"/>',
            f'<line x1="{x0 + ml:.1f}" y1="{y0 + mt:.1f}" x2="{x0 + ml:.1f}" y2="{y0 + mt + ph:.1f}" stroke="#333"/>',
        ]
    )

    for j in range(5):
        v = ymin + (ymax - ymin) * j / 4
        yy = Y(v)
        parts.extend(
            [
                f'<line x1="{x0 + ml:.1f}" y1="{yy:.1f}" x2="{x0 + ml + pw:.1f}" y2="{yy:.1f}" stroke="#ddd"/>',
                f'<text x="{x0 + ml - 5:.1f}" y="{yy + 3:.1f}" text-anchor="end" class="axis">{v:.2f}</text>',
            ]
        )

    for method, stroke in [
        ("reciprocal_linear", "#1f77b4"),
        ("reciprocal_quadratic", "#ffbf00"),
    ]:
        gg = g[g.method == method].sort_values("xmatch")
        pts = " ".join(
            f"{X(v):.1f},{Y(y):.1f}"
            for v, y in zip(gg.xmatch, gg.late_window_pass8_mean)
        )
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="2"/>'
        )
        for v, y in zip(gg.xmatch, gg.late_window_pass8_mean):
            parts.append(
                f'<circle cx="{X(v):.1f}" cy="{Y(y):.1f}" r="2.2" fill="{stroke}"/>'
            )

    parts.extend(
        [
            f'<text x="{x0 + ml + 5:.1f}" y="{y0 + mt + 12:.1f}" class="legend" fill="#1f77b4">Linear</text>',
            f'<text x="{x0 + ml + 55:.1f}" y="{y0 + mt + 12:.1f}" class="legend" fill="#b8860b">Quadratic</text>',
            f'<text x="{x0 + cw / 2:.1f}" y="{y0 + ch - 8:.1f}" text-anchor="middle" class="axis">Matched taper strength (log scale)</text>',
        ]
    )

x0 = pad + 2 * (cw + pad)
y0 = title_h + pad + 2 * (ch + pad)
notes = [
    "Merged reciprocal response curves",
    "400 cells = 128 + 104 + 108 + 60",
    "2 seeds per (task, method, λ)",
    "Linear x = λ",
    "Quadratic x = √λ",
    "Metric: late-window Pass@8",
    "Pilot / finite-horizon response evidence",
    "No convergence or universal ranking claim",
]
for i, line in enumerate(notes):
    parts.append(
        f'<text x="{x0 + 20:.1f}" y="{y0 + 35 + i * 25:.1f}" class="note">{html.escape(line)}</text>'
    )

parts.append("</svg>")
OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
print(OUT)
```

This file is deliberately Markdown rather than a new `.py` module so the evidence can include the exact reconstruction code without creating a new governed Python source path.
