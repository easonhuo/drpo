# Review notes

Version v13 addresses two issues from the compiled screenshot.

1. Left panel internal text was still small after minipage scaling.
   - Increased axis labels, tick labels, legend text, legend title, and reward labels.
   - Kept the same bar/line layout and the same legend-protocol encoding.

2. Right panel title overlapped with the table.
   - Changed `(b) External transfer` to the shorter `(b) Transfer`.
   - Removed negative vertical spacing after the title.
   - Added a small positive vertical gap before the table.

No image generation was used; this was produced with Matplotlib and TeX snippets only.
