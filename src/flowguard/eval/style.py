"""Chart styling shared by every report figure.

Categorical slots are used in this fixed order and were checked with the palette validator
(colorblind separation and contrast on the light surface). Every figure also has a table with
the same numbers, so nothing relies on color alone.
"""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"   # reference lines (perfect calibration, kept epoch)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
