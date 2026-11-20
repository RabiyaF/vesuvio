import glob
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from mantid.simpleapi import LoadAscii

# ------------------- user settings -------------------
BASE_DIR = r"/your/path/to/output/files/directory"  # <-- change this
GLOB_PATTERN = "*profiles_sum.txt"  # Either '*Workspace' (for y-space fit) or '*profiles_sum.txt' (for ncp fits)
X_LABEL = "Energy (eV)"
Y_LABEL = "Intensity (arb. units)"
FONT_FAMILY = "serif"
FONT_FACE = "Arial"
AXES_LABEL_SIZE_PT = 18
TICK_LABEL_SIZE_PT = 16
LINE_WIDTH_PT = 2
ERROR_LINE_WIDTH_PT = 1.5
LEGEND = True
LEGEND_FRAME_ON = False
LEGEND_LOC = "best"  # 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
MARKER_SIZE_PT = 4
TICK_MAJOR_SIZE = 5
TICK_MINOR_SIZE = 3
AXES_GRID = True
MATPLOTLIB_STYLE = "ggplot"  # See all of the options at https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
# -----------------------------------------------------

plt.style.use(MATPLOTLIB_STYLE)

# The following will override style
mpl.rcParams["lines.linewidth"] = LINE_WIDTH_PT
mpl.rcParams["axes.labelsize"] = AXES_LABEL_SIZE_PT
mpl.rcParams["xtick.labelsize"] = TICK_LABEL_SIZE_PT
mpl.rcParams["ytick.labelsize"] = TICK_LABEL_SIZE_PT
mpl.rcParams["lines.markersize"] = MARKER_SIZE_PT
mpl.rcParams["legend.frameon"] = LEGEND_FRAME_ON
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.sans-serif"] = FONT_FACE
plt.rcParams["axes.grid"] = AXES_GRID
plt.rcParams["xtick.major.size"] = TICK_MAJOR_SIZE
plt.rcParams["xtick.minor.size"] = TICK_MINOR_SIZE
plt.rcParams["ytick.major.size"] = TICK_MAJOR_SIZE
plt.rcParams["ytick.minor.size"] = TICK_MINOR_SIZE

# If you need further customisation, look at other options on https://matplotlib.org/stable/users/explain/customizing.html

file_names = sorted(glob.glob(os.path.join(BASE_DIR, GLOB_PATTERN)))
if not file_names:
    raise RuntimeError(f"No files match {GLOB_PATTERN} in {BASE_DIR}")

for file_name in file_names:
    # Set default for legend at every iteration
    do_legend = LEGEND

    # Read legend from first lines of file
    # Expects file format from ascii produced by Mantid
    labels = []
    with open(file_name, "r") as file:
        line = file.readline()
        # If custom header is present containing the legend info
        if not line.startswith("#"):
            header = ""
            while line.strip() != "":
                header += line
                line = file.readline()
            labels = [label.strip() for label in header.split(",Not defined,")]
            labels.pop(-1)
            print("Found the following labels in file:\n", labels)

    ws_name = Path(file_name).stem
    ws = LoadAscii(file_name, OutputWorkspace=ws_name)

    if len(labels) != ws.getNumberHistograms():
        print("Number of labels does not match number of rows, disabling legend!")
        labels = ["" for _ in range(ws.getNumberHistograms())]
        do_legend = False

    fig, ax = plt.subplots(subplot_kw={"projection": "mantid"}, layout="constrained")

    # ------------------ user edit ----------------------
    # Format for y-space fits, uncomment line
    # Set line format to black dots for first row (data) and blue dots for last row (residuals)
    # line_formats = ['.k'] + ['-'] * (ws.getNumberHistograms() - 2) + ['.b']
    # If you want to leave the residuals out, you can also do this:
    # line_formats = ['.k'] + ['-'] * (ws.getNumberHistograms() - 2)

    # Format for sum of ncp fits, uncomment line
    # Set first line to black dots for first row and solid lines for remaining rows
    line_formats = [".k"] + ["-"] * (ws.getNumberHistograms() - 1)

    # For more information on line formats, consult the notes section of the page
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    # Can also edit labels if needed
    # Should match the number of rows in workspace
    # Eg. for a y-space fit workspace:
    # labels = ["data", "fit", "residuals"]
    # ---------------------------------------------------

    # Plotting lines with the formats specified above
    for i, fmt in enumerate(line_formats):
        ax.errorbar(ws.dataX(i), ws.dataY(i), ws.dataE(i), label=labels[i], fmt=fmt, elinewidth=ERROR_LINE_WIDTH_PT)

    # Set user defined labels
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    # Trigger legend
    if do_legend:
        ax.legend(loc=LEGEND_LOC)

    fig_path = os.path.join(BASE_DIR, ws_name + ".pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure {fig_path}")
