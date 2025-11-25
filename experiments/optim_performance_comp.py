import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from io import StringIO

file_path = "../results/optimization_performance/report.csv"

# ----------------------------------------------------------------------
# 1. LOAD + CLEAN CSV (remove final broken summary line)
# ----------------------------------------------------------------------

if not os.path.exists(file_path):
    print(f"Error: File not found: {file_path}")
    sys.exit(1)

if os.stat(file_path).st_size == 0:
    print(f"Error: File is empty: {file_path}")
    sys.exit(1)

with open(file_path, "r") as f:
    lines = f.readlines()

clean_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue

    # keep header
    if line.startswith("D,#R,#X,#Y"):
        clean_lines.append(line + "\n")
        continue

    # keep dataset rows (must end with .csv)
    if line.endswith(".csv"):
        clean_lines.append(line + "\n")

# load cleaned CSV
df = pd.read_csv(StringIO("".join(clean_lines)))
df.columns = df.columns.str.strip()

# ----------------------------------------------------------------------
# 2. MAPPING OF OPTIMIZERS
# ----------------------------------------------------------------------
mapping = {
    "DEHB": "SMAC",       # DEHB replaced by SMAC
    "LITE": "ACT",        # LITE replaced ACT
    "LINE": "KM++",       # LINE replaced KM++
    "RANDOM": "RAND"      # RANDOM replaced RAND
}

# ----------------------------------------------------------------------
# 3. EXTRACT THE LAST REAL DATASET ROW
# ----------------------------------------------------------------------
last_row = df.tail(1)

samples = [6, 12, 18, 24, 50, 100, 200]

def get_values(prefix):
    """Return the numeric best-percentage values for a given method prefix."""
    vals = []
    for s in samples:
        col = f"{prefix}-{s}"
        if col not in df.columns:
            print(f"Warning: Column '{col}' missing. Filling with 0.")
            vals.append(0)
            continue

        val = last_row[col].dropna()
        if val.empty:
            vals.append(0)
        else:
            # extract number before the rank letter (e.g. "14 b" → 14)
            raw = str(val.values[0]).strip()
            num = raw.split()[0]  # keep only "14"
            try:
                vals.append(int(num))
            except:
                vals.append(0)

    return vals

# ----------------------------------------------------------------------
# 4. BUILD DATA FOR PLOTTING
# ----------------------------------------------------------------------
data = {
    "Samples": samples,
    "DEHB": get_values(mapping["DEHB"]),
    "LITE": get_values(mapping["LITE"]),
    "LINE": get_values(mapping["LINE"]),
    "RANDOM": get_values(mapping["RANDOM"])
}

plot_df = pd.DataFrame(data)

# ----------------------------------------------------------------------
# 5. PLOT
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 5))
markers = ['o', 's', 'D', '^']
colors = ['royalblue', 'firebrick', 'gold', 'forestgreen']
linestyles = ['-', '--', '-.', ':']
font_size = 14

for idx, method in enumerate(["DEHB", "LITE", "LINE", "RANDOM"]):
    plt.plot(
        plot_df["Samples"], plot_df[method],
        marker=markers[idx],
        linestyle=linestyles[idx],
        color=colors[idx],
        linewidth=2.5,
        markersize=7,
        label=method
    )

plt.xscale("log")
tick_values = [6, 12, 24, 50, 100, 200]
plt.xticks(tick_values, [str(v) for v in tick_values], fontsize=font_size)

plt.xlabel("Samples", fontsize=font_size)
plt.ylabel("% Best", fontsize=font_size)
plt.ylim(0, 105)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize=font_size)

plt.tight_layout()

out_path = "../results/optimization_performance/optimization_performance_comparison.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"Saved plot to: {out_path}")
