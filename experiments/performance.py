import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys

# ============================
# File path
# ============================
file_path = "../results/optimization_performance/report_tmp.csv"

if not os.path.exists(file_path):
    print(f"Error: File not found: {file_path}")
    sys.exit(1)

if os.stat(file_path).st_size == 0:
    print(f"Error: File is empty: {file_path}")
    sys.exit(1)

df = pd.read_csv(file_path)

if "File" not in df.columns:
    print("Error: 'File' column missing.")
    sys.exit(1)

# Clean header names
df.columns = df.columns.str.strip().str.replace('"', '').str.replace(' ', '')

# Time columns
time_cols = [c for c in df.columns if c.endswith("_time")]

# ============================
# Regex-based groups
# ============================
smac_group   = [c for c in time_cols if re.match(r"SMAC-\d+", c)]
lite_group   = [c for c in time_cols if re.match(r"ACT-\d+", c)]   # renamed ACT → LITE
line_group   = [c for c in time_cols if re.match(r"KM\+\+-\d+", c)]  # renamed KM++ → LINE
random_group = [c for c in time_cols if re.match(r"RAND-\d+", c)]

# ============================
# Output directory
# ============================
output_dir = "../results/runtime_plot"
os.makedirs(output_dir, exist_ok=True)

combined_data = []

# ============================
# Compute per-dataset averages
# ============================
for file_name in df["File"].unique():
    subset = df[df["File"] == file_name]

    avg_smac   = subset[smac_group].mean(axis=1).values[0]   if smac_group else None
    avg_lite   = subset[lite_group].mean(axis=1).values[0]   if lite_group else None
    avg_line   = subset[line_group].mean(axis=1).values[0]   if line_group else None
    avg_random = subset[random_group].mean(axis=1).values[0] if random_group else None

    entry = {"File": file_name}

    if avg_smac   is not None: entry["SMAC"] = avg_smac
    if avg_lite   is not None: entry["LITE"] = avg_lite
    if avg_line   is not None: entry["LINE"] = avg_line
    if avg_random is not None: entry["RANDOM"] = avg_random

    combined_data.append(entry)

# ============================
# Clean labels
# ============================
def clean_label(label):
    label = label.replace(".csv", "")
    label = label.replace("healthCloseIsses12mths0011-easy", "Health-easy")
    label = label.replace("healthCloseIsses12mths0001-hard", "Health-hard")
    return label

combined_df = pd.DataFrame(combined_data)

# Sort by LINE or fallback
sort_key = "LINE" if "LINE" in combined_df.columns else "SMAC"
combined_df_sorted = combined_df.sort_values(by=sort_key)

# Clean file names
combined_df_sorted["File"] = combined_df_sorted["File"].apply(clean_label)

# ============================
# Generate dataset index mapping file
# ============================
mapping_path = os.path.join(output_dir, "dataset_index_mapping.txt")
with open(mapping_path, "w") as f:
    for idx, name in enumerate(combined_df_sorted["File"], start=1):
        f.write(f"{idx}: {name}\n")

print(f"Dataset index mapping saved to: {mapping_path}")

# ============================
# Plot (index-based X-axis for 120 datasets)
# ============================
plt.figure(figsize=(14, 7))

x = list(range(1, len(combined_df_sorted) + 1))  # dataset indices

styles = {
    "SMAC":   ('o', '-',  'tab:blue'),
    "LINE":   ('s', '--', 'tab:green'),
    "RANDOM": ('^', '-.', 'tab:red'),
    "LITE":   ('h', '-.', 'tab:pink'),
}

for method in ["SMAC", "LINE", "LITE", "RANDOM"]:
    if method in combined_df_sorted.columns:
        marker, linestyle, color = styles[method]
        plt.plot(
            x,
            combined_df_sorted[method],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=method
        )

# No dataset names on x-axis (120–150 items unreadable)
plt.xticks([])

plt.xlabel(f"Datasets (1 to {len(x)}) — see dataset_index_mapping.txt for names",
           fontsize=12)
plt.ylabel("Avg. Runtime (log scale)", fontsize=12)
plt.yscale("log")

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.10),
    ncol=4,
    frameon=False,
    fontsize=14
)

plt.tight_layout()
save_path = os.path.join(output_dir, "all_datasets_performance_comparison.png")
plt.savefig(save_path)
plt.close()

print(f"Plot saved to: {save_path}")
