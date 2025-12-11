import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys

# ===========================================
# File path
# ===========================================
file_path = "../results/optimization_performance/report_tmp.csv"

if not os.path.exists(file_path):
    print(f"Error: File not found: {file_path}")
    sys.exit(1)

if os.stat(file_path).st_size == 0:
    print(f"Error: File is empty: {file_path}")
    sys.exit(1)

try:
    df = pd.read_csv(file_path, engine="python", sep=r"\s*,\s*")
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)
print(df)
# Ensure File column exists
if "File" not in df.columns:
    print("Error: 'File' column missing.")
    sys.exit(1)

# Clean column names
df.columns = df.columns.str.strip().str.replace('"', '').str.replace(' ', '')

# ===========================================
# Identify all runtime columns automatically
# method = EVERYTHING before "_time"
# ===========================================
time_cols = [c for c in df.columns if c.endswith("_time")]

method_groups = {}
for col in time_cols:
    method = col[:-5]  # remove "_time"
    method_groups[method] = [col]

print("Detected methods:", list(method_groups.keys()))

# ===========================================
# Output directory
# ===========================================
output_dir = "../results/runtime_plot"
os.makedirs(output_dir, exist_ok=True)

# ===========================================
# Compute per-dataset averages
# ===========================================
combined_data = []
print(df)
for file_name in df["File"].unique():

    subset = df[df["File"] == file_name]
    entry = {"File": file_name}

    for method, cols in method_groups.items():

        try:
            entry[method] = subset[cols].mean(axis=1).values[0]
        except Exception:
            entry[method] = None

    combined_data.append(entry)

# Build final table
combined_df = pd.DataFrame(combined_data)

# ===========================================
# Clean dataset names
# ===========================================
def clean_label(label):
    if not isinstance(label, str):
        return label
    label = label.replace(".csv", "")
    label = label.replace("healthCloseIsses12mths0011-easy", "Health-easy")
    label = label.replace("healthCloseIsses12mths0001-hard", "Health-hard")
    return label

combined_df["File"] = combined_df["File"].apply(clean_label)

# ===========================================
# Choose a default sort method
# Use the first detected method
# ===========================================
first_method = list(method_groups.keys())[0]
combined_df_sorted = combined_df.sort_values(by=first_method)

# ===========================================
# Plotting (same style as before)
# ===========================================
plt.figure(figsize=(9, 6))
x = combined_df_sorted["File"]

# Simple style for now
colors = plt.cm.tab10.colors
markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']

styles = {}
method_list = list(method_groups.keys())

for i, method in enumerate(method_list):
    styles[method] = (
        markers[i % len(markers)],
        "-",
        colors[i % len(colors)]
    )

# Plot each method
for method in method_list:
    if method in combined_df_sorted.columns:
        marker, linestyle, color = styles[method]
        plt.plot(
            x,
            combined_df_sorted[method],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=method,
        )

plt.xticks(rotation=45, ha="right")
plt.yscale("log")
plt.ylabel("Avg. Runtime (log scale)", fontsize=12)
plt.xlabel("Dataset", fontsize=12)
plt.yticks(fontsize=11)

# Legend
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=min(5, len(method_list)),
    frameon=False,
    fontsize=12
)

plt.tight_layout()

save_path = os.path.join(output_dir, "all_datasets_performance_comparison.png")
plt.savefig(save_path)
plt.close()

print("Plot saved to:", save_path)
