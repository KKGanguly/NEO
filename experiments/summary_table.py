import pandas as pd
import numpy as np
import re

# Load data
df = pd.read_csv("../results/optimization_performance/report.csv")

# Extract optimizer families + budgets
optimizers = {}
for col in df.columns:
    m = re.match(r"([A-Za-z\+\~]+)-(\d+)$", col)   
    if m:
        family, budget = m.group(1), int(m.group(2))
        optimizers.setdefault(family, []).append(budget)

# Sort budgets inside each family
for fam in optimizers:
    optimizers[fam] = sorted(optimizers[fam])

# Helpers
def is_best(cell):
    """Returns True if cell ends in 'a' ignoring spaces."""
    if not isinstance(cell, str):
        return False
    return cell.strip().endswith("a")

def numeric_value(cell):
    """Extract the numeric part from '43 a' or just '43'."""
    if isinstance(cell, str):
        m = re.match(r"\s*([0-9]*\.?[0-9]+)", cell)
        return float(m.group(1)) if m else np.nan
    return float(cell) if pd.notna(cell) else np.nan

def calculate_D(row_b4_mu, row_b4_lo, optimizer_value):
    """
    Calculate D = 100 * (B4.mu - optimizer.mu) / (B4.mu - B4.lo)
    """
    if pd.isna(row_b4_mu) or pd.isna(row_b4_lo) or pd.isna(optimizer_value):
        return np.nan
    
    denominator = row_b4_mu - row_b4_lo
    if denominator == 0:
        return np.nan
    
    return 100 * (row_b4_mu - optimizer_value) / denominator

# Build table with all metrics
table = []
for fam, budgets in optimizers.items():
    for b in budgets:
        col = f"{fam}-{b}"         # e.g., DEHB-10
        
        if col not in df.columns:
            continue
        
        # Column values
        cells = df[col]
        
        # percent best = % of rows ending with 'a'
        best_mask = cells.apply(is_best)
        percent_best = 100 * best_mask.mean()
        
        # Extract numeric values from optimizer column
        cells_numeric = cells.apply(numeric_value)
        b4_mu = df["B4.mu"].astype(float)
        b4_lo = df["B4.lo"].astype(float)
        
        # Calculate D for each row where data is available
        D_values = []
        for idx in range(len(df)):
            D = calculate_D(b4_mu[idx], b4_lo[idx], cells_numeric[idx])
            if not np.isnan(D):
                D_values.append(D)
        
        # Average D and std across all datasets
        avg_D = np.mean(D_values) if len(D_values) > 0 else np.nan
        std_D = np.std(D_values) if len(D_values) > 0 else np.nan
        
        # Rename optimizers
        display_name = fam
        if fam == "ACT":
            display_name = "LITE"
        elif fam == "KM++":
            display_name = "LINE"
        elif fam == "RAND":
            display_name = "Random"
        
        table.append([display_name, b, percent_best, avg_D, std_D])

# Store result
results_df = pd.DataFrame(table, columns=[
    "optimizer", "budget", "percent_best", "avg_D", "std_D"
])

# Define custom sort order
optimizer_order = ["LITE", "EZR", "LINE", "Random", "SMAC", "TPE", "DEHB"]
results_df["optimizer"] = pd.Categorical(results_df["optimizer"], categories=optimizer_order, ordered=True)
results_df = results_df.sort_values("optimizer")

# Create D column with std notation
results_df["D_with_std"] = results_df.apply(
    lambda row: f"{row['avg_D']:.0f}±{row['std_D']:.0f}" if pd.notna(row['avg_D']) else "NaN",
    axis=1
)

# Pivot to match original format + add D values
percent_best_pivot = results_df.pivot(index="optimizer", columns="budget", values="percent_best")
D_with_std_pivot = results_df.pivot(index="optimizer", columns="budget", values="D_with_std")

# Pretty print
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", None)

print("\n=== TABLE 3 ===\n")
print("\n--- Percent Best (% rows won) ---")
print(percent_best_pivot.round(0))

print("\n--- Average D ± Std Dev (across all datasets per optimizer-budget) ---")
print(D_with_std_pivot)
