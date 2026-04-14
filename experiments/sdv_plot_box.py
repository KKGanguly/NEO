#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

df = pd.read_csv("../results/optimization_performance/report.csv")
df_data = df[df['D'].notna()].copy().reset_index(drop=True)

RENAME = {'ACT': 'LITE', 'KM++': 'LINE', 'RAND': 'Random'}

def parse_sd_col(c):
    c = str(c).strip().replace('\xa0', ' ')
    m = re.match(r'^-?TPE\s+([\d.]+)_sd$', c)
    if m: return 'TPE', int(float(m.group(1)))
    m = re.match(r'^(.+?)-(\d+)_sd$', c)
    if m: return m.group(1), int(m.group(2))
    return None, None

opt_budget_map = {}
raw_opt_order = []
for col in df.columns:
    if not str(col).endswith('_sd'): continue
    opt, budget = parse_sd_col(col)
    if opt:
        opt_budget_map.setdefault(opt, {})[budget] = col
        if opt not in raw_opt_order:
            raw_opt_order.append(opt)

budgets = sorted({b for opt in opt_budget_map for b in opt_budget_map[opt]})
opts_display = [RENAME.get(o, o) for o in raw_opt_order]
n_opts = len(raw_opt_order)
n_bud  = len(budgets)

# Collect data: data[opt][budget] = array of SD values
data = {}
for opt, display in zip(raw_opt_order, opts_display):
    data[display] = {}
    for b in budgets:
        col = opt_budget_map[opt].get(b)
        vals = pd.to_numeric(df_data[col], errors='coerce').dropna().values if col else np.array([])
        data[display][b] = vals

# ── Plot ──
fig, axes = plt.subplots(1, n_bud, figsize=(16, 4.5), sharey=True)
fig.subplots_adjust(wspace=0.05)

colors = plt.cm.tab10(np.linspace(0, 0.9, n_opts))
color_map = dict(zip(opts_display, colors))

positions = np.arange(1, n_opts + 1)
width = 0.65

for ax, budget in zip(axes, budgets):
    box_data = [data[d][budget] for d in opts_display]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2.5, linestyle='none', alpha=0.4),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )

    for patch, display in zip(bp['boxes'], opts_display):
        c = color_map[display]
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for flier, display in zip(bp['fliers'], opts_display):
        flier.set(markerfacecolor=color_map[display], markeredgewidth=0)

    ax.set_title(f'Budget = {budget}', fontsize=9, pad=4)
    ax.set_xticks([])
    ax.set_xlim(0.3, n_opts + 0.7)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='y', labelsize=8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel('Standard Deviation of $d2h$', fontsize=10)

# Legend
patches = [mpatches.Patch(facecolor=color_map[d], alpha=0.7, label=d) for d in opts_display]
fig.legend(handles=patches, loc='lower center', ncol=n_opts,
           fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Standard Deviation of $d2h$ Across 111 Datasets by Optimizer and Budget',
             fontsize=11, y=1.01)

plt.savefig('../results/optimization_performance/sd_plot.png', bbox_inches='tight', dpi=300)
print("Saved sd_plot.pdf and sd_plot.png")