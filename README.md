# 🔍 Anonymous Repository for "How Low Can You Go? The Data-Light SE Challenge"

> **Note:** This repository has been anonymized to comply with double-blind review requirements. All identifying information (author names, institutional references, Git history) has been redacted. Full attribution and licensing will be restored upon acceptance.

---

## 📄 Summary

This repository contains the code, scripts, and data used to reproduce the results from our paper:

> **"How Low Can You Go? The Data-Light SE Challenge"**
> _Submitted to FSE 2026 (Double-Blind Review)_

We present the **BINGO effect**, a prevalent data compression phenomenon in software engineering (SE) optimization. Leveraging this, we show that **simple optimizers**—`RANDOM`, `LITE`, `LINE`—perform on par with the state-of-the-art `DEHB`, `SMAC`, and `TPE`, while running up to **10,000× faster**.

---

## 🧪 Experimental Setup

All experiments were run on a 4-core Linux (Ubuntu 24.04) system (1.30GHz, 16GB RAM, no GPU).

### Configuration

- **Datasets**: 127 MOOT tasks in `data/moot/`
- **Repeats**: 20 runs per optimizer
- **Budgets**: {6, 12, 18, 24, 50, 100, 200}
- **Optimizers**: `DEHB`, `SMAC`, `TPE`, `LITE`, `LINE`, `RANDOM`
- **Evaluation**:
  - Effectiveness / Benefit: distance-to-heaven (multi-objective)
  - Cost: no. of accessed labels, wall-clock time

---

## 📊 Reproducing the Results (Table 4, Figures 4 & 5)

These instructions reproduce all core results from the paper, including **Table 4**, **Figure 4**, and **Figure 5**.

All experiments were run using **Python 3.13**.

---

### ➤ Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### ➤ Step 2: Generate Table 4

```bash
cd experiments/LUA_run_all/
make comparez
make report
```

The output will be saved to:

```
results/optimization_performance/report.csv
```

---

### ➤ Step 3: Generate Figure 4 (%Best vs. Label Budget)

```bash
cd experiments/
python3 optim_performance_comp.py
```

---

### ➤ Step 4: Generate Figure 5 (Runtime Comparison)

```bash
cd experiments/
python3 performance.py
```

---

### 🧪 Optional: Re-run Optimizers

We include precomputed results for `DEHB`, `SMAC`, `TPE`, and `Active_Learning` (LITE) to save time. To regenerate:

```bash
# A. Remove existing results
rm -rf results/results_DEHB results/results_SMAC results/results_TPE results/results_Active_Learning

# B. Generate commands (use NAME=Active_Learning for LITE, NAME=DEHB for DEHB, NAME=SMAC for SMAC, NAME=TPE for TPE)
make generate-commands NAME=Active_Learning   # or NAME=DEHB, NAME=SMAC, NAME=TPE

# C. Run the optimizer
cd experiments/
./commands.sh
```

---

## ⚙️ Optimizers

| Optimizer | Description |
|-----------|-------------|
| `RANDOM`  | Random sampling of bucketed data |
| `LITE`    | Naive Bayes-based active learner (selects high g/r) |
| `LINE`    | Diversity sampling via KMeans++ |
| `DEHB`    | Differential Evolution + Hyperband |
| `SMAC`    | Sequential Model-Based Algorithm Configuration |
| `TPE`     | Tree-structured Parzen Estimator |

---

## 🔐 License

> Temporarily redacted for double-blind review. Includes MIT-licensed components. Full license will be restored upon acceptance.

---

## 🔗 External Links

> Will be updated upon acceptance:
- 📜 Paper DOI
- 📁 Dataset DOI
- 🧪 Artifact DOI
