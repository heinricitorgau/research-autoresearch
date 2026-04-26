"""
Autoresearch Experiment Analysis

Analysis of autonomous hyperparameter tuning results from `results.tsv`.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

TSV_PATH = "results.tsv"
if not os.path.exists(TSV_PATH):
    print(f"[ERROR] '{TSV_PATH}' not found. Run at least one experiment first.")
    print("Expected columns: commit, val_bpb, memory_gb, status, description")
    sys.exit(1)

# Load the TSV (tab-separated, 5 columns: commit, val_bpb, memory_gb, status, description)
df = pd.read_csv(TSV_PATH, sep="\t")
df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
df["status"] = df["status"].str.strip().str.upper()

print(f"Total experiments: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(df.head(10))
print("\n" + "="*80 + "\n")

counts = df["status"].value_counts()
print("Experiment outcomes:")
print(counts.to_string())

n_keep = counts.get("KEEP", 0)
n_discard = counts.get("DISCARD", 0)
n_crash = counts.get("CRASH", 0)
n_decided = n_keep + n_discard
if n_decided > 0:
    print(f"\nKeep rate: {n_keep}/{n_decided} = {n_keep / n_decided:.1%}")
    
print("\n" + "="*80 + "\n")

# Show all KEPT experiments (the improvements that stuck)
kept = df[df["status"] == "KEEP"].copy()
print(f"KEPT experiments ({len(kept)} total):\n")
for i, row in kept.iterrows():
    bpb = row["val_bpb"]
    desc = row["description"]
    print(f"  #{i:3d}  bpb={bpb:.6f}  mem={row['memory_gb']:.1f}GB  {desc}")
    
print("\n" + "="*80 + "\n")

# ---------------------------------------------------------
# Val BPB Over Time
# Track how the best (kept) val_bpb evolves as experiments progress. 
# The running minimum shows the "frontier" -- the best result achieved so far.
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(16, 8))

# Filter out crashes for plotting
valid = df[df["status"] != "CRASH"].copy()
valid = valid.reset_index(drop=True)

baseline_bpb = valid.loc[0, "val_bpb"]

# Only plot points at or below baseline (the interesting region)
below = valid[valid["val_bpb"] <= baseline_bpb + 0.0005]

# Plot discarded as faint background dots
disc = below[below["status"] == "DISCARD"]
ax.scatter(disc.index, disc["val_bpb"],
           c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded")

# Plot kept experiments as prominent green dots
kept_v = below[below["status"] == "KEEP"]
ax.scatter(kept_v.index, kept_v["val_bpb"],
           c="#2ecc71", s=50, zorder=4, label="Kept", edgecolors="black", linewidths=0.5)

# Running minimum step line
kept_mask = valid["status"] == "KEEP"
kept_idx = valid.index[kept_mask]
kept_bpb = valid.loc[kept_mask, "val_bpb"]
running_min = kept_bpb.cummin()
ax.step(kept_idx, running_min, where="post", color="#27ae60",
        linewidth=2, alpha=0.7, zorder=3, label="Running best")

# Label each kept experiment with its description
for idx, bpb in zip(kept_idx, kept_bpb):
    desc = str(valid.loc[idx, "description"]).strip()
    if len(desc) > 45:
        desc = desc[:42] + "..."

    ax.annotate(desc, (idx, bpb),
                textcoords="offset points",
                xytext=(6, 6), fontsize=8.0,
                color="#1a7a3a", alpha=0.9,
                rotation=30, ha="left", va="bottom")

n_total = len(df)
n_kept = len(df[df["status"] == "KEEP"])
ax.set_xlabel("Experiment #", fontsize=12)
ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
ax.set_title(f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements", fontsize=14)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.2)

# Y-axis: from just below best to just above baseline
if len(kept_bpb) > 0:
    best_bpb = kept_bpb.min()
    margin = (baseline_bpb - best_bpb) * 0.15 if baseline_bpb != best_bpb else 0.01
    ax.set_ylim(best_bpb - margin, baseline_bpb + margin)
else:
    ax.set_ylim(baseline_bpb - 0.05, baseline_bpb + 0.05)

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to progress.png")

print("\n" + "="*80 + "\n")

# ---------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------

# Summary stats
kept = df[df["status"] == "KEEP"].copy()
baseline_bpb = df.iloc[0]["val_bpb"]

if kept.empty:
    print(f"Baseline val_bpb:  {baseline_bpb:.6f}")
    print("No KEEP experiments yet — no improvement to report.")
else:
    best_bpb = kept["val_bpb"].min()
    best_row = kept.loc[kept["val_bpb"].idxmin()]
    print(f"Baseline val_bpb:  {baseline_bpb:.6f}")
    print(f"Best val_bpb:      {best_bpb:.6f}")
    print(f"Total improvement: {baseline_bpb - best_bpb:.6f} ({(baseline_bpb - best_bpb) / baseline_bpb * 100:.2f}%)")
    print(f"Best experiment:   {best_row['description']}")
print()

# How many experiments to find each improvement
print("Cumulative effort per improvement:")
kept_sorted = kept.reset_index()
for i, (_, row) in enumerate(kept_sorted.iterrows()):
    desc = str(row["description"]).strip()
    print(f"  Experiment #{row['index']:3d}: bpb={row['val_bpb']:.6f}  {desc}")

print("\n" + "="*80 + "\n")
    
# ---------------------------------------------------------
# Top Hits (Kept Experiments by Improvement)
# ---------------------------------------------------------

# Each kept experiment's delta is measured vs the previous kept experiment's bpb
# (since experiments are cumulative -- each one builds on the last kept state)
kept = df[df["status"] == "KEEP"].copy()

if len(kept) < 2:
    print("Not enough KEEP experiments for delta analysis (need at least 2).")
else:
    kept["prev_bpb"] = kept["val_bpb"].shift(1)
    kept["delta"] = kept["prev_bpb"] - kept["val_bpb"]

    # Drop baseline (no delta)
    hits = kept.iloc[1:].copy()

    # Sort by delta improvement (biggest first)
    hits = hits.sort_values("delta", ascending=False)

    print(f"{'Rank':>4}  {'Delta':>8}  {'BPB':>10}  Description")
    print("-" * 80)
    for rank, (_, row) in enumerate(hits.iterrows(), 1):
        print(f"{rank:4d}  {row['delta']:+.6f}  {row['val_bpb']:.6f}  {row['description']}")

    print(f"\n{'':>4}  {hits['delta'].sum():+.6f}  {'':>10}  TOTAL improvement over baseline")
