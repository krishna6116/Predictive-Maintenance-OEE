"""Builds notebooks/01_eda.ipynb (Phase 1 EDA on FD001).

Run from the project root:  python notebooks/build_01_eda.py
Then execute it:            jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Phase 1: Exploratory data analysis (FD001)

**Goal:** understand how the engines wear out and which sensors show it, before any modeling.

FD001 is the simplest C-MAPSS subset: one operating condition and one failure mode (high-pressure compressor degradation). Each row is one engine at one cycle (one flight)."""),
    code("""import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from pipeline.check_raw_data import COLS, load  # same loader the pipeline uses

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

train = load(ROOT / "data/raw/train_FD001.txt")
test = load(ROOT / "data/raw/test_FD001.txt")
true_rul = pd.read_csv(ROOT / "data/raw/RUL_FD001.txt", header=None, names=["RUL"])
print("train:", train.shape, "| test:", test.shape, "| engines:", train.unit.nunique(), "/", test.unit.nunique())
train.head()"""),
    md("""## 1. How long do engines last?

**Concept:** every training engine runs until it fails, so its last cycle is its lifetime.
**Why:** the spread of lifetimes tells us how hard prediction is. If all engines failed at the same cycle, a simple average would be enough."""),
    code("""life = train.groupby("unit")["cycle"].max()
print(life.describe().round(1))

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(life, bins=20, color="#4C72B0", edgecolor="white")
ax.axvline(life.median(), color="black", linestyle="--", label=f"median {life.median():.0f} cycles")
ax.set(title="FD001: engine lifetimes vary almost 3x", xlabel="Cycles until failure", ylabel="Engines")
ax.legend()
fig.savefig(FIG / "fd001_lifetimes.png", dpi=150, bbox_inches="tight")"""),
    md("""## 2. Which sensors never change?

**Concept:** a sensor with zero (or near-zero) spread carries no information about wear.
**Why:** dropping useless inputs makes models faster and less noisy, and each drop needs a reason we can state."""),
    code("""cols = [c for c in COLS if c not in ("unit", "cycle")]
spread = pd.DataFrame({
    "std": train[cols].std(),
    "distinct_values": train[cols].nunique(),
}).sort_values("std")
spread.round(5)"""),
    code("""constant = spread.index[spread.distinct_values <= 2].tolist()
settings = ["setting1", "setting2", "setting3"]
sensors = [c for c in cols if c.startswith("s") and c not in settings and c not in constant]
print("Drop (constant or two-valued):", constant)
print("Operating settings (one condition in FD001, used only for checks):", settings)
print(f"Keep {len(sensors)} sensors:", sensors)"""),
    md("""## 3. How do the useful sensors change as engines age?

**Concept:** plotting each sensor against *cycles before failure* lines every engine up at its failure point, so the wear trend becomes visible.
**Why:** sensors that drift steadily toward failure carry the signal the model will learn from."""),
    code("""train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
sample = train[train.unit.isin(range(1, 11))]

fig, axes = plt.subplots(5, 3, figsize=(14, 16), sharex=True)
for ax, s in zip(axes.flat, sensors):
    for _, g in sample.groupby("unit"):
        ax.plot(-g.RUL, g[s], linewidth=0.8, alpha=0.7)
    ax.set_title(s)
for ax in axes.flat[len(sensors):]:
    ax.remove()
fig.supxlabel("Cycles before failure (0 = failure)")
fig.suptitle("FD001: 14 sensors drift as engines approach failure (10 engines shown)", y=1.0)
fig.tight_layout()
fig.savefig(FIG / "fd001_sensor_trends.png", dpi=150, bbox_inches="tight")"""),
    md("""## 4. Which sensors are most linked to remaining life?

**Concept:** correlation runs from -1 to +1. A strong negative value means the sensor rises as remaining life falls.
**Why:** it ranks the sensors and exposes near-duplicates (pairs that carry the same information)."""),
    code("""corr_rul = train[sensors + ["RUL"]].corr()["RUL"].drop("RUL").sort_values(key=abs, ascending=False)
print(corr_rul.round(3).to_string())

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(train[sensors + ["RUL"]].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, ax=ax)
ax.set_title("FD001: sensor correlations (last row = link to remaining life)")
fig.savefig(FIG / "fd001_correlations.png", dpi=150, bbox_inches="tight")"""),
    code("""pairs = train[sensors].corr().abs()
dupes = [(a, b, round(pairs.loc[a, b], 3)) for i, a in enumerate(sensors) for b in sensors[i + 1:] if pairs.loc[a, b] > 0.9]
print("Sensor pairs with |correlation| > 0.9:", dupes)"""),
    md("""## 5. What does the test set look like?

**Concept:** test engines stop before failure. We predict remaining life at each engine's last observed cycle and score against `RUL_FD001.txt`.
**Why:** this decides the RUL cap (Phase 3) and tells us how much history the models will see for each engine."""),
    code("""observed = test.groupby("unit")["cycle"].max()
print("Cycles observed per test engine:", observed.describe().round(1).to_dict())
print("True remaining life:", true_rul.RUL.describe().round(1).to_dict())
print(f"Test engines with true RUL above 125: {(true_rul.RUL > 125).mean():.0%}")"""),
    md("""## Findings

See `docs/EDA_FINDINGS.md` for the written summary and the decisions they lead to."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/01_eda.ipynb")
print("wrote notebooks/01_eda.ipynb")
