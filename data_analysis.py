import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load excel file
df = pd.read_excel("experiment_log.xlsx")
df["math_score_per_sec"] = df["math_score"] / df["completion_time"]

# Group and aggregate
grouped = df.groupby("condition").agg({
    "math_score_per_sec": ['mean', 'std'],
    "num_collisions": ['mean', 'std'],
    "num_offroad_events": ['mean', 'std']
})
grouped.columns = ['_'.join(col) for col in grouped.columns]
grouped.reset_index(inplace=True)

# Setup
conditions = grouped["condition"]
metrics = ["num_collisions", "num_offroad_events", "math_score_per_sec"]
labels = ["Collisions", "Off-road", "Math/sec"]
x = np.arange(len(metrics))
n_conditions = len(conditions)
bar_width = 0.15
offsets = np.linspace(-bar_width * (n_conditions - 1) / 2, bar_width * (n_conditions - 1) / 2, n_conditions)

# Normalize math/sec to fit on same scale
math_max = grouped["math_score_per_sec_mean"].max()
scale_factor = max(grouped["num_collisions_mean"].max(), grouped["num_offroad_events_mean"].max()) / math_max

# Plotting
#plt.style.use("ggplot")
fig, ax1 = plt.subplots(figsize=(8, 5.5))
ax1.set_facecolor("white")

# Bars
for i, cond in enumerate(conditions):
    values = [
        grouped.loc[i, "num_collisions_mean"],
        grouped.loc[i, "num_offroad_events_mean"],
        grouped.loc[i, "math_score_per_sec_mean"] * scale_factor
    ]
    errors = [
        grouped.loc[i, "num_collisions_std"],
        grouped.loc[i, "num_offroad_events_std"],
        grouped.loc[i, "math_score_per_sec_std"] * scale_factor
    ]
    ax1.bar(x + offsets[i], values, bar_width, yerr=errors, capsize=6, label=cond)

# Secondary y-axis for math/sec
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim()[0] / scale_factor, ax1.get_ylim()[1] / scale_factor)
ax2.set_ylabel("Math Score per Second [1/s]", fontsize=14)

# Formatting
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=13)
ax1.set_ylabel("Collisions / Off-road Count", fontsize=14)
ax1.set_title("Performance by Feedback Condition", fontsize=18, weight='bold')
ax1.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax1.legend(title="Condition", fontsize=12, title_fontsize=13, loc="upper left")

plt.tight_layout()
plt.show()