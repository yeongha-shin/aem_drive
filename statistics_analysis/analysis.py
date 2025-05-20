import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel, shapiro
from itertools import combinations

# 1. Load your data (ensure long format: Subject, Condition, Offroad, MathScore)
#    Replace 'your_data.csv' with your actual filename
df = pd.read_csv('your_data.csv')

# List of conditions
conditions = df['Condition'].unique()

# 2. MANOVA: overall multivariate test
print("=== MANOVA ===")
maov = MANOVA.from_formula('Offroad + MathScore ~ Condition', data=df)
print(maov.mv_test(), "\n")

# 3. Univariate repeated-measures ANOVAs
print("=== Repeated-Measures ANOVAs ===")
for dv in ['Offroad', 'MathScore']:
    aov = AnovaRM(df, dv, 'Subject', within=['Condition']).fit()
    print(f"--- {dv} ---")
    print(aov.anova_table, "\n")

# 4. Post-hoc pairwise t-tests with Holm correction
def holm_correction(pvals):
    sorted_pairs = sorted(enumerate(pvals), key=lambda x: x[1])
    m = len(pvals)
    adj = [None] * m
    # initial adjustment
    for rank, (idx, p) in enumerate(sorted_pairs):
        adj[idx] = min((m - rank) * p, 1.0)
    # enforce monotonicity
    for i in range(m-2, -1, -1):
        adj[i] = max(adj[i], adj[i+1])
    return adj

print("=== Post-hoc Pairwise Comparisons (paired t-tests) ===")
for dv in ['Offroad', 'MathScore']:
    print(f"--- {dv} ---")
    pairs = list(combinations(conditions, 2))
    pvals = []
    diffs = []
    for (c1, c2) in pairs:
        x = df[df.Condition == c1][dv].values
        y = df[df.Condition == c2][dv].values
        t, p = ttest_rel(x, y)
        diff = np.mean(x - y)
        sd_diff = np.std(x - y, ddof=1)
        pvals.append(p)
        diffs.append((c1, c2, t, p, diff, sd_diff))
    adj_p = holm_correction(pvals)
    for (c1, c2, t, p, diff, sd), p_adj in zip(diffs, adj_p):
        d = diff / sd  # Cohen's d for paired samples
        print(f"{c1} vs {c2}: t={t:.3f}, p={p:.4f}, p_adj={p_adj:.4f}, d={d:.3f}")
    print()

# 5. Assumption Checks: Normality of difference scores (Shapiro-Wilk)
print("=== Assumption Checks: Normality of Difference Scores ===")
for dv in ['Offroad', 'MathScore']:
    print(f"--- {dv} ---")
    for (c1, c2) in combinations(conditions, 2):
        diff = df[df.Condition == c1][dv].values - df[df.Condition == c2][dv].values
        stat, p = shapiro(diff)
        print(f"{c1} vs {c2} diff normality: W={stat:.3f}, p={p:.4f}")
    print()
