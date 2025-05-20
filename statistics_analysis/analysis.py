import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel, shapiro
from itertools import combinations


# 1. Load your data (ensure long format: Subject, Condition, Offroad, MathScore)
#    Replace 'your_data.csv' with your actual filename
df = pd.read_csv('example_data.csv')


# List of conditions
conditions = df['Condition'].unique()
df = pd.read_csv("example_data.csv")

# Replace any blank / NaN entries in Condition with "None"
df["Condition"] = df["Condition"].fillna("None")

# # Now verify you have exactly four levels:
# print("Conditions now:", df["Condition"].unique())
# print("Counts per subject:\n", df.groupby("Subject")["Condition"].nunique())



# 2. MANOVA: overall multivariate test
'''Purpose: Tests whether the four feedback groups differ on the combination of Off‑road duration and Math score together.

Key statistics (from the printed table):
Wilks’ λ: Multivariate ratio of within‑group to total variance. Smaller λ → larger group differences.
F Value and Pr > F: Tests if λ differs from 1 more than expected by chance.

Interpretation:
Look under the “Condition” row. If Pr > F < 0.05, there’s a significant multivariate effect (i.e. feedback condition shifts the joint outcome).'''

print("=== MANOVA ===")
maov = MANOVA.from_formula('Offroad + MathScore ~ Condition', data=df)
print(maov.mv_test(), "\n")

# 3. Univariate repeated-measures ANOVAs
'''Runs twice, once for dv='Offroad', once for dv='MathScore'.
Purpose: Even if the MANOVA is non‑significant, you still get these to see each outcome in isolation.

Table columns:
    F Value: Ratio of between‑condition variance to residual variance.
    Num DF = k–1 (3 conditions)
    Den DF = (n–1)(k–1) (here 9×3=27)
    Pr > F: p‑value for the Condition effect.
    
Interpretation:
If Pr > F < 0.05, there’s a significant effect of Condition on that outcome.'''

print("=== Repeated-Measures ANOVAs ===")
for dv in ['Offroad', 'MathScore']:
    aov = AnovaRM(df, dv, 'Subject', within=['Condition']).fit()
    print(f"--- {dv} ---")
    print(aov.anova_table, "\n")

# 4. Post-hoc pairwise t-tests with Holm correction
'''Purpose: After a significant univariate ANOVA (here, for MathScore), identify which specific pairs of conditions differ.

Table columns:
t: paired‑samples t‑statistic.
p: raw p‑value for that test.
p_adj: Holm‑adjusted p-value, controlling family‑wise error across the 6 pairs.
d_z: Cohen’s d for paired data (effect size).

Interpretation:
For MathScore, the only pair with raw p < 0.05 was Haptic vs None (p=0.0009), but its adjusted p_adj=0.5109, so not significant after correction.
'''

def holm_correction(pvals):
    sorted_pairs = sorted(enumerate(pvals), key=lambda x: x[1])
    m = len(pvals)
    adj = [None] * m
    for rank, (idx, p) in enumerate(sorted_pairs):
        adj[idx] = min((m - rank) * p, 1.0)
    for i in range(m-2, -1, -1):
        adj[i] = max(adj[i], adj[i+1])
    return adj

print("=== Post-hoc Pairwise Comparisons (paired t-tests) ===")

for dv in ['Offroad', 'MathScore']:
    print(f"--- {dv} ---")
    # Pivot to wide: rows are subjects, cols are conditions
    wide = df.pivot(index='Subject', columns='Condition', values=dv)
    
    pairs = list(combinations(wide.columns, 2))
    pvals = []
    results = []
    for c1, c2 in pairs:
        x = wide[c1]
        y = wide[c2]
        t, p = ttest_rel(x, y, nan_policy='raise')  # will error if any NaNs remain
        diff = (x - y).mean()
        sd_diff = (x - y).std(ddof=1)
        pvals.append(p)
        results.append((c1, c2, t, p, diff, sd_diff))
    
    adj_p = holm_correction(pvals)
    
    for (c1, c2, t, p, diff, sd), p_adj in zip(results, adj_p):
        d = diff / sd  # Cohen's dz for paired samples
        print(f"{c1} vs {c2}: t={t:.3f}, p={p:.4f}, p_adj={p_adj:.4f}, d_z={d:.3f}")
    print()


# 5. Assumption Checks: Normality of difference scores (Shapiro-Wilk)
'''Purpose: The paired t‑test assumes the distribution of differences between two conditions is approximately normal.

Output:
W: Shapiro–Wilk statistic (closer to 1 means closer to normal).
p: Tests the null hypothesis of normality.

Interpretation:
p > 0.05 → cannot reject normality → it’s OK to use the paired t‑test.
If any p < 0.05, you’d consider a non‑parametric Wilcoxon signed‑rank test for that pair'''

print("=== Assumption Checks: Normality of Difference Scores ===")
for dv in ['Offroad', 'MathScore']:
    print(f"--- {dv} ---")
    # Pivot to wide form again
    wide = df.pivot(index='Subject', columns='Condition', values=dv)
    for c1, c2 in combinations(wide.columns, 2):
        diff = wide[c1] - wide[c2]
        # Now diff is a Series of length n_subjects
        stat, p = shapiro(diff)
        print(f"{c1} vs {c2} diff normality: W={stat:.3f}, p={p:.4f}")
    print()
