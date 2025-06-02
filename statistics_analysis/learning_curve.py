import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load your dataset
file_path = "data_clean.xlsx"  # Change if needed
df = pd.read_csv(r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\RO47015 Applied Experimental Methods\aem_drive\data_clean.csv')


# Assign trial order within each subject
df['TrialOrder'] = df.groupby('Subject').cumcount() + 1

# Prepare unique colors per subject
unique_subjects = df['Subject'].unique()
palette = sns.color_palette("husl", len(unique_subjects))
subject_colors = dict(zip(unique_subjects, palette))

# --- Regression with statsmodels to get p-values ---
def regression_and_pvalue(df, dependent_var):
    X = sm.add_constant(df['TrialOrder'])
    y = df[dependent_var]
    model = sm.OLS(y, X).fit()
    print(f"\n=== {dependent_var} vs TrialOrder ===")
    print(model.summary())
    return model.pvalues['TrialOrder'], model.rsquared

pval_offroad, r2_offroad = regression_and_pvalue(df, 'Offroad')
pval_mathscore, r2_mathscore = regression_and_pvalue(df, 'MathScore')

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# Plot 1: Offroad
for subject in unique_subjects:
    subj_data = df[df['Subject'] == subject]
    axs[0].scatter(subj_data['TrialOrder'], subj_data['Offroad'],
                   label=f"Subject {subject}", color=subject_colors[subject], s=50)

sns.regplot(x="TrialOrder", y="Offroad", data=df, ax=axs[0],
            scatter=False, color='black', line_kws={'linewidth': 2, 'linestyle': 'dashed'})

axs[0].set_title(f"Offroad vs Trial Order\n(p = {pval_offroad:.3f}, R² = {r2_offroad:.3f})")
axs[0].set_xlabel("Trial Order")
axs[0].set_ylabel("Offroad")
axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Subjects", fontsize="small")

# Plot 2: MathScore
for subject in unique_subjects:
    subj_data = df[df['Subject'] == subject]
    axs[1].scatter(subj_data['TrialOrder'], subj_data['MathScore'],
                   label=f"Subject {subject}", color=subject_colors[subject], s=50)

sns.regplot(x="TrialOrder", y="MathScore", data=df, ax=axs[1],
            scatter=False, color='black', line_kws={'linewidth': 2, 'linestyle': 'dashed'})

axs[1].set_title(f"MathScore vs Trial Order\n(p = {pval_mathscore:.3f}, R² = {r2_mathscore:.3f})")
axs[1].set_xlabel("Trial Order")
axs[1].set_ylabel("MathScore")

plt.tight_layout()
plt.show()
