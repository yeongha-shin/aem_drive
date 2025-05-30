import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\RO47015 Applied Experimental Methods\aem_drive\data_clean.csv')


# Step 1: Assign trial order per subject
df['TrialOrder'] = df.groupby('Subject').cumcount() + 1

# Step 2: Define function to check learning effect via linear regression
def check_learning_effect(metric):
    X = df[['TrialOrder']]
    y = df[metric]
    X = sm.add_constant(X)  # Add intercept term
    model = sm.OLS(y, X).fit()
    print(f"=== Linear Regression for {metric} ~ TrialOrder ===")
    print(model.summary())
    return model

# Run regression for Offroad and MathScore
offroad_model = check_learning_effect('Offroad')
mathscore_model = check_learning_effect('MathScore')

# Optional: Plot the trends
sns.set(style="whitegrid")

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Offroad plot
sns.regplot(x="TrialOrder", y="Offroad", data=df, ax=axs[0], ci=None)
axs[0].set_title("Offroad vs Trial Order")
axs[0].set_xlabel("Trial Order")
axs[0].set_ylabel("Offroad")

# MathScore plot
sns.regplot(x="TrialOrder", y="MathScore", data=df, ax=axs[1], ci=None)
axs[1].set_title("MathScore vs Trial Order")
axs[1].set_xlabel("Trial Order")
axs[1].set_ylabel("MathScore")

plt.tight_layout()
plt.show()
