import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load rolling correlation matrices
with open("../../data/structures/rolling_corr_matrices.pkl", "rb") as f:
    rolling_corr = pickle.load(f)

print(f"Loaded matrices: {len(rolling_corr)}")

results = []

for date, corr_mat in rolling_corr.items():

    C = corr_mat.values
    n = C.shape[0]

    # The eigenvalues of the correlation matrix
    eigvals = np.linalg.eigvalsh(C)

    # upper triangle indices
    triu_idx = np.triu_indices(n, k=1)
    corr_vals = C[triu_idx]

    # 1 mean correlation
    mean_corr = np.mean(corr_vals)

    # 2 std correlation
    std_corr = np.std(corr_vals)

    # 3 share negative correlations
    share_negative = np.mean(corr_vals < 0)

    # 4 Number of significant factors (Kaiser rule: eigenvalues Kaiser rule > 1)
    n_factors_kaiser = np.sum(eigvals > 1)

    results.append({
        "date": date,
        "mean_corr": mean_corr,
        "std_corr": std_corr,
        "share_negative": share_negative,
        "n_factors_kaiser": n_factors_kaiser,

    })

# Convert to DataFrame
state_vectors = pd.DataFrame(results)
state_vectors = state_vectors.set_index("date").sort_index()

print("\nState vectors shape:")
print(state_vectors.shape)

print("\nPreview:")
print(state_vectors.head())

# Save
state_vectors.to_csv("../../data/preprocessed/state_vectors.csv")

#Standartization
scaler = StandardScaler()

state_vectors_scaled = pd.DataFrame(
    scaler.fit_transform(state_vectors),
    index=state_vectors.index,
    columns=state_vectors.columns
)

print("\nScaled preview:")
print(state_vectors_scaled.head())

# Save scaled version
state_vectors_scaled.to_csv("../../data/preprocessed/state_vectors_scaled.csv")


# Checking whether there is any correlation between features
sns.heatmap(state_vectors.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation between state features")
plt.savefig('../../assets/plots/corr_state_features.png')
plt.show()