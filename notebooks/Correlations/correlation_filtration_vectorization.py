import pickle
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# rolling correlation matrices
with open("../../data/structures/rolling_corr_matrices.pkl", "rb") as f:
    rolling_corr = pickle.load(f)

# normalized returns
norm_returns = pd.read_csv(
    "../../data/preprocessed/norm_returns.csv",
    index_col=0,
    parse_dates=True
)

print(f"Downloaded matrices: {len(rolling_corr)}")

# Filtration Parameters
RHO_THRESHOLD = 0.2   # threshold by |ρ|
FDR_ALPHA = 0.05      # level FDR

# Calculating p-values for Pearson correlation coefficients.
def corr_pvalues(corr_matrix, T):
    r = corr_matrix.values
    t_stat = r * np.sqrt((T - 2) / (1 - r**2 + 1e-12))
    pvals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=T - 2))
    return pd.DataFrame(
        pvals,
        index=corr_matrix.index,
        columns=corr_matrix.columns
    )
#Vectorization and Filtration
vectors = []

for date, corr_mat in rolling_corr.items():
    T = 252
    # 1. p-values
    pval_mat = corr_pvalues(corr_mat, T)

    # 2. upper triangle
    mask = np.triu(np.ones(corr_mat.shape), k=1).astype(bool)

    corr_vals = corr_mat.values[mask]
    pvals = pval_mat.values[mask]

    # 3. value filter
    magnitude_mask = np.abs(corr_vals) >= RHO_THRESHOLD
    corr_vals = corr_vals[magnitude_mask]
    pvals = pvals[magnitude_mask]

    if len(corr_vals) == 0:
        continue

    # 4. FDR correction
    reject, _, _, _ = multipletests(
        pvals,
        alpha=FDR_ALPHA,
        method="fdr_bh"
    )

    filtered_corrs = corr_vals[reject]

    if len(filtered_corrs) == 0:
        continue

    # 5. feature vector
    vector = {f"corr_{i}": val for i, val in enumerate(filtered_corrs)}
    vector["date"] = date

    vectors.append(vector)

print(f"Constructed vectors: {len(vectors)}")


corr_vectors_df = pd.DataFrame(vectors)
corr_vectors_df = corr_vectors_df.set_index("date").sort_index()

print("Dimension corr_vectors:")
print(corr_vectors_df.shape)

corr_vectors_df.to_csv("../../data/preprocessed/corr_vectors.csv")