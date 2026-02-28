import pickle
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# Load data
with open("../../data/structures/rolling_corr_matrices.pkl", "rb") as f:
    rolling_corr = pickle.load(f)

print(f"Loaded matrices: {len(rolling_corr)}")

# Parameters
RHO_THRESHOLD = 0.25
FDR_ALPHA = 0.05
T = 252  # rolling window length


# Helper: p-values
def corr_pvalues(corr_matrix, T):
    r = corr_matrix.values
    t_stat = r * np.sqrt((T - 2) / (1 - r**2 + 1e-12))
    pvals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=T - 2))
    return pd.DataFrame(
        pvals,
        index=corr_matrix.index,
        columns=corr_matrix.columns
    )

# Filtering step
filtered_corr = {}

for date, corr_mat in rolling_corr.items():

    pval_mat = corr_pvalues(corr_mat, T)

    mask = np.triu(np.ones(corr_mat.shape), k=1).astype(bool)

    corr_vals = corr_mat.values[mask]
    pvals = pval_mat.values[mask]

    # filter by magnitude
    mag_mask = np.abs(corr_vals) >= RHO_THRESHOLD
    corr_vals = corr_vals[mag_mask]
    pvals = pvals[mag_mask]

    if len(corr_vals) == 0:
        continue

    # FDR correction
    reject, _, _, _ = multipletests(
        pvals, alpha=FDR_ALPHA, method="fdr_bh"
    )

    # build filtered matrix
    filtered = corr_mat.copy()
    filtered.iloc[:, :] = 0.0

    idx = np.where(mask)
    idx = (idx[0][mag_mask][reject], idx[1][mag_mask][reject])

    filtered.values[idx] = corr_vals[reject]
    filtered.values[(idx[1], idx[0])] = corr_vals[reject]

    filtered_corr[date] = filtered

print(f"Filtered matrices: {len(filtered_corr)}")


# Save result
with open("../../data/structures/filtered_corr_matrices.pkl", "wb") as f:
    pickle.dump(filtered_corr, f)