import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

window = 252        # 1 year
min_periods = 126   # min half a year
norm_returns = pd.read_csv(
    "../../data/preprocessed/norm_returns.csv",
    index_col=0,
    parse_dates=True
)
#Rolling correlation
rolling_mean_corr = []

dates = norm_returns.index

for t in tqdm(range(window, len(dates))):
    window_data = norm_returns.iloc[t - window:t]

    corr_matrix = window_data.corr()

    # Only upper triangle without diagonal
    corr_values = corr_matrix.values
    upper = corr_values[np.triu_indices_from(corr_values, k=1)]

    rolling_mean_corr.append({
        "date": dates[t],
        "mean_corr": np.nanmean(upper),
        "median_corr": np.nanmedian(upper),
        "std_corr": np.nanstd(upper),
        "neg_corr_share": np.mean(upper < 0)
    })

rolling_corr_df = pd.DataFrame(rolling_mean_corr).set_index("date")


#Visualisation
plt.figure()
plt.plot(rolling_corr_df["mean_corr"])
plt.title("Rolling mean correlation (1Y window)")
plt.savefig('../../assets/plots/rolling_mean_corr.png')
plt.show()


plt.figure()
plt.plot(rolling_corr_df["neg_corr_share"])
plt.title("Share of negative correlations")
plt.savefig('../../assets/plots/negative_corr.png')
plt.show()


plt.figure()
plt.plot(rolling_corr_df["std_corr"])
plt.title("Std of correlations")
plt.savefig('../../assets/plots/std_corr.png')
plt.show()


rolling_corr_matrices = {}

for t in tqdm(range(window, len(dates))):
    date = dates[t]
    window_data = norm_returns.iloc[t - window:t]
    rolling_corr_matrices[date] = window_data.corr()
with open("../../data/structures/rolling_corr_matrices.pkl", "wb") as f:
    pickle.dump(rolling_corr_matrices, f)