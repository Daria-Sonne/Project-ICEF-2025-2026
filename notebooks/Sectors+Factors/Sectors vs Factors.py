import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. LOAD DATA
sector_factors = pd.read_csv("../../data/sectors/sector_pca_factors.csv",index_col=0,parse_dates=True)
external = pd.read_csv("../../data/raw/external_factors.csv",index_col=0,parse_dates=True)


# 2. EXTERNAL RETURNS
external_ret = np.log(external / external.shift(1))
external_ret = external_ret.dropna()

print("External returns:", external_ret.shape)

# 3. ALIGN DATASETS
data = sector_factors.join( external_ret, how="inner")
print("Merged dataset:", data.shape)


# 4. REGIMES
regimes = {"R1_2015_2016":("2015-08-25", "2016-03-14"),
           "R2_2017_2018":("2017-06-27", "2018-02-09"),
           "R3_2019_2020":("2019-02-01", "2020-02-24"),
           "R4_COVID":("2020-02-24", "2020-06-15"),
           "R5_2021":("2021-06-14", "2021-12-01"),
           "R6_2025":("2025-04-10", "2026-04-10")}


# 5. COLUMN DETECTION
external_cols = [c for c in external_ret.columns]
sector_cols = [c for c in sector_factors.columns]

print("\nExternal columns:")
print(external_cols)

print("\nSector columns:")
print(sector_cols)


# 6. CORRELATIONS BY REGIME
all_results = []

for regime_name, (start, end) in regimes.items():

    print("\n==========================")
    print(regime_name)
    print("==========================")

    window = data.loc[start:end]

    print(f"Observations: {len(window)}")

    corr = window.corr()
    sector_external_corr = corr.loc[sector_cols,external_cols]

    print("\nSector ↔ External correlation matrix:")
    print(sector_external_corr.round(3))

    # SAVE MATRIX
    sector_external_corr.to_csv(f"../../data/sectors/{regime_name}_sector_external_corr.csv")

    # LONG FORMAT
    tmp = (sector_external_corr.stack().reset_index())
    tmp.columns = ["sector","factor","corr"]
    tmp["regime"] = regime_name
    all_results.append(tmp)


    # HEATMAP
    plt.figure(figsize=(9, 6))
    sns.heatmap(sector_external_corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f")

    plt.title(f"Sector vs External Factors\n{regime_name}")
    plt.tight_layout()
    plt.savefig(f"../../assets/plots/{regime_name}_heatmap.png",dpi=300)
    plt.show()


# 7. SAVE ALL RESULTS
all_results = pd.concat(all_results,ignore_index=True)
all_results.to_csv("../../data/sectors/all_sector_external_correlations.csv", index=False)


# 8. REGIME COMPARISON TABLE
comparison = all_results.pivot_table( index=["sector", "factor"],columns="regime",values="corr")
comparison.to_csv("../../data/sectors/regime_comparison.csv")
print("\nComparison table:")
print(comparison.head(20))


# 9. STRONGEST RELATIONSHIPS
print("\n===== TOP CORRELATIONS =====")

top_corr = (all_results.reindex(all_results["corr"].abs().sort_values(ascending=False).index))
print(top_corr.head(30))
