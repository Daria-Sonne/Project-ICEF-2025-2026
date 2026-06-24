import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# LOAD
returns = pd.read_csv("../../data/preprocessed/norm_returns.csv",index_col=0, parse_dates=True)
sector_df = pd.read_csv("../../data/preprocessed/sectors_industry_filtered.csv")

returns = returns.dropna()
print(returns.shape)
print(sector_df.shape)


# PCA FACTORS BY SECTOR
sector_factors = pd.DataFrame(index=returns.index)

explained = []

for sector in sorted(sector_df["sector"].dropna().unique()):

    tickers = sector_df.loc[sector_df["sector"] == sector,"ticker"]
    tickers = [t for t in tickers if t in returns.columns]

    if len(tickers) < 3:
        continue

    X = returns[tickers]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(X_scaled)
    sector_factors[sector] = factor.flatten()

    explained.append({"sector": sector,
                      "explained_variance_ratio": pca.explained_variance_ratio_[0],
                      "n_stocks": len(tickers)})

# SAVE
sector_factors.to_csv( "../../data/sectors/sector_pca_factors.csv")
pd.DataFrame(explained).to_csv("../../data/sectors/sector_pca_explained.csv",index=False)

print(sector_factors.shape)