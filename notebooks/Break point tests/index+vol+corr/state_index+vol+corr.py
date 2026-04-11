import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- LOAD ---
# correlation features
corr = pd.read_csv("../../../data/preprocessed/state_vectors.csv", index_col=0,parse_dates=True)

# raw returns
log_returns = pd.read_csv("../../../data/preprocessed/log_returns.csv",index_col=0,parse_dates=True)

print(f"Corr shape: {corr.shape}")
print(f"Returns shape: {log_returns.shape}")

# --- MARKET ---
market_return = log_returns.mean(axis=1)

# --- VOLATILITY ---
window = 252
min_periods = 126

market_vol = market_return.rolling(window=window,min_periods=min_periods).std()

# --- COMBINE ---
combined = pd.concat([
    corr,
    market_return.rename("market_return"),
    market_vol.rename("market_vol")], axis=1)

combined = combined.dropna()

print("\nCombined preview:")
print(combined.head())

# --- SCALE ---
scaler = StandardScaler()

combined_scaled = pd.DataFrame(
    scaler.fit_transform(combined),
    index=combined.index,
    columns=combined.columns)

# --- SAVE ---
combined.to_csv("../../../data/preprocessed/state_index+vol+corr.csv")
combined_scaled.to_csv("../../../data/preprocessed/state_index+vol+corr_scaled.csv")
