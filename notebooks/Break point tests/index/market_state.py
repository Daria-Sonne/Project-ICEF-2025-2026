import pandas as pd
from sklearn.preprocessing import StandardScaler

# LOAD
log_returns = pd.read_csv(
    "../../../data/preprocessed/log_returns.csv",index_col=0,parse_dates=True)

# MARKET RETURN
market_return = log_returns.mean(axis=1)

market_state = pd.DataFrame({"market_return": market_return}).dropna()

# SCALE
scaler = StandardScaler()
market_state_scaled = pd.DataFrame(
    scaler.fit_transform(market_state),
    index=market_state.index,
    columns=market_state.columns
)

# SAVE
market_state.to_csv("../../../data/preprocessed/market_state.csv")
market_state_scaled.to_csv("../../../data/preprocessed/market_state_scaled.csv")