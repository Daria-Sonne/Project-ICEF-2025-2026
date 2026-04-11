import pandas as pd
from sklearn.preprocessing import StandardScaler

# LOAD
log_returns = pd.read_csv("../../../data/preprocessed/log_returns.csv",index_col=0,parse_dates=True)

# FEATURES
market_return = log_returns.mean(axis=1)
dispersion = log_returns.std(axis=1)

market_disp_state = pd.DataFrame({"market_return": market_return,
    "dispersion": dispersion}).dropna()

# SCALE
scaler = StandardScaler()
market_disp_scaled = pd.DataFrame(
    scaler.fit_transform(market_disp_state),
    index=market_disp_state.index,
    columns=market_disp_state.columns)

# SAVE
market_disp_state.to_csv("../../../data/preprocessed/market_disp_state.csv")
market_disp_scaled.to_csv("../../../data/preprocessed/market_disp_state_scaled.csv")