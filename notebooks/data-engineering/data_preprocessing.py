import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Загрузка отфильтрованных цен
prices = pd.read_csv(
    "../../data/preprocessed/sp500_adj_close_filtered.csv",
    index_col=0,
    parse_dates=True
)

print(f"Загружены цены: {prices.shape}")

# Лог-доходности
log_returns = np.log(prices / prices.shift(1))

# Удаляем первую строку с NaN
log_returns = log_returns.dropna(how="all")

log_returns.to_csv("../../data/preprocessed/log_returns.csv")

# ADF Test
adf_results = []

for ticker in tqdm(log_returns.columns, desc="ADF тест"):
    series = log_returns[ticker].dropna()

    if len(series) < 252:
        continue

    adf_stat, p_value, _, _, critical_values, _ = adfuller(series, autolag="AIC")

    adf_results.append({
        "ticker": ticker,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "is_stationary_5pct": p_value < 0.05
    })

adf_df = pd.DataFrame(adf_results)
adf_df.to_csv("../../data/preprocessed/adf_stationarity_results.csv", index=False)

print(adf_df["is_stationary_5pct"].value_counts())

#Rolling volatility normalization
vol_window = 252   # 1 год
min_periods = 126  # минимум полгода

# Rolling volatility (std)
rolling_vol = log_returns.rolling(
    window=vol_window,
    min_periods=min_periods
).std()

# Нормализованные доходности
norm_returns = log_returns / rolling_vol

# Удаляем периоды, где vol ещё не определена
norm_returns = norm_returns.dropna(how="all")

norm_returns.to_csv("../../data/preprocessed/norm_returns.csv")

# Validation
sample_tickers = log_returns.columns[:5]
ticker = sample_tickers[0]

plt.figure()
plt.plot(rolling_vol[ticker])
plt.title(f"Rolling volatility: {ticker}")
plt.show()

#Средние и стандартные отклонения по всем акциям
plt.figure()
plt.hist(norm_returns.std(), bins=50)
plt.title("Std of normalized returns (cross-section)")
plt.show()

#Cross-sectional mean во времени
cross_sectional_mean = norm_returns.mean(axis=1)

plt.figure()
plt.plot(cross_sectional_mean)
plt.title("Cross-sectional mean of normalized returns")
plt.show()

#Корреляционная структура: до и после
corr_raw = log_returns.corr()
corr_norm = norm_returns.corr()

plt.figure()
plt.hist(corr_raw.values.flatten(), bins=100)
plt.title("Correlations (raw log-returns)")
plt.show()

plt.figure()
plt.hist(corr_norm.values.flatten(), bins=100)
plt.title("Correlations (normalized returns)")
plt.show()