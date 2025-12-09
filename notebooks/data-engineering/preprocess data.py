# Фильтрация, очистка и подготовка данных (из уже скачанного CSV)
import pandas as pd
import numpy as np
from tqdm import tqdm


# 1. Читаем сырой файл (который создали на первом этапе)
raw_file = "../../data/raw/sp500_adj_close_raw_2012-01-01_to_20251209.csv"
prices_raw = pd.read_csv(raw_file, index_col=0, parse_dates=True)
print(f"Загружен сырой файл: {prices_raw.shape}")

# 2. Фильтрация
min_days_required = 252 * 5
max_missing_ratio = 0.05


good_tickers = []
for ticker in tqdm(prices_raw.columns, desc="Фильтрация"):
    series = prices_raw[ticker].dropna()

    if len(series) < min_days_required:
        continue

    total_days = len(prices_raw)
    missing_ratio = 1 - len(series) / total_days
    if missing_ratio > max_missing_ratio:
        continue

    good_tickers.append(ticker)

# pd.Series(good_tickers).to_csv("../../data/preprocessed/good_tickers.csv", index=False, header=False)

#3. Загружаем базовый файл с секторами и индустриями и фильтруем
df = pd.read_csv("../../data/raw/sectors_industry.csv")
df_filtered = df[df['ticker'].isin(good_tickers)].copy()


prices = prices_raw[good_tickers].copy()
prices = prices.ffill().bfill()

prices.to_csv("../../data/preprocessed/sp500_adj_close_filtered.csv")
df_filtered.to_csv("../../data/preprocessed/sectors_industry_filtered.csv", index=False)
print(f"\nПосле фильтрации: {len(good_tickers)} тикеров")

