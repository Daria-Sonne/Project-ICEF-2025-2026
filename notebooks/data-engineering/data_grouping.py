import pandas as pd

# 1. Загружаем отфильтрованный файл с секторами и индустриями
df_filtered = pd.read_csv("../../data/preprocessed/sectors_industry_filtered.csv")

# 2. Группировка по Sector
tickers_by_sector = df_filtered.groupby('sector')['ticker'].apply(lambda x: ','.join(x)).reset_index()
tickers_by_sector.to_csv("../../data/preprocessed/tickers_by_sector.csv", index=False)
print(f"Сохранен файл с тикерами по секторам")

# 3. Группировка по Industry
tickers_by_industry = df_filtered.groupby('industry')['ticker'].apply(lambda x: ','.join(x)).reset_index()
tickers_by_industry.to_csv("../../data/preprocessed/tickers_by_industry.csv", index=False)
print(f"Сохранен файл с тикерами по индустриям")
