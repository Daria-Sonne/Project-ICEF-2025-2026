import pandas as pd

# 1. Загружаем отфильтрованный файл с секторами и индустриями
df_filtered = pd.read_csv("../../data/preprocessed/sectors_industry_filtered.csv")

# Проверим, как называются колонки
print("Колонки в файле:", df_filtered.columns.tolist())

# 2. Группировка по Sector
tickers_by_sector = df_filtered.groupby('Sector')['Ticker'].apply(lambda x: ','.join(x)).reset_index()
tickers_by_sector_file = "../../data/preprocessed/tickers_by_sector.csv"
tickers_by_sector.to_csv(tickers_by_sector_file, index=False)
print(f"Сохранен файл с тикерами по секторам: {tickers_by_sector_file}")

# 3. Группировка по Industry
tickers_by_industry = df_filtered.groupby('Industry')['Ticker'].apply(lambda x: ','.join(x)).reset_index()
tickers_by_industry_file = "../../data/preprocessed/tickers_by_industry.csv"
tickers_by_industry.to_csv(tickers_by_industry_file, index=False)
print(f"Сохранен файл с тикерами по индустриям: {tickers_by_industry_file}")