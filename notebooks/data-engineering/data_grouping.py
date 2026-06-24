import pandas as pd

# 1. Loading a filtered file with sectors and industries
df_filtered = pd.read_csv("../../data/preprocessed/sectors_industry_filtered.csv")

# 2. Grouping by Sector
tickers_by_sector = df_filtered.groupby('sector')['ticker'].apply(lambda x: ','.join(x)).reset_index()
tickers_by_sector.to_csv("../../data/preprocessed/tickers_by_sector.csv", index=False)
print(f"Saved file with tickers by sectors")

# 3. Grouping by Industry
tickers_by_industry = df_filtered.groupby('industry')['ticker'].apply(lambda x: ','.join(x)).reset_index()
tickers_by_industry.to_csv("../../data/preprocessed/tickers_by_industry.csv", index=False)
print(f"A file with tickers by industry has been saved.")
