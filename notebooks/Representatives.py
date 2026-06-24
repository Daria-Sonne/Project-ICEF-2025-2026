import pandas as pd
import matplotlib.pyplot as plt


# 1. LOAD CENTRALITY RESULTS
stats = pd.read_csv("../data/preprocessed/mst_centrality_stats.csv")
print(f"Loaded rows: {stats.shape}")


# 2. GLOBAL REPRESENTATIVE STOCKS
# average eigenvector centrality across all regimes
global_rep = (stats.groupby(["ticker", "sector"])["eigenvector"].mean().reset_index()
    .sort_values("eigenvector", ascending=False))

print("\n===== GLOBAL REPRESENTATIVE STOCKS =====")
print(global_rep.head(20))

# save
#global_rep.to_csv("../data/preprocessed/global_representative_stocks.csv",index=False)



# 3. REGIME-SPECIFIC REPRESENTATIVES
top_n = 10
regime_results = []

print("\n\n===== REGIME REPRESENTATIVES =====")

for regime in stats["regime"].unique():

    subset = (stats[stats["regime"] == regime].sort_values("eigenvector", ascending=False)
        .head(top_n).copy())

    subset["rank"] = range(1, len(subset) + 1)

    regime_results.append(subset)

    print(f"\n--- {regime} ---")
    print(subset[["rank", "ticker", "sector", "eigenvector"]])


# combine
regime_df = pd.concat(regime_results)

# save
#regime_df.to_csv("../data/networks/regime_representative_stocks.csv",index=False)



# 4. LEADERSHIP STABILITY
# how often stock appears in top-N
leader_counts = (
    regime_df.groupby(["ticker", "sector"]).size().reset_index(name="appearances")
    .sort_values("appearances", ascending=False))

print("\n\n===== STABLE MARKET LEADERS =====")
print(leader_counts.head(20))

# save
#leader_counts.to_csv(  "../data/networks/stable_market_leaders.csv", index=False)



# 5. VISUALIZATION — TOP GLOBAL REPRESENTATIVES
top_plot = global_rep.head(15)

plt.figure(figsize=(12, 6))

plt.bar(top_plot["ticker"],top_plot["eigenvector"])

plt.xticks(rotation=45)

plt.title("Top Representative Stocks (Average Eigenvector Centrality)")
plt.ylabel("Average Eigenvector Centrality")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../assets/plots/networks/top_representative_stocks.png")
plt.show()



# 6. SECTOR DISTRIBUTION OF REPRESENTATIVES
sector_counts = (regime_df.groupby("sector").size().sort_values(ascending=False))

print("\n\n===== SECTOR DISTRIBUTION =====")
print(sector_counts)

plt.figure(figsize=(12, 6))
sector_counts.plot(kind="bar")
plt.xticks(rotation=45)
plt.title("Sector Distribution of Representative Stocks")
plt.ylabel("Count")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig( "../assets/plots/networks/sector_distribution_representatives.png")
plt.show()
