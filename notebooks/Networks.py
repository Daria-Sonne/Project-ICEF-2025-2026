import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
from networkx.algorithms.community import greedy_modularity_communities


# 1. LOAD DATA
# normalized returns
returns = pd.read_csv("../data/preprocessed/norm_returns.csv",index_col=0,parse_dates=True)

# sector info
sector_df = pd.read_csv("../data/preprocessed/sectors_industry_filtered.csv")
sector_df["sector"] = sector_df["sector"].fillna("Unknown")

ticker_to_sector = dict(zip(sector_df["ticker"], sector_df["sector"]))

print(f"Returns shape: {returns.shape}")


# 2. DEFINE REGIMES
# based on breakpoint analysis
regimes = {
    "R1": ("2015-08-25", "2016-03-14"),
    "R2": ("2017-06-27", "2018-02-09"),
    "R3": ("2019-02-01", "2020-02-24"),
    "R4": ("2020-02-24", "2020-06-15"),
    "R5": ("2021-06-14", "2021-12-01"),
    "R6": ("2025-04-10", "2026-04-10")}

# 3. MST FUNCTION
def build_mst(window_returns):

    # correlation matrix
    corr = window_returns.corr()

    # Mantegna distance
    dist = np.sqrt(2 * (1 - corr))

    # remove numerical issues
    dist = dist.fillna(2)

    # MST
    mst_sparse = minimum_spanning_tree(dist.values)

    # convert to graph
    G = nx.Graph()
    tickers = corr.columns.tolist()
    rows, cols = mst_sparse.nonzero()

    for i, j in zip(rows, cols):

        G.add_edge(
            tickers[i],
            tickers[j],
            weight=dist.iloc[i, j],
            corr=corr.iloc[i, j])

    return G, corr, dist



# 4. BUILD NETWORKS
mst_results = {}

for regime_name, (start, end) in regimes.items():

    print(f"\n--- {regime_name} ---")

    window = returns.loc[start:end]

    print(f"Window shape: {window.shape}")

    G, corr, dist = build_mst(window)
    mst_results[regime_name] = {
        "graph": G,
        "corr": corr,
        "dist": dist}

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")



# 5. VISUALIZATION
for regime_name, result in mst_results.items():

    G = result["graph"]

    # CENTRALITY
    eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)

    # dataframe for sorting
    cent_df = pd.DataFrame({
        "ticker": list(G.nodes()),
        "eigen": [eigen_cent[n] for n in G.nodes()]})

    cent_df = cent_df.sort_values(by="eigen",ascending=False)

    # KEEP ONLY TOP N NODES
    TOP_N = 80
    top_nodes = cent_df.head(TOP_N)["ticker"].tolist()
    G_sub = G.subgraph(top_nodes).copy()

    # recompute centrality on subgraph
    eigen_sub = nx.eigenvector_centrality(G_sub,max_iter=1000)

    # LAYOUT
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G_sub,seed=42,k=0.35)

    # COLORS BY SECTOR
    unique_sectors = sorted([str(x) for x in set(ticker_to_sector.values())] )
    sector_color_map = {sector: i for i, sector in enumerate(unique_sectors)}
    node_colors = []

    for node in G_sub.nodes():

        sector = ticker_to_sector.get(node, "Unknown")

        node_colors.append(sector_color_map.get(sector, 0))

    # NODE SIZE = CENTRALITY
    node_sizes = [12000 * eigen_sub[n] for n in G_sub.nodes()]

    # DRAW
    nx.draw_networkx_nodes(
        G_sub,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        alpha=0.9)

    nx.draw_networkx_edges(G_sub,pos,alpha=0.3)

    # LABEL ONLY TOP HUBS
    top10 = cent_df.head(10)["ticker"].tolist()

    labels = {
        node: node
        for node in G_sub.nodes()
        if node in top10}

    nx.draw_networkx_labels(
        G_sub,
        pos,
        labels,
        font_size=9)

    # TITLE
    plt.title(f"MST Network — {regime_name}",fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.show()



# 6. CENTRALITY ANALYSIS
all_stats = []

for regime_name, result in mst_results.items():

    G = result["graph"]
    degree_cent = nx.degree_centrality(G)
    eigen_cent = nx.eigenvector_centrality(G,max_iter=1000)
    between_cent = nx.betweenness_centrality(G)

    stats_df = pd.DataFrame({
        "ticker": list(G.nodes()),
        "degree": [degree_cent[n] for n in G.nodes()],
        "eigenvector": [eigen_cent[n] for n in G.nodes()],
        "betweenness": [between_cent[n] for n in G.nodes()],})

    stats_df["sector"] = stats_df["ticker"].map(ticker_to_sector)
    stats_df["regime"] = regime_name
    stats_df = stats_df.sort_values(by="eigenvector",ascending=False)
    all_stats.append(stats_df)

    print(f"\n===== {regime_name} =====")
    print("\nTop eigenvector centrality:")
    print(stats_df[["ticker", "sector", "eigenvector"]].head(10))

# 7. SAVE RESULTS
final_stats = pd.concat(all_stats)

final_stats.to_csv("../data/preprocessed/mst_centrality_stats.csv",index=False)


# 8. COMMUNITY DETECTION
for regime_name, result in mst_results.items():

    G = result["graph"]
    communities = greedy_modularity_communities(G)

    print(f"\n===== {regime_name} =====")
    print(f"Communities found: {len(communities)}")

    sizes = [len(c) for c in communities]

    print("Community sizes:")
    print(sorted(sizes, reverse=True)[:10])