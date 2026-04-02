import pandas as pd
import matplotlib.pyplot as plt


# 1. LOAD DATA
state_path = "../../data/preprocessed/state_vectors_scaled.csv"
breaks_path = "../../data/tests/final_breakpoints.csv"

df = pd.read_csv(state_path, index_col=0, parse_dates=True)
breaks = pd.read_csv(breaks_path, parse_dates=["date"])

X = df.copy()
dates = X.index

print(f"Loaded state vectors: {X.shape}")
print(f"Final breakpoints: {len(breaks)}")


# 2. AGGREGATED INDEX PLOT
aggregate = X.mean(axis=1)

plt.figure(figsize=(14, 6))
plt.plot(dates, aggregate, label="Aggregate state", alpha=0.8)

for i, row in breaks.iterrows():
    plt.axvline(row["date"], color="black", linestyle="--",
                label="Breakpoint" if i == 0 else "")

plt.title("Aggregate Market State with Breakpoints")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("../../assets/plots/final_breakpoints_aggregate.png")
plt.show()


# 3. BEFORE / AFTER ANALYSIS
window = 30
results = []

for _, row in breaks.iterrows():

    date = row["date"]

    if date not in X.index:
        continue

    idx = X.index.get_loc(date)

    before = X.iloc[max(0, idx - window):idx]
    after = X.iloc[idx:min(len(X), idx + window)]

    stats = {"date": date}

    for col in X.columns:
        stats[f"{col}_before"] = before[col].mean()
        stats[f"{col}_after"] = after[col].mean()
        stats[f"{col}_change"] = stats[f"{col}_after"] - stats[f"{col}_before"]

    results.append(stats)

changes_df = pd.DataFrame(results)
changes_df.to_csv("../../data/tests/final_breakpoint_changes.csv", index=False)

print("\nChanges saved.")


# 4. BAR PLOTS (changes)
for _, row in changes_df.iterrows():

    date = row["date"]

    changes = [row[col] for col in changes_df.columns if "_change" in col]
    labels = [col.replace("_change", "") for col in changes_df.columns if "_change" in col]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, changes)

    plt.axhline(0)
    plt.title(f"Changes around breakpoint {date}")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    #plt.savefig(f"../../assets/plots/change_{date}.png")
    plt.show()
    plt.close()


# 5. LOCAL DYNAMICS
for _, row in breaks.iterrows():

    date = row["date"]

    if date not in X.index:
        continue

    idx = X.index.get_loc(date)

    start = max(0, idx - window)
    end = min(len(X), idx + window)

    subset = X.iloc[start:end]

    plt.figure(figsize=(12, 6))

    for col in subset.columns:
        plt.plot(subset.index, subset[col], label=col)

    plt.axvline(date, color="black", linestyle="--")

    plt.title(f"Local dynamics around {date}")
    plt.legend()
    plt.grid(alpha=0.3)

    #plt.savefig(f"../../assets/plots/local_{date}.png")
    plt.show()
    plt.close()
