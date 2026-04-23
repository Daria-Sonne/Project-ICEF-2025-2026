import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
from collections import Counter

# 1. LOAD DATA
data_path = "../../../data/preprocessed/state_vectors_scaled.csv"

df = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"Loaded state vectors: {df.shape}")

X = df.values
dates = df.index


# 2. KCP MODEL
model = rpt.KernelCPD(kernel="rbf").fit(X)

# 3. PENALTY GRID
penalties = [1, 2, 5, 10, 20, 50]

all_breakpoints = {}

print("\n--- Running KCP with different penalties ---")

for pen in penalties:
    bkpts = model.predict(pen=pen)
    all_breakpoints[pen] = bkpts

    print(f"pen={pen:>3} → {len(bkpts) - 1} breakpoints")


# 4. STABILITY ANALYSIS
# Сollect all breakpoints (except the last ones)
all_bkpt_indices = []

for pen, bkpts in all_breakpoints.items():
    all_bkpt_indices.extend(bkpts[:-1])

# counting frequencies
counts = Counter(all_bkpt_indices)

# convert to DataFrame
stability_df = pd.DataFrame({
    "index": list(counts.keys()),
    "count": list(counts.values())
})

# add dates
stability_df["date"] = stability_df["index"].apply(lambda i: dates[i])

# sorting based on stability
stability_df = stability_df.sort_values(by="count", ascending=False)

print("\n--- Most stable breakpoints ---")
print(stability_df.head(10))


# 5. VISUALIZATION (SUBPLOTS)
fig, axes = plt.subplots(len(df.columns), 1, figsize=(14, 10), sharex=True)

for i, col in enumerate(df.columns):
    axes[i].plot(dates, X[:, i])
    axes[i].set_title(col)

    # all breakpoints
    for pen, bkpts in all_breakpoints.items():
        for b in bkpts[:-1]:
            axes[i].axvline(dates[b], alpha=0.1, color="red")

    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../../../assets/plots/tests_corr/kcp_corr_subplots_all_breaks.png")
plt.show()


# 6. STABLE BREAKPOINTS (SUBPLOTS)
threshold = 3
stable_points = stability_df[stability_df["count"] >= threshold]

fig, axes = plt.subplots(len(df.columns), 1, figsize=(14, 10), sharex=True)

for i, col in enumerate(df.columns):
    axes[i].plot(dates, X[:, i])
    axes[i].set_title(col)

    # only stable breakpoints
    for idx in stable_points["index"]:
        axes[i].axvline(dates[idx], linestyle="--", color="black")

    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../../../assets/plots/tests_corr/kcp_corr_subplots_stable.png")
plt.show()

# 7. SAVE RESULTS
stability_df.to_csv("../../../data/tests/Corr/kcp_corr_breakpoints_stability.csv", index=False)

stable_points.to_csv("../../../data/tests/Corr/kcp_corr_stable_breakpoints.csv", index=False)

print("\nResults saved.")