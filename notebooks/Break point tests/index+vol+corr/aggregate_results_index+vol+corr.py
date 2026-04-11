import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATA
data_path = "../../../data/preprocessed/state_index+vol+corr_scaled.csv"

df = pd.read_csv(data_path, index_col=0, parse_dates=True)
X = df.values
dates = df.index

print(f"Loaded state vectors: {df.shape}")


# 2. LOAD BREAKPOINTS
kcp_path = "../../../data/tests/index+vol+corr/kcp_index+vol+corr_stable_breakpoints.csv"
pelt_path = "../../../data/tests/index+vol+corr/pelt_index+vol+corr_stable_breakpoints.csv"

kcp_df = pd.read_csv(kcp_path)
pelt_df = pd.read_csv(pelt_path)

kcp_indices = kcp_df["index"].values
pelt_indices = pelt_df["index"].values

print(f"KCP breakpoints: {len(kcp_indices)}")
print(f"PELT breakpoints: {len(pelt_indices)}")


# 3. MATCH BREAKPOINTS (with window!)
tolerance = 10  # days

matches = []
used_pelt = set()

for k in kcp_indices:
    best_match = None
    best_dist = np.inf

    for p in pelt_indices:
        dist = abs(k - p)
        if dist <= tolerance and dist < best_dist:
            best_match = p
            best_dist = dist

    if best_match is not None:
        matches.append({
            "kcp_index": k,
            "pelt_index": best_match,
            "kcp_date": dates[k],
            "pelt_date": dates[best_match],
            "distance_days": best_dist
        })
        used_pelt.add(best_match)

# DataFrame of matches
matches_df = pd.DataFrame(matches)

print("\n--- MATCHED BREAKPOINTS ---")
print(matches_df)


# 4. UNIQUE BREAKPOINTS
matched_kcp = set(matches_df["kcp_index"])
matched_pelt = set(matches_df["pelt_index"])

kcp_unique = [k for k in kcp_indices if k not in matched_kcp]
pelt_unique = [p for p in pelt_indices if p not in matched_pelt]

print("\nSummary:")
print(f"Matched: {len(matches_df)}")
print(f"KCP unique: {len(kcp_unique)}")
print(f"PELT unique: {len(pelt_unique)}")


# 5. SAVE RESULTS
matches_df.to_csv("../../../data/tests/index+vol+corr/index+vol+corr_breakpoint_matches.csv", index=False)

# 6. SUBPLOTS (KCP vs PELT)
aggregate = X.mean(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# ---- KCP ----
axes[0].plot(dates, aggregate, label="Aggregate state", alpha=0.8)

for idx in kcp_indices:
    axes[0].axvline(dates[idx], linestyle="--", color="red")

axes[0].set_title("KCP index+vol+corr_Breakpoints (Aggregate)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# ---- PELT ----
axes[1].plot(dates, aggregate, label="Aggregate state", alpha=0.8)

for idx in pelt_indices:
    axes[1].axvline(dates[idx], linestyle="--", color="blue")

axes[1].set_title("PELT index+vol+corr_Breakpoints (Aggregate)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../../../assets/plots/tests_index+vol+corr/index+vol+corr_kcp_vs_pelt_aggregate_subplots.png")
plt.show()

# 6. VISUALIZATION (AGGREGATE)

plt.figure(figsize=(14, 6))
plt.plot(dates, aggregate, label="Aggregate state", alpha=0.8)

# matched
for i, row in matches_df.iterrows():
    plt.axvline(row["kcp_date"], color="black", linestyle="--", alpha=0.8,
                label="Matched" if i == 0 else "")

# KCP unique
for i, idx in enumerate(kcp_unique):
    plt.axvline(dates[idx], color="red", linestyle=":", alpha=0.6,
                label="KCP unique" if i == 0 else "")

# PELT unique
for i, idx in enumerate(pelt_unique):
    plt.axvline(dates[idx], color="blue", linestyle=":", alpha=0.6,
                label="PELT unique" if i == 0 else "")

plt.title("Breakpoint Comparison (KCP vs PELT)")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("../../../assets/plots/tests_index+vol+corr/index+vol+corr_kcp_vs_pelt_overlay.png")
plt.show()