import pandas as pd
import matplotlib.pyplot as plt


# 1. LOAD DATA
data_path = "../../data/preprocessed/state_vectors_scaled.csv"

df = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"Loaded state vectors: {df.shape}")

X = df.values
dates = df.index

# 2. LOAD BREAKPOINTS
kcp_path = "../../data/tests/kcp_stable_breakpoints.csv"
pelt_path = "../../data/tests/pelt_stable_breakpoints.csv"

kcp_df = pd.read_csv(kcp_path)
pelt_df = pd.read_csv(pelt_path)

kcp_indices = kcp_df["index"].values
pelt_indices = pelt_df["index"].values

print(f"KCP breakpoints: {len(kcp_indices)}")
print(f"PELT breakpoints: {len(pelt_indices)}")


# 3. AGGREGATED INDEX
aggregate = X.mean(axis=1)


# 4. SUBPLOTS (KCP vs PELT)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# ---- KCP ----
axes[0].plot(dates, aggregate, label="Aggregate state", alpha=0.8)

for idx in kcp_indices:
    axes[0].axvline(dates[idx], linestyle="--", color="red")

axes[0].set_title("KCP Breakpoints (Aggregate)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# ---- PELT ----
axes[1].plot(dates, aggregate, label="Aggregate state", alpha=0.8)

for idx in pelt_indices:
    axes[1].axvline(dates[idx], linestyle="--", color="blue")

axes[1].set_title("PELT Breakpoints (Aggregate)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../../assets/plots/kcp_vs_pelt_aggregate_subplots.png")
plt.show()


# 5. OVERLAY (оба метода)
plt.figure(figsize=(14, 6))
plt.plot(dates, aggregate, label="Aggregate state", alpha=0.8)

# KCP
for i, idx in enumerate(kcp_indices):
    plt.axvline(
        dates[idx],
        linestyle="--",
        color="red",
        alpha=0.7,
        label="KCP" if i == 0 else ""
    )

# PELT
for i, idx in enumerate(pelt_indices):
    plt.axvline(
        dates[idx],
        linestyle=":",
        color="blue",
        alpha=0.7,
        label="PELT" if i == 0 else ""
    )

plt.title("KCP vs PELT Breakpoints (Overlay)")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("../../assets/plots/kcp_vs_pelt_overlay.png")
plt.show()