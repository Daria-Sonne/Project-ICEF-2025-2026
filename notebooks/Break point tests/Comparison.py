import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# LOAD
corr = pd.read_csv("../../data/tests/Corr/corr_final_breakpoints.csv", parse_dates=["date"])
market = pd.read_csv("../../data/tests/index/index_final_breakpoints.csv", parse_dates=["date"])
market_disp = pd.read_csv("../../data/tests/index+vol/index+vol_final_breakpoints.csv", parse_dates=["date"])
combined = pd.read_csv("../../data/tests/index+vol+corr/index+vol+corr_final_breakpoints.csv", parse_dates=["date"])
state = pd.read_csv("../../data/preprocessed/state_vectors_scaled.csv",index_col=0,parse_dates=True)
df_sp = pd.read_csv("../../data/raw/sp500_data.csv", index_col=0, parse_dates=["Date"])

# add labels
corr["model"] = "corr"
market["model"] = "market"
market_disp["model"] = "market_disp"
combined["model"] = "combined"

#Comparison function
def match_breakpoints(df1, df2, tolerance=10):
    matches = []

    for d1 in df1["date"]:
        for d2 in df2["date"]:
            if abs((d1 - d2).days) <= tolerance:
                matches.append((d1, d2))

    return matches

#Comparison
pairs = [
    ("corr", corr, "market", market),
    ("corr", corr, "market_disp", market_disp),
    ("corr", corr, "combined", combined),]

results = []

for name1, df1, name2, df2 in pairs:
    matches = match_breakpoints(df1, df2)

    results.append({
        "model_1": name1,
        "model_2": name2,
        "n_model_1": len(df1),
        "n_model_2": len(df2),
        "matches": len(matches),
        "match_ratio": len(matches) / len(df1) if len(df1) > 0 else 0
    })

results_df = pd.DataFrame(results)
print(results_df)

# # Vizualization
# dates = state.index
# aggregate = state.mean(axis=1)
#
# # PLOT
# plt.figure(figsize=(14, 6))
# plt.plot(dates, aggregate, label="Aggregate state", alpha=0.7)
#
# # corr
# for i, d in enumerate(corr["date"]):
#     plt.axvline(d, color="black", linestyle="--", alpha=0.8,
#                 label="corr" if i == 0 else "")
#
# # market
# for i, d in enumerate(market["date"]):
#     plt.axvline(d, color="red", linestyle="-", alpha=0.6,
#                 label="market" if i == 0 else "")
#
# # market+disp
# for i, d in enumerate(market_disp["date"]):
#     plt.axvline(d, color="green", linestyle=":", alpha=0.6,
#                 label="market+disp" if i == 0 else "")
#
# # combined
# for i, d in enumerate(combined["date"]):
#     plt.axvline(d, color="green", linestyle="-.", alpha=0.7,
#                 label="combined" if i == 0 else "")
#
# plt.legend()
# plt.title("Breakpoint comparison across models")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("../../assets/plots/model_comparison_overlay.png")
# plt.show()

dates = state.index
aggregate = state.mean(axis=1)

fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True, sharey=False)
# breakpoints
models = [("corr", corr, "black", "--", "corr"),
    ("market", market, "red", "-", "market"),
    ("market+disp", market_disp, "green", ":", "market+disp"),
    ("combined", combined, "green", "-.", "combined")]

for ax, (title, data, color, linestyle, label) in zip(axes.flat, models):
    ax.plot(dates, aggregate, label="Aggregate state", alpha=0.7, color="blue")

    for i, d in enumerate(data["date"]):
        ax.axvline(d, color=color, linestyle=linestyle, alpha=0.8,
                   label=label if i == 0 else "")


    ax.legend()
    ax.grid(alpha=0.3)


# S&P 500
ax_sp = axes[4]
ax_sp.plot(df_sp.index, df_sp['Close'], label='S&P 500 (Close)', color='#1f77b4', linewidth=1.5)
ax_sp.set_xlabel('Date')
ax_sp.set_ylabel('Price (USD)')
ax_sp.grid(True, linestyle='--', alpha=0.6)
ax_sp.legend(loc='upper left')

ax_sp.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_sp.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax_sp.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()


plt.savefig("../../assets/plots/breakpoints_with_sp500.png")
plt.show()
