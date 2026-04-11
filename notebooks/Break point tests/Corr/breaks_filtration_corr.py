import pandas as pd


# 1. LOAD MATCHES
df = pd.read_csv("../../../data/tests/Corr/corr_breakpoint_matches.csv", parse_dates=["kcp_date"])
print(f"Total matches: {len(df)}")

# 2. FILTER BY DISTANCE
df = df[df["distance_days"] <= 3].copy()
print(f"After distance filter: {len(df)}")

# 3. SORT BY DATE
df = df.sort_values("kcp_date")

# 4. MERGE CLOSE BREAKPOINTS
final_breaks = []
current_group = [df.iloc[0]]

for i in range(1, len(df)):
    prev = current_group[-1]
    curr = df.iloc[i]

    # if the breaks are close together (≤ 30 days)
    if (curr["kcp_date"] - prev["kcp_date"]).days <= 30:
        current_group.append(curr)
    else:
        # take the average
        avg_date = pd.to_datetime([row["kcp_date"] for row in current_group]).mean()

        final_breaks.append(avg_date)
        current_group = [curr]

# last cluster
avg_date = pd.to_datetime([row["kcp_date"] for row in current_group]).mean()
final_breaks.append(avg_date)

# 5. RESULT
final_df = pd.DataFrame({"date": final_breaks})
print("\nFINAL BREAKPOINTS:")
print(final_df)

final_df.to_csv("../../../data/tests/Corr/corr_final_breakpoints.csv", index=False)