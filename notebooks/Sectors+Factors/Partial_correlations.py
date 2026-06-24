import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# LOAD DATA
sector_factors = pd.read_csv("../../data/sectors/sector_pca_factors.csv",index_col=0,parse_dates=True)
external = pd.read_csv("../../data/raw/external_factors.csv",index_col=0,parse_dates=True)
df = sector_factors.join(external, how="inner")


# REGIMES
regimes = {"R1_2015_2016": ("2015-08-25", "2016-03-14"),
           "R2_2017_2018": ("2017-06-27", "2018-02-09"),
           "R3_2019_2020": ("2019-02-01", "2020-02-24"),
           "R4_COVID": ("2020-02-24", "2020-06-15"),
           "R5_2021": ("2021-06-14", "2021-12-01"),
           "R6_2025": ("2025-04-10", "2026-04-10"),}

regimes1 = {"R1": ("2014-01-02", "2015-08-24"),
           "R2": ("2016-03-15","2017-06-26"),
           "R3": ("2018-02-10", "2019-01-31"),
           "R4": ("2020-06-16","2021-06-13"),
           "R5": ("2021-12-02","2025-04-9"),}


# VARIABLES
external_cols = ["VIX", "WTI", "Gold", "DXY"]
sector_cols = [c for c in df.columns if c not in external_cols]


# PARTIAL CORR FUNCTION
def partial_corr(data, x, y, controls):

    tmp = data[[x, y] + controls].dropna()

    if len(tmp) < 30:
        return np.nan

    X_controls = tmp[controls]

    # residuals of x
    reg_x = LinearRegression()
    reg_x.fit(X_controls, tmp[x])

    resid_x = tmp[x] - reg_x.predict(X_controls)

    # residuals of y
    reg_y = LinearRegression()
    reg_y.fit(X_controls, tmp[y])

    resid_y = tmp[y] - reg_y.predict(X_controls)

    return np.corrcoef(resid_x, resid_y)[0, 1]



# PARTIAL CORRELATIONS
results = []

for regime_name, (start, end) in regimes.items():

    print(regime_name)

    window = df.loc[start:end]

    print("Observations:", len(window))

    for sector in sector_cols:

        for factor in external_cols:

            controls = [x for x in external_cols if x != factor]
            pcorr = partial_corr( window,sector,factor,controls)

            results.append({"regime": regime_name,
                            "sector": sector,
                            "factor": factor,
                            "partial_corr": pcorr})


# SAVE
partial_df = pd.DataFrame(results)

partial_df.to_csv("../../data/sectors/sector_partial_correlations.csv",index=False)
print("\nSaved:")
print(partial_df.shape)



# TOP RELATIONSHIPS
print("\n===== TOP PARTIAL CORRELATIONS =====")
top = (partial_df.dropna().assign(abs_corr=lambda x: x["partial_corr"].abs()).sort_values("abs_corr", ascending=False))

print(top[["regime","sector","factor","partial_corr"]].head(50))