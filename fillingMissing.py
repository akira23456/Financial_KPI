# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python (churn_env)
#     language: python
#     name: churn_env
# ---

# %%
import pandas as pd

# ============================================
# Load merged data
# ============================================
df = pd.read_csv("merged_financials.csv")

# ============================================
# Make sure 'stock' is a column
# ============================================
if 'stock' not in df.columns and df.index.name == 'stock':
    df = df.reset_index()

# Normalize all column names
df.columns = df.columns.str.strip().str.lower()

# ============================================
# Forward-fill within each company (ROBUST FIX)
# ============================================
# Set 'stock' and 'enddate' as index for robust time-series grouping,
# forward-fill, then reset the index to turn them back into columns.
df = (
    df.set_index(['stock', 'enddate'])
      .groupby(level=0) # Group by the first level of the index ('stock')
      .ffill()
      .reset_index() # Bring 'stock' and 'enddate' back as regular columns
)

# ============================================
# Fill cash-flow-type columns with zeros
# ============================================
cashflow_cols = [
    'dividendspaid','repurchaseofstock',
    'capitalexpenditures','changetoinventory',
    'changetoaccountreceivables','changeincash'
]

for col in cashflow_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# ============================================
# Compute KPIs
# ============================================
df["current_ratio"] = (df["totalcurrentassets"] / df["totalcurrentliabilities"]).replace([pd.NA, float("inf")], 0)
df["debt_to_equity"] = (df["longtermdebt"] / df["totalstockholderequity"]).replace([pd.NA, float("inf")], 0)
df["roe"] = (df["netincome"] / df["totalstockholderequity"]).replace([pd.NA, float("inf")], 0)

# ============================================
# FILL ALL REMAINING NaNs WITH ZEROS (Added for ML readiness)
# ============================================
df = df.fillna(0)

# ============================================
# Save the final dataset
# ============================================
# Saving to a new file name to denote it's fully ready for the neural network
df.to_csv("financial_kpi.csv", index=False)

# %%
