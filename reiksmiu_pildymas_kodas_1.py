import pandas as pd
import numpy as np

df = pd.read_csv("EKG_1500.csv", sep=';', na_values=['', 'NA', 'NaN'])
label_col = 'label' if 'label' in df.columns else None
y = df[label_col] if label_col else None

for c in df.columns:
    if c != label_col:
        df[c] = pd.to_numeric(df[c], errors='coerce')

rr_base = [c for c in df.columns if (c.startswith('RR_l_') or c.startswith('RR_r_')) and '/' not in c]
for c in rr_base:
    df[c] = df[c].mask(df[c] <= 0)

def recompute_ratio(num_col, den_col, ratio_col):
    if all(col in df.columns for col in [num_col, den_col, ratio_col]):
        m = df[ratio_col].isna() & df[num_col].notna() & df[den_col].notna() & (df[den_col] != 0)
        df.loc[m, ratio_col] = df.loc[m, num_col] / df.loc[m, den_col]

for i in range(4):
    recompute_ratio(f'RR_l_{i}', f'RR_l_{i+1}', f'RR_l_{i}/RR_l_{i+1}')
    recompute_ratio(f'RR_r_{i}', f'RR_r_{i+1}', f'RR_r_{i}/RR_r_{i+1}')

df.replace([np.inf, -np.inf], np.nan, inplace=True)

num_cols = [c for c in df.columns if c != label_col]
med = df[num_cols].median(numeric_only=True)
df[num_cols] = df[num_cols].fillna(med)
bounded = [c for c in df.columns if c.endswith('_pos') or c.endswith('_val')]
for c in bounded:
    df[c] = df[c].clip(0, 1)

if label_col:
    df[label_col] = y

    befor = len(df)
    df = df.dropna(subset=[label_col])
    afterr = len(df)
    print(f"Ištrinta eilučių be label reikšmės: {befor - afterr}")

missing_by_col = df.isna().sum()
total_missing = int(missing_by_col.sum())
print("Praleistų reikšmių pagal stulpelį:\n", missing_by_col)
print("\nViso praleistų reikšmių:", total_missing)


out_path = "EKG_1500_filled.csv"
df.to_csv(out_path, index=False, sep=';')
print(f"\nIšsaugota: {out_path}")
