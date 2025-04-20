import pandas as pd

# Open the csv.gz folder and check the full data.
df = pd.read_csv('tox21.csv.gz', nrows=1000)
print(df.head())

# We want only rows (molecules) with SR-MMP values 0.0 or 1.0, not NaN.
df_filtered = df[['smiles', 'SR-MMP']].dropna()

# Rename the column SR-MMP to 'toxicity'.
df_filtered.columns = ['smiles', 'toxicity']

# Convert toxicity values from float to integer.
df_filtered['toxicity'] = df_filtered['toxicity'].astype(int)

# Save the cleaned data.
df_filtered.to_csv('tox21_sr-mmp.csv', index=False)

df2 = pd.read_csv('tox21_sr-mmp.csv', nrows=1000)
print(df2.head())

# With this, we now have a clean, ready-to-go CSV file with molecule SMILES strings and binary toxicity labels.