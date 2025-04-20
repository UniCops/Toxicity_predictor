import os
import pandas as pd

# Get the path of this script.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input: tox21.csv.gz in ../data/
input_path = os.path.join(script_dir, '..', 'data', 'tox21.csv.gz')
df = pd.read_csv(input_path)

# Preprocess.
df_filtered = df[['smiles', 'SR-MMP']].dropna()
df_filtered.columns = ['smiles', 'toxicity']
df_filtered['toxicity'] = df_filtered['toxicity'].astype(int)

# Save output to ../data/
output_path = os.path.join(script_dir, '..', 'data', 'tox21_sr-mmp.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_filtered.to_csv(output_path, index=False)

print(f"Saved cleaned data to: {output_path}")

# Load again to preview.
df2 = pd.read_csv(output_path, nrows=1000)
print(df2.head())