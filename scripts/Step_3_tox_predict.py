import os
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
import joblib

def main():
    # === Get this script's folder location ===
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # === Step 1: Load model + descriptors ===
    model_path = os.path.join(script_dir, '..', 'model', 'random_forest_model.pkl')
    descriptor_path = os.path.join(script_dir, '..', 'model', 'descriptor_names.txt')

    model = joblib.load(model_path)
    with open(descriptor_path, 'r') as f:
        descriptor_names = f.read().splitlines()

    # === Step 2: Load new SMILES to predict ===
    input_path = os.path.join(script_dir, '..', 'data', 'new_molecules.csv')
    df = pd.read_csv(input_path)

    if 'smiles' not in df.columns:
        raise ValueError("The input file must contain a column named 'smiles'.")

    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df[df['mol'].notnull()].reset_index(drop=True)

    # === Step 3: Compute descriptors ===
    calc = Calculator(descriptors, ignore_3D=True)
    X_new = calc.pandas(df['mol'])

    # === Step 4: Clean and align with training features ===
    X_new = X_new.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    X_new = X_new[[col for col in descriptor_names if col in X_new.columns]]

    # Optional: sanity check.
    missing = set(descriptor_names) - set(X_new.columns)
    if missing:
        raise ValueError(f"Missing descriptors from new data: {missing}")

    # === Step 5: Predict ===
    predictions = model.predict(X_new)
    df['toxicity_prediction'] = predictions

    # === Step 6: Save predictions ===
    output_path = os.path.join(script_dir, '..', 'data', 'predictions.csv')
    df.to_csv(output_path, index=False)

    print(f"\n✅ Predictions saved to: {output_path}")
    print(df[['smiles', 'toxicity_prediction']].head())

# === Fix multiprocessing issue on Windows ===
if __name__ == "__main__":
    main()