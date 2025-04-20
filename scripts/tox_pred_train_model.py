import os
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def main():
    # 1. Load your cleaned dataset from preprocess.py.

    # Dynamically find the path relative to this script
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '..', 'data', 'tox21_sr-mmp.csv')

    df = pd.read_csv(data_path)

    # Convert SMILES to RDKit Mol objects.
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df[df['mol'].notnull()].reset_index(drop=True)  # Drop failed parses/invalid molecules.

    # Calculate Mordred descriptors.
    calc = Calculator(descriptors, ignore_3D=True)
    X = calc.pandas(df['mol'])

    # Clean descriptor matrix (remove NaN/infinite column values).
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    # Labels.
    y = df['toxicity'].values

    # Train/test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model.
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate.
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.3f}")
 		
if __name__ == "__main__":
    main()