import os
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

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

    # After training your model:
    clf.fit(X_train, y_train)

    # Get the directory where THIS script lives (scripts/).
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create ../model/ path relative to this script.
    model_dir = os.path.join(script_dir, '..', 'model')
    os.makedirs(model_dir, exist_ok=True)

    # Save model.
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    joblib.dump(clf, model_path)

    # Save descriptor names.
    descriptor_path = os.path.join(model_dir, 'descriptor_names.txt')
    with open(descriptor_path, 'w') as f:
        f.write('\n'.join(X.columns))

    print("Model and descriptors saved to:", model_path)
 		
if __name__ == "__main__":
    main()