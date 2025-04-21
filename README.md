# 🧪 Toxicity Predictor (SR-MMP)

This project is a lightweight machine learning tool that predicts molecular toxicity using the SR-MMP (mitochondrial membrane potential) endpoint from the [Tox21 dataset](https://tripod.nih.gov/tox21/).

---

## 🚀 Features

- Parses SMILES strings from real chemical data
- Calculates Mordred molecular descriptors
- Trains a Random Forest model for binary toxicity classification
- Evaluates model performance with precision, recall, F1-score, and ROC AUC
- Predicts toxicity on new molecules via SMILES input
- Provides exploratory data analysis and molecule visualization

---

## 🧠 Technologies

- Python 3.x
- RDKit
- Mordred
- Scikit-learn
- Seaborn / Matplotlib
- Pandas / NumPy
- Jupyter Notebooks

---

## 📂 Project Structure

```
toxicity-predictor/
├── data/                      # Cleaned dataset and input/output molecules
│   └── new_molecules.csv
│   └── predictions.csv
│   └── tox21_sr-mmp.csv
|   └── tox21.csv.gz (which has been preprocessed by Step_1_preprocess.py to tox21_sr-mmp)
│
├── scripts/                   # Python scripts for pipeline stages
│   ├── Step_1_preprocess.py   # Clean original dataset
│   ├── Step_2_train_model.py  # Train and evaluate RandomForest model
│   └── Step_3_tox_predict.py  # Predict toxicity for new SMILES
│
├── model/                     # Trained model + feature names
│   ├── descriptor_names.txt
│   └── random_forest_model.pkl
│
├── notebooks/                 # Jupyter notebooks for EDA
│   └── eda.ipynb
│
├── .gitattributes
├── .gitignore                 # Files/folders to ignore
├── README.md                  # Project overview
└── requirements.txt               # Python dependencies
```

---

## 🧪 Exploratory Data Analysis

All EDA is located in `notebooks/eda.ipynb`. It includes:

- Toxicity class imbalance visualization
- Distribution of molecular properties (Molecular Weight, LogP, TPSA, Rotatable Bonds)
- Paired visual comparison of toxic vs non-toxic molecules based on structural similarity
- Top 20 feature importances from trained Random Forest model

---

## 🏁 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/toxicity-predictor.git
cd toxicity-predictor
```

### 2. Set up a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate         # On Windows: .\venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the model training script
```bash
python scripts/train_model.py
```

### 5. Predict toxicity on new molecules
```bash
python scripts/predict_toxicity.py
```

---

## 📊 Example output of model training

```
              precision    recall  f1-score   support

           0       0.86      0.99      0.92       121
           1       0.89      0.29      0.43        28

    accuracy                           0.86       149
   macro avg       0.87      0.64      0.68       149
weighted avg       0.86      0.86      0.83       149

ROC AUC Score: 0.899
```

---

## Example output of predicting the toxicity of new molecules

```
smiles,mol,toxicity_prediction
CC(=O)OC1=CC=CC=C1C(=O)O,<rdkit.Chem.rdchem.Mol object at 0x00000234A8498F90>,0
CN1CCC(CC1)NC2=NC=NC3=CC=CC=C23,<rdkit.Chem.rdchem.Mol object at 0x00000234B8729690>,0
CC(C)CC(=O)O,<rdkit.Chem.rdchem.Mol object at 0x00000234B8729620>,0
CCOc1ccc(cc1OC)C(=O)O,<rdkit.Chem.rdchem.Mol object at 0x00000234B8729700>,0
```

---

## 📜 License

This project is open source under the MIT License.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change or improve.
