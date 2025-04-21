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
│   └── tox21_sr-mmp.csv
│   └── new_molecules.csv
│   └── predictions.csv
│
├── scripts/                   # Python scripts for pipeline stages
│   ├── preprocess_data.py     # Clean original dataset
│   ├── train_model.py         # Train and evaluate RandomForest model
│   └── predict_toxicity.py    # Predict toxicity for new SMILES
│
├── model/                     # Trained model + feature names
│   ├── random_forest_model.pkl
│   └── descriptor_names.txt
│
├── notebooks/                 # Jupyter notebooks for EDA
│   └── eda.ipynb
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── .gitignore                 # Files/folders to ignore
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

## 📊 Example Output

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

## 📜 License

This project is open source under the MIT License.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change or improve.
