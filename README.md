# 🧪 Toxicity Predictor (SR-MMP)

This project is a lightweight machine learning tool that predicts molecular toxicity using the SR-MMP (mitochondrial membrane potential) endpoint from the [Tox21 dataset](https://tripod.nih.gov/tox21/).

---

## 🚀 Features

- Parses SMILES strings from real chemical data
- Calculates Mordred molecular descriptors
- Trains a Random Forest model for binary toxicity classification
- Evaluates model performance with F1-score, precision, recall, and ROC AUC

---

## 🧠 Technologies

- Python 3.x
- RDKit
- Mordred
- Scikit-learn
- Pandas
- NumPy

---

## 📂 Project Structure

```
toxicity-predictor/
├── data/                      # Cleaned dataset (Tox21 SR-MMP)
│   └── tox21_sr-mmp.csv
│
├── scripts/                   # Python scripts
│   └── train_model.py         # Main training script
│
├── notebooks/                 # Optional: Jupyter notebooks
│   └── eda.ipynb
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Files and folders to ignore
```

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
