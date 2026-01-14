# 🩺 Breast Tissue Risk Predictor

Predicting breast tissue types and assessing breast cancer risk using machine learning.

## 🔍 Overview

This project uses a Random Forest-based ML pipeline to classify breast tissue samples from spectral and frequency features. It predicts tissue types such as:

Carcinoma (car) 🧬

Fibroadenoma (fad) 🩹

Mastopathy (mas) ⚕️

Glandular (gla) 🧪

Connective (con) 🔗

Adipose (adi) 🥓

The system helps identify tissue abnormalities and assess the risk of breast cancer, providing early insights for diagnostics.
---

## ✨ Features

Multi-class classification for 6 tissue types

Handles small, imbalanced datasets

Preprocessing pipeline with scaling for relevant models

Stratified train-test split for balanced representation

Saves model and pipeline for easy inference

Generates predictions and saves them for analysis ✅
---

## 🛠️ Usage

1️⃣ Training the model
python train_model.py


Reads data.csv

Performs preprocessing

Trains a Random Forest classifier

Saves model (model.pkl) and pipeline (pipeline.pkl)

2️⃣ Making predictions
python train_model.py


Loads saved model and pipeline

Transforms test data (Testing_data.csv)

Outputs predictions in Predictions.csv 📝
---

## 📊 Evaluation

For evaluation, see the Jupyter notebook, which includes:

Accuracy

Classification report

Confusion matrix visualization

📦 Requirements

Python 3.x

pandas 🐼

numpy 🔢

scikit-learn 🤖

matplotlib 📈

joblib 💾

Install dependencies with:

pip install -r requirements.txt
---

## 🙏 Credits

Dataset author on Kaggle: https://www.kaggle.com/datasets/tarktunataalt/breast-tissue-impedance-measurements 🌟

Original dataset citation:

S, JP and Jossinet, J. (2010). Breast Tissue. UCI Machine Learning Repository. https://doi.org/10.24432/C5P31H
 📚
---

## ⚖️ License

This project is released under the CC BY-SA 4.0 License
