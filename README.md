### Breast Cancer Detection

Breast cancer is one of the most common cancers worldwide, and early detection can make a huge difference in treatment outcomes. This project is my attempt to build a machine learning model that can help in predicting whether a tumor is benign or malignant based on medical imaging data.

I wanted to work on this not just as a coding exercise, but as something meaningful — a small step toward applying data science in healthcare, where even small improvements can have real impact.

### 📌 Project Overview

Dataset: The Breast Cancer Wisconsin dataset, available directly from sklearn.datasets.

Goal: Train and evaluate a machine learning model that predicts tumor type.

Features: Characteristics of cell nuclei (mean radius, texture, perimeter, area, smoothness, etc.).

Target:

0 → malignant

1 → benign

### ⚙️ Technologies Used

Python 3.13

Pandas for data handling

NumPy for numerical operations

Matplotlib / Seaborn for visualization

Scikit-learn for machine learning

### 🚀 Steps in the Project

Data Loading – Loaded the dataset using sklearn.datasets.load_breast_cancer().

Data Preprocessing – Created a pandas DataFrame, added target labels, and did some cleaning.

Exploratory Data Analysis (EDA) – Plotted distributions and correlations to understand the data better.

Model Building – Trained ML models such as Logistic Regression, Decision Trees, and Random Forests.

Evaluation – Compared models using accuracy, confusion matrix, precision, recall, and F1-score.

### 📊 Results

The Random Forest classifier performed the best (so far), with high accuracy on both training and test sets.

Logistic Regression also performed surprisingly well, proving that sometimes simple models work best.

### 🌱 Future Improvements

Experiment with more advanced models (e.g., SVM, XGBoost, Neural Networks).

Use hyperparameter tuning for better optimization.

Deploy the model as a simple web app so others can test it interactively.

❤️ A Note

This project is not meant to replace medical diagnosis. It’s a learning project that explores how machine learning can be applied in healthcare. I hope to keep improving it and maybe one day contribute to something that could actually help doctors and patients.
