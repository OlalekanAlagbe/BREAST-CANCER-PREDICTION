# BREAST-CANCER-PREDICTION
Breast cancer classification using Logistic Regression, Decision Tree, Random Forest, and SVM. Includes feature selection, scaling, and evaluation using accuracy, ROC-AUC, and classification reports. Built with scikit-learn and visualized in Python.

# 🧬 Breast Cancer Classification Using Logistic Regression and Other Models

This project explores the effectiveness of various machine learning algorithms in predicting whether a tumor is **malignant** or **benign**, using the **Breast Cancer Wisconsin Diagnostic dataset** provided by `sklearn.datasets`.

---

## 📦 Dataset Overview

- **Source:** `load_breast_cancer()` from scikit-learn
- **Samples:** 569
- **Features:** 30 numeric features extracted from digitized images of fine needle aspirate (FNA) of breast mass.
- **Target:** Binary classification
  - `0` = malignant
  - `1` = benign

---

## 🎯 Project Goals

- Build a classification model to predict tumor malignancy
- Perform **feature selection** by removing highly correlated features
- Apply **feature scaling** for model optimization
- Evaluate and compare the performance of four classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Use evaluation metrics like **accuracy**, **ROC-AUC**, and **classification report**

---

## 🛠️ Tools and Libraries Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🧪 Models Explored

### 1. Logistic Regression
- Used as a baseline model
- Handled multicollinearity by dropping features with correlation > 0.95
- Accuracy: ~96%
- ROC-AUC: ~0.99

### 2. Decision Tree Classifier
- Simple tree-based model
- Good interpretability
- Accuracy: ~93–95%
- ROC-AUC: ~0.96

### 3. Random Forest Classifier
- Ensemble method combining multiple decision trees
- Higher robustness and accuracy
- Accuracy: ~96–97%
- ROC-AUC: ~0.99

### 4. Support Vector Machine (SVM)
- Used with `probability=True` to allow ROC-AUC computation
- Accurate on clean and scaled datasets
- Accuracy: ~96%
- ROC-AUC: ~0.99

---

## 📈 Evaluation Metrics

Each model was evaluated using:

- **Accuracy Score**
- **Classification Report**
  - Precision
  - Recall
  - F1-score
- **ROC-AUC Score**
- Confusion matrix (available in notebooks or plots)
- Results saved to: `breast_cancer_model_comparison.txt`

---
