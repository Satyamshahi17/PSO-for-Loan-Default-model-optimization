# 📊 Loan Default Prediction — PSO-Optimized Models

## Overview  
This repository contains a ML pipeline to predict loan defaults using multiple classification algorithms — optimized with Particle Swarm Optimization (PSO). The project demonstrates how hyperparameter optimization combined with boosted/tree-based models and classical classifiers can improve credit-risk prediction accuracy.  

It includes notebooks for different models and allows comparison across algorithms, along with performance evaluation and insights on feature importance.  

Each notebook covers: data preprocessing, handling class imbalance, PSO hyperparameter tuning, model training, evaluation, and model comparison.

---

## 🎯 Goal / Problem Statement  

Lenders and financial institutions need robust credit-risk models to identify potential defaulters and minimize losses. Predicting loan defaults with high accuracy helps:  
- Reduce financial risk and non-performing loans  
- Improve lending decision-making  
- Automate credit-scoring with data-driven insights  

This project aims to build a reproducible, hyperparameter-optimized ML pipeline for loan default prediction, comparing different algorithms under the same preprocessing and optimization framework.

---
## 🗂️ Dataset Used

This project uses the **UCI Lending Club Loan Dataset (30,000 records)** — a widely used benchmark dataset for credit-risk modeling and loan default prediction.

### 📌 Dataset Details
- **Source:** UCI Machine Learning Repository  
- **Dataset Name:** Lending Club Loan Data  
- **Size:** ~30,000 loan applications  
- **Type:** Binary classification (Default vs Non-Default)  
- **Domain:** Financial risk assessment / Credit scoring
  
## 🧠 Methods & Models  

- **Hyperparameter Optimization:**  
  - Particle Swarm Optimization (PSO) for tuning model hyperparameters  

- **Data Balancing:**  
  - **SMOTE (Synthetic Minority Oversampling Technique)** used to address class imbalance by generating synthetic samples for the minority (defaulter) class  

- **Feature Selection:**  
  - **Filter Method:** Correlation and statistical tests for removing irrelevant or redundant features  
  - **Wrapper Method:** PSO-driven model-based feature selection to identify the optimal subset of predictive variables  

- **Models Used:**  
  - **XGBoost**  
  - **LightGBM**  
  - **Support Vector Machine (SVM)**  
  - **Decision Tree Classifier (DTC)**  

# 📈 Model Results (PSO-Optimized)

The tables below summarize the evaluation results for each PSO-optimized model.  
The reported metrics correspond to the majority class (non-defaulters).  
Detailed class-wise performance, including precision and recall for the minority defaulter class, is provided within the respective Jupyter notebooks.

---

## 🔶 1. PSO-Optimized XGBoost

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.80 |
| **Precision** | 0.85 |
| **Recall** | 0.90 |
| **F1-Score** | 0.88 |

### 🔍 Notes:
- Very strong recall (best at identifying non - defaulters)
- Balanced performance across all metrics
- Reliable for risk-sensitive tasks

---

## 🔷 2. PSO-Optimized LightGBM

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.81 |
| **Precision** | 0.85 |
| **Recall** | 0.91 |
| **F1-Score** | 0.88 |

### 🔍 Notes:
- **Highest accuracy among all models**
- Best recall (0.91) → strongest at identifying non-defaulters
- Fastest training time among boosting models
- Performs best overall

---

## 🟩 3. PSO-Optimized Decision Tree (DTC)

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.79 |
| **Precision** | 0.86 |
| **Recall** | 0.87 |
| **F1-Score** | 0.87 |

### 🔍 Notes:
- Simple and interpretable
- Good baseline performance
- Slightly lower generalization than boosting models

---

## ⚫ 4. PSO-Optimized SVM

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.77 |
| **Precision** | 0.88 |
| **Recall** | 0.82 |
| **F1-Score** | 0.85 |

### 🔍 Notes:
- Highest precision (0.88)
- Lower recall → misses more non - defaulters than boosting models
- Performs well on smaller datasets

---

# 🏆 Model Comparison Summary

| Model | Accuracy | Precision | Recall | F1-Score | Best Attribute |
|-------|----------|-----------|--------|-----------|----------------|
| **LightGBM (PSO)** | **0.81** | 0.85 | **0.91** | 0.88 | Best overall |
| **XGBoost (PSO)** | 0.80 | 0.85 | 0.90 | 0.88 | Stable & high recall |
| **Decision Tree (PSO)** | 0.79 | 0.86 | 0.87 | 0.87 | Simple & interpretable |
| **SVM (PSO)** | 0.77 | **0.88** | 0.82 | 0.85 | Highest precision |

---

## ✔️ Final Insight

**PSO-optimized LightGBM is the best-performing model**  
due to:
- highest accuracy (0.81)
- highest recall (0.91)
- highest F1-score (0.88)
- efficient training

This makes it the strongest choice for real-world credit-risk prediction where identifying defaulters early is critical.

---
## 🔧 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- LightGBM  
- XGBoost  
- Matplotlib / Seaborn  
- PSO implementation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/Satyamshahi17/PSO-for-Loan-Default-model-optimization.git

## Author
Satyam Kumar  
CSE undergrad
