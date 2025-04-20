# Credit Card Default Prediction
This MSc AI (H9DAI) project evaluates the performance of machine learning models for prediciting credit card default using the **UCI Credit Card Default dataset**. It follows the **CRISP-DM methodology** and include comparision between multiple feature subset and models.

---

## 📊 Dataset Overview

### 1. **Taiwanese Credit Card Default Prediction**  
📌 Source: [UCI ML Repository – Credit Default](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
📂 Location: data_analysis/datasets/raw/default of credit card clients.xls

Features:
- **Demographics**: Age, Sex, Education, Marital Status  
- **Financial History**: Credit Limit, Bill Amounts, Previous Payments  
- **Behavioral Indicators**: Payment status for the last 6 months  
- **Target**: `default.payment.next.month` (0 = no, 1 = yes)
---

## Models Evaluated
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
---

## 🔍 Current Progress

### ✅ Completed:
- [x] CRISP-DM Stage 1: Business & Data Understanding
- [x] Initial exploratory data analysis
- [x] Data cleaning and preprocessing
- [x] Feature engineering & hyperparameter tuning
- [x] ML models (Random Forest, Logistic Regression)
- [x] Visualized `classification_report` as a heatmap

### 🔄 In Progress:
- [ ] Deployment in Django (user input + prediction interface)
- [ ] Documentation and evaluation report

---

## 📌 Objectives

- Predict likelihood of a customer defaulting on a credit payment.
- Understand and visualize key contributing factors.
- Create an interactive web app (with Django) for user testing and feedback.(Pending)

---

## 🔧 Setup & Requirements

Clone the Repository:

```bash
git clone https://github.com/danilot390/financial-risk-ml.git
cd financial-risk-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Note: Ensure that the datasets are stored in the correct directories as specified above.

---

## References 
	•	Yeh, I.C., & Lien, C.H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480.
	•	[UCI ML Repository – Credit Default](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Author
Danilo Angel Tito Rodriguez
MSc in Aritificial Intelligence
GitHub: @danilot390