# Customer Churn Prediction (CodSoft ML Internship - Task 3)

A machine learning project that predicts whether a bank customer will **leave the bank (churn)** or stay,
using customer demographics and account behaviour.

## Dataset
`Churn_Modelling.csv` - 10,000 bank customers with 14 columns.

| Column group | Columns |
|---|---|
| Demographics | Geography, Gender, Age |
| Account behaviour | CreditScore, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary |
| Target | `Exited` (1 = customer left, 0 = stayed) |

About 20.4% of the customers churned, and there are no missing values.
`RowNumber`, `CustomerId` and `Surname` are identifiers, so they were dropped.

## Approach
1. Drop identifier columns and one-hot encode `Geography` and `Gender`
2. Stratified 80/20 train/test split (8,000 train, 2,000 test) so both sets keep the same churn ratio
3. Train three models: Logistic Regression (with scaling), Random Forest, Gradient Boosting
4. Handle the imbalance (only 1 in 5 customers churns) with `class_weight="balanced"` for Logistic Regression and Random Forest
5. Evaluate with accuracy, precision, recall, F1-score and ROC-AUC on the churn class, and check feature importance

## Results
Evaluated on 2,000 test customers (407 of them churned):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7135 | 0.3872 | 0.7002 | 0.4987 | 0.7771 |
| **Random Forest** | 0.8340 | 0.5776 | 0.6855 | **0.6270** | 0.8643 |
| Gradient Boosting | 0.8720 | 0.7984 | 0.4963 | 0.6121 | 0.8709 |

**Best model by F1-score: Random Forest**

```
              precision    recall  f1-score   support

      stayed       0.92      0.87      0.89      1593
     churned       0.58      0.69      0.63       407

    accuracy                           0.83      2000
   macro avg       0.75      0.78      0.76      2000
weighted avg       0.85      0.83      0.84      2000
```

## Most Important Features
| Feature | Importance |
|---|---|
| Age | 0.334 |
| NumOfProducts | 0.213 |
| Balance | 0.117 |
| EstimatedSalary | 0.072 |
| CreditScore | 0.071 |
| Geography_Germany | 0.055 |

## Key Observations
- Age is the strongest signal: the churn rate is 8% for customers up to 30, 34% for ages 41-50 and 56% for ages 51-60.
- The number of products matters: customers with 2 products rarely leave (7.6%), while 1 product (27.7%),
  3 products (82.7%) and 4 products (100%) are far riskier.
- Gradient Boosting has the highest accuracy (87.2%) and ROC-AUC, but it only catches about half of the churners (recall 0.50).
  Random Forest catches 69% of churners, so it was chosen as the best model for a retention team that wants to find at-risk customers.
- Logistic Regression is the weakest because churn depends on non-linear patterns.
- There is a trade-off between precision and recall: catching more churners means more false alarms.

## Possible Improvements
- Tune the decision threshold depending on the cost of losing a customer versus the cost of a retention offer
- Hyperparameter search (GridSearchCV) and models such as XGBoost
- Try SMOTE oversampling for the minority class

## How to Run
1. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Place `Churn_Modelling.csv` in the same folder as the code.
3. Run:
   ```bash
   python churn_prediction.py
   ```
   It finishes in under a minute. The results table, feature importance and sample customer predictions
   are printed in the terminal.

## Files
- `churn_prediction.py` - main code
- `Churn_Modelling.csv` - dataset
- `requirements.txt` - required libraries
- `README.md` - project documentation

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
