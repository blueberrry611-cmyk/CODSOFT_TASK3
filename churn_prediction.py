import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)

RANDOM_STATE = 42

# ---------- 1. Load and clean data ----------
df = pd.read_csv("Churn_Modelling.csv")
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Churn rate: {:.1f}%".format(100 * df["Exited"].mean()))

# RowNumber, CustomerId and Surname are identifiers, they say nothing about churn
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# Turn text columns into numbers (one-hot encoding)
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True, dtype=int)

X = df.drop(columns="Exited")
y = df["Exited"]

# ---------- 2. Train / test split (stratified keeps the churn ratio equal) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print("Train:", len(X_train), " Test:", len(X_test))

# ---------- 3. Models ----------
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
}

# ---------- 4. Train and evaluate ----------
results, fitted = {}, {}
for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    fitted[name] = (model, pred, proba)
    results[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }

results_df = pd.DataFrame(results).T
print("\nModel comparison (churn class):\n", results_df.round(4))

# ---------- 5. Best model ----------
best_name = results_df["F1"].idxmax()
best_model, best_pred, best_proba = fitted[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best_pred, target_names=["stayed", "churned"]))

# Which features matter most?
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top features:\n", imp.head(6).round(3).to_string())

cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["stayed", "churned"], yticklabels=["stayed", "churned"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.show()

# ---------- 6. Demo: churn risk of a few test customers (3 who churned, 3 who stayed) ----------
import numpy as np
rng = np.random.RandomState(RANDOM_STATE)
churned_rows = rng.choice(np.where(y_test.values == 1)[0], size=3, replace=False)
stayed_rows = rng.choice(np.where(y_test.values == 0)[0], size=3, replace=False)

print("\nSample customers from the test set:")
for i in list(churned_rows) + list(stayed_rows):
    p = best_proba[i]
    actual = "CHURNED" if y_test.iloc[i] == 1 else "STAYED"
    predicted = "CHURN" if best_pred[i] == 1 else "STAY"
    row = X_test.iloc[i]
    print(f"  Age {int(row['Age'])}, {int(row['NumOfProducts'])} product(s), "
          f"active={int(row['IsActiveMember'])} -> churn probability {p:.2f} | "
          f"predicted: {predicted:5s} | actual: {actual}")
