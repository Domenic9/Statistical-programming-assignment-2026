import pandas as pd
import numpy as np

# 1. 
# Load training data

X_train = pd.read_csv("Assignment/Aufgaben in Python/data_assignment/X_train.csv", index_col=0)

y_train = pd.read_csv("Assignment/Aufgaben in Python/data_assignment/y_train.csv", index_col=0)

print("Data loaded successfully")


# Number of observations and features
print("\nShape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

n_obs, n_features = X_train.shape

print("\nNumber of observations:", n_obs)
print("Number of features:", n_features)



# Summary statistics
print("\nSummary statistics:")
print(X_train.describe())


# Check for outliers (min / max)
print("\nMinimum values:")
print(X_train.min())

print("\nMaximum values:")
print(X_train.max())


# Check extreme values using quantiles

print("\nQuantiles (1% and 99%):")
print(X_train.quantile([0.01, 0.99]))


# Check feature ranges (important for scaling)
feature_ranges = X_train.max() - X_train.min()

print("\nFeature ranges:")
print(feature_ranges.sort_values(ascending=False))


# Comment for scaling question

print("\nNOTE:")
print(
    "Features have very different ranges. "
    "Distance-based methods like KNN require scaling, "
    "otherwise variables with large ranges dominate the distance.")


# 2.
# Logistic regression for default prediction

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Convert y to 1D array
y_train_vec = y_train.squeeze()

# Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# Fit logistic regression
logit = LogisticRegression(max_iter=1000)

logit.fit(X_train_scaled, y_train_vec)


# Training accuracy
y_pred_train = logit.predict(X_train_scaled)

train_acc = accuracy_score(y_train_vec, y_pred_train)

print("\nTraining accuracy:", train_acc)


# Coefficients
coef = logit.coef_[0]

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coef": coef,
    "abs_coef": np.abs(coef)
})

coef_df = coef_df.sort_values("abs_coef", ascending=False)

print("\nTop coefficients:")
print(coef_df.head(10))


# Top 3 features
top3 = coef_df.head(3)

print("\nTop 3 features with largest absolute coefficients:")
print(top3)


# 3.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# Use scaled data from before
X = X_train_scaled
y = y_train_vec


k_values = [5, 20, 50]
weight_types = ["uniform", "distance"]


results = []


for k in k_values:
    for w in weight_types:

        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights=w
        )

        scores = cross_val_score(
            knn,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

        mean_acc = scores.mean()
        error_rate = 1 - mean_acc

        results.append({
            "k": k,
            "weights": w,
            "accuracy": mean_acc,
            "error_rate": error_rate
        })


results_df = pd.DataFrame(results)

print("\nKNN results:")
print(results_df)


# Best setting

best_row = results_df.sort_values("accuracy", ascending=False).iloc[0]

print("\nBest setting:")
print(best_row)


# 4.
# Support Vector Machine

from sklearn.svm import SVC

# Use scaled data
X = X_train_scaled
y = y_train_vec

svm_results = []

# (a) Linear kernel
svm_linear = SVC(kernel="linear")

linear_scores = cross_val_score(
    svm_linear,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

svm_results.append({
    "model": "SVM_linear",
    "C": None,
    "accuracy": linear_scores.mean(),
    "error_rate": 1 - linear_scores.mean()
})

# (b) RBF kernel with different C
C_values = [0.1, 1, 10]

for c in C_values:

    svm_rbf = SVC(kernel="rbf", C=c)

    rbf_scores = cross_val_score(
        svm_rbf,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    svm_results.append({
        "model": "SVM_rbf",
        "C": c,
        "accuracy": rbf_scores.mean(),
        "error_rate": 1 - rbf_scores.mean()
    })

svm_results_df = pd.DataFrame(svm_results)

print("\nSVM results:")
print(svm_results_df)

# Best SVM setting
best_svm = svm_results_df.sort_values("accuracy", ascending=False).iloc[0]

print("\nBest SVM setting:")
print(best_svm)

# Compare with logistic regression
logit_acc = train_acc

print("\nComparison with Logistic Regression:")
print("Logistic Regression training accuracy:", logit_acc)
print("Best SVM CV accuracy:", best_svm["accuracy"])


# 5.
# Decision boundaries: Logistic Regression vs SVM (RBF)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Select two features
selected_features = ["debt_to_income", "credit_utilization"]

X_two = X_train[selected_features].copy()
y_two = y_train_vec.copy()

# Standardize the two features
scaler_two = StandardScaler()
X_two_scaled = scaler_two.fit_transform(X_two)

# Fit models
logit_2d = LogisticRegression(max_iter=1000)
logit_2d.fit(X_two_scaled, y_two)

svm_rbf_2d = SVC(kernel="rbf", C=1)
svm_rbf_2d.fit(X_two_scaled, y_two)

# Create mesh grid
x_min, x_max = X_two_scaled[:, 0].min() - 1, X_two_scaled[:, 0].max() + 1
y_min, y_max = X_two_scaled[:, 1].min() - 1, X_two_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# Predict on grid
Z_logit = logit_2d.predict(grid).reshape(xx.shape)
Z_svm = svm_rbf_2d.predict(grid).reshape(xx.shape)

# Plot decision boundaries
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Logistic Regression plot
axes[0].contourf(xx, yy, Z_logit, alpha=0.3)
axes[0].scatter(
    X_two_scaled[:, 0],
    X_two_scaled[:, 1],
    c=y_two,
    alpha=0.5
)
axes[0].set_title("Logistic Regression Decision Boundary")
axes[0].set_xlabel(selected_features[0] + " (scaled)")
axes[0].set_ylabel(selected_features[1] + " (scaled)")

# SVM RBF plot
axes[1].contourf(xx, yy, Z_svm, alpha=0.3)
axes[1].scatter(
    X_two_scaled[:, 0],
    X_two_scaled[:, 1],
    c=y_two,
    alpha=0.5
)
axes[1].set_title("SVM (RBF Kernel) Decision Boundary")
axes[1].set_xlabel(selected_features[0] + " (scaled)")
axes[1].set_ylabel(selected_features[1] + " (scaled)")

plt.tight_layout()
plt.show()


# 6.
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


X = X_train
y = y_train_vec


rf_results = []

n_estimators_list = [100, 300]
max_depth_list = [3, 5, 10]


for n in n_estimators_list:
    for d in max_depth_list:

        rf = RandomForestClassifier(
            n_estimators=n,
            max_depth=d,
            random_state=42
        )

        scores = cross_val_score(
            rf,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

        rf_results.append({
            "n_estimators": n,
            "max_depth": d,
            "accuracy": scores.mean(),
            "error_rate": 1 - scores.mean()
        })


rf_results_df = pd.DataFrame(rf_results)

print("\nRandom Forest results:")
print(rf_results_df)


# Best RF model
best_rf_row = rf_results_df.sort_values(
    "accuracy",
    ascending=False
).iloc[0]

print("\nBest RF setting:")
print(best_rf_row)


# Fit best model again
best_rf = RandomForestClassifier(
    n_estimators=int(best_rf_row["n_estimators"]),
    max_depth=int(best_rf_row["max_depth"]),
    random_state=42
)

best_rf.fit(X, y)


# Feature importances
importances = best_rf.feature_importances_

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\nTop 5 important features (Random Forest):")
print(importance_df.head(5))


# 7.
# Compare all models
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


X_scaled = X_train_scaled
X_raw = X_train
y = y_train_vec


results = []

# Logistic Regression
logit = LogisticRegression(max_iter=1000)

scores = cross_val_score(
    logit,
    X_scaled,
    y,
    cv=5,
    scoring="accuracy"
)

results.append({
    "model": "Logistic",
    "accuracy": scores.mean()
})

# KNN (best)
knn = KNeighborsClassifier(
    n_neighbors=20,
    weights="distance"
)

scores = cross_val_score(
    knn,
    X_scaled,
    y,
    cv=5,
    scoring="accuracy"
)

results.append({
    "model": "KNN",
    "accuracy": scores.mean()
})


# SVM best
svm = SVC(
    kernel="rbf",
    C=1
)

scores = cross_val_score(
    svm,
    X_scaled,
    y,
    cv=5,
    scoring="accuracy"
)

results.append({
    "model": "SVM_RBF",
    "accuracy": scores.mean()
})


# Random Forest best
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

scores = cross_val_score(
    rf,
    X_raw,
    y,
    cv=5,
    scoring="accuracy"
)

results.append({
    "model": "RandomForest",
    "accuracy": scores.mean()
})

results_df = pd.DataFrame(results)

print("\nModel comparison:")
print(results_df.sort_values("accuracy", ascending=False))



# 8.
# Predict default probabilities for new customers
from sklearn.svm import SVC

# Load test data
X_test = pd.read_csv(
    "Assignment/Aufgaben in Python/data_assignment/X_test.csv",
    index_col=0
)

print("X_test shape:", X_test.shape)

# Use best model (SVM RBF)
best_model = SVC(
    kernel="rbf",
    C=1,
    probability=True,   # needed for predict_proba
    random_state=42
)


# Scale data (same scaler as before)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit on all training data
best_model.fit(X_train_scaled, y_train_vec)

# Predict probabilities
probs = best_model.predict_proba(X_test_scaled)[:, 1]

# Create submission file
pred_df = pd.DataFrame({
    "id": np.arange(len(probs)),
    "probability": probs
})


print(pred_df.head())

# Save CSV

pred_df.to_csv(
    "predictions.csv",
    index=False
)

print("predictions.csv saved")