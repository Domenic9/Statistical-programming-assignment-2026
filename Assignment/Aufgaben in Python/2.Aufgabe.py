import pandas as pd
import numpy as np
import re

# Load Airbnb data
df = pd.read_csv("Assignment/Aufgaben in Python/data_assignment/listings_berlin.csv")

print("Original dataset shape:", df.shape)


# Cleaning functions (from Section 1)
def clean_price(price_str):

    if pd.isna(price_str):
        return None

    price_str = str(price_str).strip()
    price_str = price_str.replace("$", "")
    price_str = price_str.replace(",", "")

    if price_str == "":
        return None

    price = float(price_str)

    if price == 0:
        return None

    return price


def extract_bathrooms(text):

    if pd.isna(text):
        return None

    text = str(text).lower().strip()

    if "half" in text:
        return 0.5

    match = re.search(r"\d+\.?\d*", text)

    if match:
        return float(match.group())

    return None


# Basic cleaning
df["price_clean"] = df["price"].apply(clean_price)
df["bathrooms"] = df["bathrooms_text"].apply(extract_bathrooms)

df["host_is_superhost"] = df["host_is_superhost"].map({"t": True, "f": False})
df["instant_bookable"] = df["instant_bookable"].map({"t": True, "f": False})

df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
df["host_tenure_years"] = 2025 - df["host_since"].dt.year

# reviews_per_month: missing often means no reviews
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)


# Create target variable log(price)
df = df[df["price_clean"].notna()].copy()
df = df[df["price_clean"] > 0].copy()

df["log_price"] = np.log(df["price_clean"])

print("Cleaned dataset shape:", df.shape)


# Select predictors
predictor_cols = [
    "accommodates",
    "bedrooms",
    "bathrooms",
    "host_tenure_years",
    "host_listings_count",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "availability_365",
    "host_is_superhost",
    "instant_bookable",
    "room_type"
]

reg_df = df[["log_price"] + predictor_cols].copy()

# Remove rows with missing predictor values
reg_df = reg_df.dropna()

print("Regression dataset shape:", reg_df.shape)


# Dummy coding for room_type
reg_df = pd.get_dummies(reg_df, columns=["room_type"], drop_first=True)

print("Dataset after dummy coding:", reg_df.shape)


# Define X and y
y = reg_df["log_price"]
X = reg_df.drop(columns=["log_price"])

# Convert booleans to integers
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

print("\nPredictor variables:")
print(X.columns)

print("\nSample preview:")
print(reg_df.head())

# 2.
import statsmodels.api as sm

# Define y and X
y = reg_df["log_price"].astype(float)
X = reg_df.drop(columns=["log_price"]).copy()

# Convert boolean columns to integers
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

# Convert all predictors to float
X = X.astype(float)

# Add constant
X_ols = sm.add_constant(X)

# Fit OLS model
ols_model = sm.OLS(y, X_ols).fit()

print(ols_model.summary())


# 3. 
import matplotlib.pyplot as plt

# Diagnostic plots

fitted_vals = ols_model.fittedvalues
residuals = ols_model.resid

#Residuals vs Fitted
plt.figure(figsize=(8,5))

plt.scatter(fitted_vals, residuals, alpha=0.3)

plt.axhline(0)

plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")

plt.show()

# true vs Predicted
plt.figure(figsize=(8,5))

plt.scatter(y, fitted_vals, alpha=0.3)

plt.plot([y.min(), y.max()], [y.min(), y.max()])

plt.xlabel("True log(price)")
plt.ylabel("Predicted log(price)")
plt.title("True vs Predicted Values")

plt.show()


# 4. 

# Ridge regression function

def ridge_estimate(X, y, lam):

    X = X.values
    y = y.values.reshape(-1,1)

    n_features = X.shape[1]

    I = np.eye(n_features)

    beta = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y
    
    return beta

# Ridge estimates for different lambda values

lambdas = [0, 1, 10, 100]
ridge_results = {}

for lam in lambdas:

    beta = ridge_estimate(X, y, lam)

    ridge_results[lam] = beta

    print("\nLambda =", lam)
    print(beta.flatten())

# Ridge cross-validation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

lambdas = np.logspace(-2, 3, 30)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_errors = []

for lam in lambdas:

    fold_errors = []

    for train_idx, val_idx in kf.split(X):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        beta = ridge_estimate(X_train, y_train, lam)

        X_val_np = X_val.values
        y_pred = X_val_np @ beta

        mse = mean_squared_error(y_val, y_pred)

        fold_errors.append(mse)

    cv_errors.append(np.mean(fold_errors))

# Best lambda
best_lambda = lambdas[np.argmin(cv_errors)]

print("\nBest lambda:", best_lambda)

plt.figure(figsize=(8,5))

plt.plot(lambdas, cv_errors)
plt.axvline(best_lambda, linestyle="--")

plt.xscale("log")

plt.xlabel("Lambda")
plt.ylabel("CV Error")
plt.title("Ridge Cross Validation")

plt.show()


# 5. 
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# LASSO with cross-validation

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit LASSO with CV
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_scaled, y)

print("Best alpha", lasso_cv.alpha_)

# Watch Coefficients
lasso_coefs = pd.Series(lasso_cv.coef_, index=X.columns)

print("\nLASSO coefficients:")
print(lasso_coefs)

print("\nVariables with zero coefficients:")
print(lasso_coefs[lasso_coefs == 0])

print("\nVariables retained by LASSO:")
print(lasso_coefs[lasso_coefs !=0])



from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Task 6: Compare OLS, Ridge, and LASSO using 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

ols_rmse_list = []
ridge_rmse_list = []
lasso_rmse_list = []

best_alpha = lasso_cv.alpha_

for train_idx, test_idx in kf.split(X):

    # Split data
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    # 1. OLS
    ols = LinearRegression()

    ols.fit(X_train, y_train)

    y_pred_ols = ols.predict(X_test)

    ols_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ols))

    ols_rmse_list.append(ols_rmse)

    # 2. Ridge
    scaler_ridge = StandardScaler()

    X_train_ridge = scaler_ridge.fit_transform(X_train)
    X_test_ridge = scaler_ridge.transform(X_test)

    ridge = Ridge(alpha=best_lambda)

    ridge.fit(X_train_ridge, y_train)

    y_pred_ridge = ridge.predict(X_test_ridge)

    ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

    ridge_rmse_list.append(ridge_rmse)

    # 3. LASSO
    scaler_lasso = StandardScaler()

    X_train_lasso = scaler_lasso.fit_transform(X_train)
    X_test_lasso = scaler_lasso.transform(X_test)

    lasso = Lasso(alpha=best_alpha, max_iter=10000)

    lasso.fit(X_train_lasso, y_train)

    y_pred_lasso = lasso.predict(X_test_lasso)

    lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

    lasso_rmse_list.append(lasso_rmse)

# Average RMSE across folds
results_df = pd.DataFrame({
    "Model": ["OLS", "Ridge", "LASSO"],
    "Mean_RMSE": [
        np.mean(ols_rmse_list),
        np.mean(ridge_rmse_list),
        np.mean(lasso_rmse_list)
    ]
})

results_df = results_df.sort_values("Mean_RMSE")

print("\nOut-of-sample RMSE comparison:")
print(results_df)


print("\nDetailed RMSE by model:")
print("OLS RMSEs:", ols_rmse_list)
print("Ridge RMSEs:", ridge_rmse_list)
print("LASSO RMSEs:", lasso_rmse_list)


print("\nBest performing model:")
print(results_df.iloc[0])


# 8. LAD regression

# Estimate LAD regression (median regression)
lad_model = sm.QuantReg(y, X_ols).fit(q=0.5)

print("\nLAD Regression Results:")
print(lad_model.summary())

# COmpare OLS and LAD coefficients
comparison_df = pd.DataFrame({
    "OLS": ols_model.params,
    "LAD": lad_model.params
})

# Difference between coefficients
comparison_df["Difference"] = comparison_df["LAD"] - comparison_df["OLS"]

print("\nOLS vs LAD coefficient comparison:")
print(comparison_df)

#Show largest coefficient changes
largest_changes = comparison_df.reindex(
    comparison_df["Difference"].abs().sort_values(ascending=False).index
)

print("\nVariables with largest coefficient differences:")
print(largest_changes.head(10))


# 9. Robust inference for one selected coefficient

# Re-estimate OLS with HC3 robust standard errors
ols_hc3 = ols_model.get_robustcov_results(cov_type="HC3")

# Choose coefficient of interest
coef_name = "accommodates"

# Get position of coefficient
coef_idx = list(X_ols.columns).index(coef_name)

# Extract HC3-robust results
beta_hat = ols_hc3.params[coef_idx]
se_hc3 = ols_hc3.bse[coef_idx]
t_hc3 = ols_hc3.tvalues[coef_idx]
p_hc3 = ols_hc3.pvalues[coef_idx]

# 95% confidence interval
ci_low, ci_high = ols_hc3.conf_int()[coef_idx]

# COnvert log-point estimate to approximate percentage effect
pct_effect = np.exp(beta_hat) - 1

print(f"\nSelected coefficient: {coef_name}")
print("HC3 robust coefficient:", beta_hat)
print("HC3 robust standard error:", se_hc3)
print("HC3 robust t-statistic:", t_hc3)
print("HC3 robust p-value:", p_hc3)
print("95% HC3 confidence interval:", (ci_low, ci_high))
print("Approximate percentage effect:", pct_effect)