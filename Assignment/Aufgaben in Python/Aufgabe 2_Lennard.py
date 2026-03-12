#Imports
# Standardbibliotheken

# Datenverarbeitung & Finanzen
import numpy as np
import pandas as pd


# Visualisierung
import matplotlib.pyplot as plt
import seaborn as sns


# Statistik & Machine Learning (Sklearn & Statsmodels)
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LassoCV

from sklearn.metrics import accuracy_score, classification_report

# Spezialisierte Tools



df_listings_berlin = pd.read_csv('listings_berlin.csv')
# %%
#required here from Section 1 Question 3


#Task 3
df_listings_berlin['price'] = df_listings_berlin['price'].astype(str)
df_listings_berlin.dropna(subset=['price'], inplace=True)

df_listings_berlin['clean_price'] = (
    df_listings_berlin['price']
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
)

clean_price_2 = df_listings_berlin.dropna(subset=['clean_price'])
clean_price_2 = df_listings_berlin[df_listings_berlin['clean_price'] > 0].copy()


clean_price = pd.DataFrame({
    'Original_String': df_listings_berlin['price'],
    'Clean_Numeric': clean_price_2['clean_price']
})


# %%
df_listings_berlin['bathrooms_num'] = (
    df_listings_berlin['bathrooms_text']
    .astype(str)
    .str.extract('(\d+\.?\d*)')
    .astype(float)
)

# Spalten auswählen
features = ['review_scores_rating', 'review_scores_cleanliness', 'instant_bookable', 'neighbourhood_cleansed', 'bedrooms', 'bathrooms_num']
X = df_listings_berlin[features].copy()

# Fehlende Werte füllen (Modelle mögen keine NaNs)
X = X.fillna(0)

# Kategoriale Daten umwandeln (Text -> Spalten mit 0 und 1)
X = pd.get_dummies(X, columns=['instant_bookable', 'neighbourhood_cleansed'], drop_first=True)

# 2. Zielvariable (y) vorbereiten
# HINWEIS: LogisticRegression braucht Klassen (z.B. 0 für günstig, 1 für teuer)
# Wenn df_clean_price kontinuierliche Preise sind, nutze LinearRegression oder kategorisiere y:

df_clean_price = df_listings_berlin[['clean_price']]

y = (df_clean_price > df_clean_price.median()).astype(int) # Beispiel: 1 wenn über Median-Preis

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Trainieren
model = LogisticRegression(max_iter=1000) # max_iter erhöhen für Konvergenz
model.fit(X_train, y_train)

# 5. Predict & Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Calculate sample sizes
n_total = len(X)
n_train = len(X_train)
n_test = len(X_test)

print(f"Total listings: {n_total}")
print(f"Training sample size (70%): {n_train}")
print(f"Testing sample size (30%): {n_test}")
# %%

df_model = X.copy()
df_model['target_price'] = df_clean_price['clean_price']   

# 2. NaNs und Infs entfernen
# Löscht alle Zeilen, in denen entweder in den Features oder im Preis ein NaN steht
df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

# 3. Features und Zielvariable wieder trennen
y_final = df_model['target_price']
X_final = df_model.drop(columns=['target_price'])

# 4. Konstante hinzufügen (WICHTIG für statsmodels)
X_final = sm.add_constant(X_final)

# 5. OLS Modell schätzen
ols_model = sm.OLS(y_final, X_final).fit()

# 6. Ergebnis ausgeben
print(ols_model.summary())
# %%
# 1. Generate predicted (fitted) values from the OLS model
y_fitted = ols_model.fittedvalues
residuals = ols_model.resid

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# (a) Residuals-vs-Fitted Plot
sns.scatterplot(x=y_fitted, y=residuals, ax=ax1, alpha=0.5)
ax1.axhline(y=0, color='red', linestyle='--')
ax1.set_title('Residuals vs. Fitted Values')
ax1.set_xlabel('Fitted Values (Predicted Price)')
ax1.set_ylabel('Residuals (Error)')

# (b) True-vs-Fitted Plot
sns.scatterplot(x=y_fitted, y=y_final, ax=ax2, alpha=0.5)
# Add a 45-degree line representing a perfect match
max_val = max(y_fitted.max(), y_final.max())
ax2.plot([0, max_val], [0, max_val], color='red', linestyle='--')
ax2.set_title('True vs. Fitted Values')
ax2.set_xlabel('Fitted Values (Predicted Price)')
ax2.set_ylabel('True Values (Actual Price)')

plt.tight_layout()
plt.show()
# %%
def ridge_estimate(X, y, lam):
    """
    Computes Ridge coefficients using the formula:
    beta = (X.T @ X + lam * I)^-1 @ X.T @ y
    """
    # Number of features (including constant)
    n_features = X.shape[1]
    
    # Create the Identity matrix I
    I = np.eye(n_features)
    
    # X.T @ X
    XTX = X.T @ X
    
    # (X.T @ X + lam * I)
    # Use np.linalg.inv or np.linalg.solve for better numerical stability
    term1_inv = np.linalg.inv(XTX + lam * I)
    
    # term1_inv @ X.T @ y
    beta_ridge = term1_inv @ X.T @ y
    
    return beta_ridge

# Prepare data as numpy arrays
X_np = X_final.values
y_np = y_final.values

lambdas = [0.01, 1, 100]
results = {}

for lam in lambdas:
    beta = ridge_estimate(X_np, y_np, lam)
    # Calculate L2 norm: sqrt(sum of squares of coefficients)
    l2_norm = np.linalg.norm(beta)
    results[lam] = l2_norm
    print(f"Lambda: {lam:>5} | L2-Norm: {l2_norm:.4f}")
# %%
# 1. Pipeline definieren
# Wir nutzen Pipeline(), um sicherzustellen, dass StandardScaler und LassoCV 
# in einem Objekt gekapselt sind.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lassocv', LassoCV(cv=5, random_state=42, max_iter=10000))
])

# 2. Modell anpassen
# Stelle sicher, dass X_final und y_final definiert sind (aus dem OLS-Schritt)
pipeline.fit(X_final, y_final)

# 3. Die Variablen explizit definieren, um den NameError zu vermeiden
lasso_model = pipeline.named_steps['lassocv']
best_alpha = lasso_model.alpha_

# 4. Ergebnisse ausgeben
print(f"Optimales Alpha: {best_alpha:.4f}")

# Koeffizienten prüfen
coef_series = pd.Series(lasso_model.coef_, index=X_final.columns)
print(f"Anzahl der Features mit Koeffizient > 0: {(coef_series != 0).sum()}")# %%

# %%
# 2. LAD-Regression initialisieren (QuantReg mit tau = 0.5)
# Wir nutzen y_final und X_final direkt
lad_model = sm.QuantReg(y_final, X_final)

# 3. Modell fitten
# Die Methode 'med' steht für Median-Regression
lad_results = lad_model.fit(q=0.5)

# 4. Koeffiziententabelle ausgeben
print("LAD Regression Results (Median Regression, tau=0.5):")
print(lad_results.summary())
# %%
# 1. HC3-robuste Ergebnisse berechnen
robust_results = ols_model.get_robustcov_results(cov_type='HC3')

# 2. Variable definieren und Index finden
var_name = 'review_scores_cleanliness'
if var_name in ols_model.model.exog_names:
    idx = list(ols_model.model.exog_names).index(var_name)
    
    coef_clean = robust_results.params[idx]
    std_err_clean = robust_results.bse[idx]
    t_stat_clean = robust_results.tvalues[idx]
    p_val_clean = robust_results.pvalues[idx]
    conf_int = robust_results.conf_int()[idx]

    print(f"--- Ergebnisse für: {var_name} ---")
    print(f"Coefficient (Beta): {coef_clean:.4f}")
    print(f"HC3 Robust t-statistic: {t_stat_clean:.4f}")
    print(f"HC3 Robust p-value: {p_val_clean:.4f}")
    print(f"95% HC3 Confidence Interval: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")

    # 3. Prozentualer Effekt berechnen
    # CHECK: Ist die Zielvariable log-transformiert? 
    # (Wir prüfen, ob die Mittelwerte der Zielvariable klein sind, typisch für Log-Werte wie 3.5 bis 5.0)
    y_mean = ols_model.model.endog.mean()
    
    if y_mean < 10:  # Wahrscheinlich Log-Modell (log(Price))
        percentage_effect = (np.exp(coef_clean) - 1) * 100
        print(f"Interpretation (Log-Modell): Ein Anstieg der Sauberkeit um 1 Punkt ")
        print(f"führt zu einer Preisänderung von: {percentage_effect:.2f}%")
    else:            # Wahrscheinlich Lineares Modell (Price in Euro)
        # In einem linearen Modell ist der Koeffizient direkt der Euro-Wert
        avg_price = y_mean
        percentage_effect = (coef_clean / avg_price) * 100
        print(f"Interpretation (Lineares Modell): Ein Anstieg der Sauberkeit um 1 Punkt ")
        print(f"entspricht ca. {coef_clean:.2f}€, was {percentage_effect:.2f}% des Durchschnittspreises ist.")

else:
    print(f"Variable {var_name} wurde nicht im Modell gefunden!")
# %%
#Irgendwo muss y noch gelogt werde
# 1. Zielvariable LOGARITHMIEREN
y_log = np.log(y_final) 

# 2. Modell neu fitten
ols_model_log = sm.OLS(y_log, X_final).fit()

# 3. Robusten Koeffizienten holen
robust_results = ols_model_log.get_robustcov_results(cov_type='HC3')
idx = list(ols_model_log.model.exog_names).index('review_scores_cleanliness')
coef_clean = robust_results.params[idx]

# 4. Prozentualen Effekt neu berechnen
percentage_effect = (np.exp(coef_clean) - 1) * 100
print(f"Realistischer Effekt: {percentage_effect:.2f}%")