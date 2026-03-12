#%%
#Imports
# Standardbibliotheken
import datetime as dt
import json
import math as m
import random
from functools import reduce
from pathlib import Path

# Datenverarbeitung & Finanzen
import numpy as np
import pandas as pd
import yfinance as yf

# Visualisierung
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import PercentFormatter

# Statistik & Machine Learning (Sklearn & Statsmodels)
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Spezialisierte Tools
from tabulate import tabulate
from shapely.geometry import shape


df_listings_berlin = pd.read_csv('listings_berlin.csv')

# %%
#Section 1
#______________________________
#Task 1 & 2
# Load listings_berlin.csv into a pandas DataFrame. Report the number of observations and variables. For each column, compute the number and percentage of missing values.

df_listings_berlin.head()
df_listings_berlin.info()
df_listings_berlin.describe()

missing_count = df_listings_berlin.isnull().sum()
print(f"Number of missing values: {missing_count}")

percent_missing = df_listings_berlin.isnull().sum()/len(df_listings_berlin) * 100

print(percent_missing)

# 2. Spalten mit 0% fehlenden Werten herausfiltern
missing_pct_filtered = missing_count[missing_count > 0]

# 3. Für eine bessere Übersicht absteigend sortieren
missing_pct_filtered = missing_pct_filtered.sort_values(ascending=False)

# 4. Den Graph erstellen
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_pct_filtered.index, y=percent_missing[missing_pct_filtered.index], palette='magma')

# Layout-Anpassungen
plt.title('Prozentsatz fehlender Werte (nur Spalten mit Lücken)', fontsize=14)
plt.ylabel('Fehlend in %', fontsize=12)
plt.xlabel('Spaltenname', fontsize=12)
plt.xticks(rotation=45) # Dreht die Beschriftung, falls die Namen lang sind
plt.tight_layout()
plt.show()


# %%

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


clean_price.min()
clean_price.max()
clean_price.mean()
clean_price.median()
 
# %%

#Task 4
#bathrooms = df_listings_berlin['bathrooms_text'].str.replace('bath', '').str.replace('baths', '').str.replace(' shared', '').str.replace('s', '').astype(float)
print(df_listings_berlin['bathrooms_text'].unique())
df_listings_berlin['bathrooms_text']  = df_listings_berlin['bathrooms_text'].str.replace('shared bath','').str.replace('half bath','.5').str.replace('shared baths','').str.replace('private half-bath','0,5').str.replace('[^0-9.,]','',regex=True).replace(' ','0').replace(',','.')
df_listings_berlin['bathrooms_text'] = pd.to_numeric(df_listings_berlin['bathrooms_text'], errors='coerce').fillna(0)
print(df_listings_berlin['bathrooms_text'].head(10))

# %%

#Task 5
print(df_listings_berlin[['host_is_superhost','instant_bookable']].head())
superhost = (df_listings_berlin['host_is_superhost']).str.replace('t', 'True').str.replace('f', 'False').astype(bool)
instant_bookable = (df_listings_berlin['instant_bookable']).str.replace('t', 'True').str.replace('f', 'False').astype(bool)
print(superhost)
print(instant_bookable)
# %%


print(df_listings_berlin['host_since'])
time_since = pd.to_datetime(df_listings_berlin['host_since'])
time_stamp = pd.Timestamp('2025-12-31')
time_since = (time_stamp-time_since).dt.days/365
print(time_since)
#host_since = 2025 - pd.to_numeric(df_listings_berlin['host_tenure_years‘)]
# %%

#Task 6_1
print(clean_price.describe())
df_clean_price = df_listings_berlin[['clean_price']]
sns.displot(df_clean_price, x="clean_price", bins=1000,  binwidth=10)
plt.title('Distribution of Prices')
plt.xlabel('Price in €')

plt.ylabel('Frequency')
# %%

#Task 6_2
print(clean_price.describe())
df_clean_price = df_listings_berlin[['clean_price']]
sns.displot(df_clean_price, x='clean_price', bins=50,  binwidth=10,)
plt.title('Distribution of Prices')
plt.xlabel('Price in €')
plt.ylabel('Frequency')

plt.xscale('log')
plt.show()
# %%


#Task 6_3
df_clean_price = df_listings_berlin[['clean_price']]
sns.displot(df_clean_price, x="clean_price", kde='kde')
plt.title('Distribution of Prices')
plt.xlabel('Price in €')
plt.ylabel('Frequency')
# %%

#Task 6_4
df_clean_price = df_listings_berlin[['clean_price']]
sns.displot(df_clean_price, x="clean_price", kde='kde',log_scale=True)
plt.title('Distribution of Prices')
plt.xlabel('Price in €')
plt.ylabel('Frequency')

# %%
#Task 7
#print(df_listings_berlin['neighbourhood_cleansed'].head())
#print(df_listings_berlin['neighbourhood_cleansed'].describe())
#print(neighbourhood_counts)
df_listings_berlin['price'] = clean_price['Clean_Numeric']
neighbourhood_counts = (
    df_listings_berlin.groupby('neighbourhood_cleansed')['price']
    .agg(['median', 'count'])
    .reset_index()
)

neighbourhood_counts.columns = ['Neighborhood', 'Median Price', 'Count']
neighbourhood_counts = neighbourhood_counts.sort_values(by='Median Price', ascending=False)


header = ['Neighborhood', 'Count', 'Median Price']
neighbourhood_counts = neighbourhood_counts.sort_values(by='Median Price', ascending=False)
header = ['Neighborhood', 'Median Price (€)', 'Anzahl Inserate']
print(tabulate(neighbourhood_counts.head(5), headers=header, tablefmt='grid'))

unique_prices = df_listings_berlin['price'].unique()
print( unique_prices)
# %%

#Task 8
clean_price = df_listings_berlin['clean_price'].dropna().astype(int)
def trimmed_mean(x, pct):
    lower_bound = np.percentile(x,pct)
    upper_bound = np.percentile(x,100-pct)
    return x[(x>=lower_bound) & (x<=upper_bound)].mean()

print(trimmed_mean(clean_price, 5))
print(clean_price.mean())
# %%

#Task 9
df_listings_berlin['price'] = pd.to_numeric(df_listings_berlin['price'], errors='coerce')
df_listings_berlin['price'] = df_listings_berlin['price'].sort_values(ascending=False)
fig = px.scatter(
    df_listings_berlin, 
    x='bathrooms_text', 
    y='price', 
    color='price',
    title='Zusammenhang zwischen Anzahl der Bäder und Preis',
    labels={'bathrooms_text': 'Anzahl Badezimmer', 'price': 'Price in €'}
)
fig.update_yaxes(
    type='log',
    tick0=0,
    dtick=100,
    visible=True, 
    showticklabels=True, 
    showline=True
)
fig.show()
# %%

#Task 10
df_listings_berlin['price'] = pd.to_numeric(df_listings_berlin['price'], errors='coerce')
df_listings_berlin['price'] = df_listings_berlin['price'].sort_values(ascending=False)
fig = px.scatter(
    df_listings_berlin, 
    x='bedrooms', 
    y='price', 
    color='price',
    title='Zusammenhang zwischen Anzahl der Schlafzimmer und Preis',
    labels={'bedrooms': 'Anzahl Schlafzimmer', 'price': 'Price in €'}
)
fig.update_yaxes(
    type='log',
    tick0=0,
    dtick=100,
    visible=True, 
    showticklabels=True, 
    showline=True
)
fig.show()

# %%
df_listings_berlin['review_scores_rating'] = df_listings_berlin[['review_scores_rating']].clip(lower=1).round(0).dropna()
fig = px.box(df_listings_berlin, 
            x='review_scores_rating',
            y='review_scores_cleanliness',
            title='Boxplot der Bewertungen',
            color='review_scores_rating',
            labels={
                'review_scores_rating': 'Gesamtbewertung (gerundet)',
                'review_scores_cleanliness': 'Sauberkeits-Score (0-5)'
            }
)

fig.show()

# %%

#Task 11
with open("berlin_bezirke.geojson", "r", encoding="utf-8") as f:
    berlin_geojson = json.load(f)

geo_data_list = []
for feature in berlin_geojson["features"]:
    geo_data_list.append({
        "raw_name": feature["properties"]["BZR_NAME"],
        "geometry": feature["geometry"]
    })

def clean_names(series):
    return (series.astype(str)
            .str.replace('str.', 'straße', regex=False)
            .str.replace('Alt-', '', regex=False)
            .str.replace('Alt', '', regex=False)
            .str.replace('Süd', '', regex=False)
            .str.replace('Nord', '', regex=False)
            .str.replace('West', '', regex=False)
            .str.replace('Ost', '', regex=False)
            .str.replace('alt', '', regex=False)
            .str.replace('süd', '', regex=False)
            .str.replace('nord', '', regex=False)
            .str.replace('west', '', regex=False)
            .str.replace('ost', '', regex=False)
            .str.replace('Nördliche', '', regex=False)
            .str.replace('nördliche', '', regex=False)
            .str.replace('stadt', '', regex=False)
            .str.replace('1 - ', '', regex=False)
            .str.replace('2 - ', '', regex=False)
            .str.replace('3 - ', '', regex=False)
            .str.replace('4 - ', '', regex=False)
            .str.replace('5 - ', '', regex=False)
            .str.replace('-', '', regex=False)
            .str.replace(' ', '', regex=False)
            .str.replace('sß', 'ß', regex=False)
            .str.replace('1', 'Frohnau/Hermsdorf', regex=False)
            .str.replace('2', 'Waidmannslust/Wittenau/Lübars', regex=False)
            .str.replace('3', 'Borsigwalde/FreieScholle', regex=False)
            .str.replace('4', 'AugusteViktoriaAllee', regex=False)
            .str.replace('5', 'Tegel', regex=False)
            .str.replace('Kölln.', 'Köllnische', regex=False)
            .str.replace('f.', 'feld', regex=False)
            .str.replace('/Kantstraße', '', regex=False)
            .str.replace('Kantstraße', 'OttoSuhrAllee', regex=False)
            .str.replace('/Hessenwinkel', '', regex=False)
            .str.replace('Hessenwinkel', 'Rahnsdorf', regex=False)
            .str.replace('Neue', '', regex=False)
            .str.replace('/Flughafensee', '', regex=False)
            .str.replace('Waidmannslust/Wittenau/Lübars', '', regex=False)
            .str.replace('Frohnau/Hermsdorf', '', regex=False)
            .str.replace('/Karolinenhof/Rauchfangswerder', '', regex=False)
            .str.replace('SchlossCharlottenburg', 'SchloßCharlottenburg', regex=False)
            .str.replace('liche', '', regex=False)
            .str.strip())

df_listings_berlin['neighbourhood_cleansed'] = clean_names(df_listings_berlin['neighbourhood_cleansed'])
geo_df = pd.DataFrame(geo_data_list)
geo_df['clean_name'] = clean_names(geo_df['raw_name'])
geo_coord_map = dict(zip(geo_df['clean_name'], geo_df['geometry']))

korrekturen = {'Fridenau': 'Friedenau', 'Treptow': 'AltTreptow'}
for s, z in korrekturen.items():
    df_listings_berlin['neighbourhood_cleansed'] = df_listings_berlin['neighbourhood_cleansed'].replace(s, z, regex=True)
    geo_df['clean_name'] = geo_df['clean_name'].replace(s, z, regex=True)

set_berlin = set(df_listings_berlin['neighbourhood_cleansed'].unique())
set_geojson = set(geo_df['clean_name'].unique())
match_list = sorted(list(set_berlin.intersection(set_geojson)))

def apply_fuzzy_match(current_set, reference_matches, target_df, col_name):
    only_list = current_set - set(reference_matches)
    for unsauber in only_list:
        if len(unsauber) >= 5:
            for offiziell in reference_matches:
                if unsauber in offiziell or offiziell in unsauber:
                    target_df.loc[target_df[col_name] == unsauber, col_name] = offiziell
                    break

apply_fuzzy_match(set_berlin, match_list, df_listings_berlin, 'neighbourhood_cleansed')
apply_fuzzy_match(set_geojson, match_list, geo_df, 'clean_name')

final_berlin_names = set(df_listings_berlin['neighbourhood_cleansed'].unique())
final_geojson_names = set(geo_df['clean_name'].unique())
match_list = sorted(list(final_berlin_names.intersection(final_geojson_names)))
only_berlin = sorted(list(final_berlin_names - final_geojson_names))
only_geojson = sorted(list(final_geojson_names - final_berlin_names))

df_listings_berlin['geometry'] = df_listings_berlin['neighbourhood_cleansed'].map(geo_coord_map)

max_len = max(len(match_list), len(only_berlin), len(only_geojson))
df_Bezirke = pd.DataFrame({
    'Match': match_list + [None] * (max_len - len(match_list)),
    'Only in df_berlin': only_berlin + [None] * (max_len - len(only_berlin)),
    'Only in GeoJSON': only_geojson + [None] * (max_len - len(only_geojson))
})

for col in df_Bezirke.columns:
    df_Bezirke[col] = df_Bezirke[col].sort_values(ascending=True, na_position='last').values

print(tabulate(df_Bezirke, headers='keys', tablefmt='grid', showindex=False))  

stats_per_district = df_listings_berlin.groupby('neighbourhood_cleansed').agg(
    avg_price=('price', 'mean'),
    listing_count=('price', 'count'),
    geometry=('geometry', 'first')
).reset_index()

points_data = []
for _, row in stats_per_district.iterrows():
    if row['geometry']:
        poly = shape(row['geometry'])
        centroid = poly.centroid
        points_data.append({
            'Bezirk': row['neighbourhood_cleansed'],
            'lat': centroid.y,
            'lon': centroid.x,
            'Durchschnittspreis': round(row['avg_price'], 2),
            'Anzahl Inserate': row['listing_count']
        })

df_points = pd.DataFrame(points_data)

fig = px.scatter_mapbox(
    df_points,
    lat="lat",
    lon="lon",
    hover_name="Bezirk",
    color="Durchschnittspreis",
    size="Anzahl Inserate",
    hover_data={"lat": False, "lon": False, "Durchschnittspreis": True, "Anzahl Inserate": True},
    color_continuous_scale=px.colors.sequential.Viridis,
    zoom=9.33,
    height=600,
    title="Berlin: Durchschnittspreis und Anzahl der Inserate nach Bezirken"
)

fig.update_layout(
    mapbox_style="carto-positron",
    margin={"r":0,"t":40,"l":0,"b":0}
)

fig.show()
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

# %%
x_train = pd.read_csv("X_train.csv", index_col=0)
y_train = pd.read_csv("y_train.csv", index_col=0)
x_test = pd.read_csv("X_test.csv", index_col=0)
print(len(x_train), len(y_train), len(x_test))
print(x_train.info())
print(y_train.info())
perc_default = y_train[y_train == 1].count() / len(y_train) * 100 
print(perc_default)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Daten laden
x_train = pd.read_csv("X_train.csv", index_col=0)
y_train = pd.read_csv("y_train.csv", index_col=0)
x_test = pd.read_csv("X_test.csv", index_col=0)

# 2. Basis-Informationen
print(f"Trainingsdaten: {len(x_train)}, Testdaten: {len(x_test)}")
print(x_train.info())

# 3. Zielvariable analysieren (Default-Rate)
# .value_counts() ist hier oft einfacher
perc_default = (y_train.value_counts(normalize=True) * 100)
print(f"Verteilung Zielvariable (%):\n{perc_default}")

# 4. Feature Scaling (Wichtig für KNN, SVM, Lasso/Ridge)
scaler = StandardScaler()

# WICHTIG: Fit nur auf Trainingsdaten, dann auf beide anwenden
# So verhindern wir Data Leakage (Informationen aus Testset fließen ins Training)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Umwandlung zurück in DataFrame (optional, für bessere Übersicht)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)

print("\nFeatures nach der Skalierung (Beispiel):")
print(x_train_scaled.describe().loc[['mean', 'std', 'min', 'max'], 'annual_income':'age'])
# %%


# --- VORBEREITUNG: Daten laden ---
X_full = pd.read_csv("X_train.csv", index_col=0)
y_full = pd.read_csv("y_train.csv", index_col=0).iloc[:, 0]
X_new_customers = pd.read_csv("X_test.csv", index_col=0)

# --- ABSCHNITT 2: Logistic Regression & Standardization ---
# Warum Standardisierung? Um Koeffizienten trotz unterschiedlicher Skalen (z.B. Einkommen vs. Alter) vergleichbar zu machen.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_scaled, y_full)

train_acc_lr = accuracy_score(y_full, log_reg.predict(X_scaled))
coef_df = pd.DataFrame({'feature': X_full.columns, 'coef': log_reg.coef_[0]}).sort_values(by='coef', ascending=False)

print(f"2. LogReg Training Accuracy: {train_acc_lr:.4f}")
print("Top 3 Features (Einfluss auf Default):")
print(coef_df.iloc[np.abs(coef_df['coef']).argsort()[-3:]])

# --- ABSCHNITT 3: K-Nearest Neighbors (KNN) ---
# Vergleich von k und Gewichtung
knn_results = []
for k in [5, 20, 50]:
    for w in ['uniform', 'distance']:
        knn = KNeighborsClassifier(n_neighbors=k, weights=w)
        # Hier nutzen wir beispielhaft die skalierten Daten
        knn.fit(X_scaled, y_full)
        acc = accuracy_score(y_full, knn.predict(X_scaled))
        knn_results.append({'k': k, 'weights': w, 'accuracy': acc})

print("\n3. KNN Ergebnisse (Training):")
print(pd.DataFrame(knn_results))

# --- ABSCHNITT 4: Support Vector Machine (SVM) ---
# Linear vs. RBF Kernel
svm_lin = SVC(kernel='linear', random_state=42).fit(X_scaled, y_full)
svm_rbf = SVC(kernel='rbf', C=1.0, random_state=42).fit(X_scaled, y_full)

print(f"\n4. SVM Linear Accuracy: {accuracy_score(y_full, svm_lin.predict(X_scaled)):.4f}")
print(f"4. SVM RBF Accuracy: {accuracy_score(y_full, svm_rbf.predict(X_scaled)):.4f}")

# --- ABSCHNITT 5: Decision Boundaries (Visualisierung) ---
# Wir wählen zwei Features für den Plot: annual_income (Index 0) und credit_score (Index 4)
f1, f2 = 0, 4
X_vis = X_scaled[:, [f1, f2]]
lr_vis = LogisticRegression().fit(X_vis, y_full)
svm_vis = SVC(kernel='rbf', C=1.0).fit(X_vis, y_full)

h = .05
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for clf, ax, title in zip([lr_vis, svm_vis], [ax1, ax2], ['Logistic Regression', 'SVM (RBF)']):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    ax.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3)
    ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y_full, s=10, alpha=0.3, edgecolors='k')
    ax.set_title(title)
plt.show()

# --- ABSCHNITT 6: Random Forest ---
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_full, y_full) # RF braucht keine Skalierung

importances = pd.Series(rf.feature_importances_, index=X_full.columns).sort_values(ascending=False)
print("\n6. Random Forest Top 5 Features:")
print(importances.head(5))

# --- ABSCHNITT 7: Vergleich aller Modelle ---
# Zusammenfassung der Accuracy (hier auf Training/Full Data für den Vergleich)
print("\n7. Modellvergleich (Accuracy):")
models = {'LogReg': log_reg, 'SVM_RBF': svm_rbf, 'RF': rf}
for name, model in models.items():
    data = X_scaled if name != 'RF' else X_full
    acc = accuracy_score(y_full, model.predict(data))
    print(f"{name}: {acc:.4f}")

# --- ABSCHNITT 8: Vorhersage für neue Kunden ---

# 1. Daten laden (Wir definieren hier die Namen exakt)
X_train_all = pd.read_csv("X_train.csv", index_col=0)
y_train_all = pd.read_csv("y_train.csv", index_col=0).iloc[:, 0]
X_test_new = pd.read_csv("X_test.csv", index_col=0)

test_probs = rf.predict_proba(X_test_new)[:, 1]
results = pd.DataFrame({'customer_id': X_test_new.index, 'default_prob': test_probs})
results['prediction'] = (results['default_prob'] > 0.5).astype(int)
results.to_csv("final_predictions.csv", index=False)

print("\n8. Vorhersage für neue Kunden abgeschlossen. Datei 'final_predictions.csv' erstellt.")
# 2. Das beste Modell (Random Forest) auf allen Trainingsdaten trainieren
# Wir nutzen die Parameter, die in Schritt 6/7 am besten abgeschnitten haben
best_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
best_model.fit(X_train_all, y_train_all)

# 3. Ausfallwahrscheinlichkeiten vorhersagen (p_hat_i = P(default = 1 | Xi))
# .predict_proba() gibt ein Array mit [Wahrsch. für 0, Wahrsch. für 1] zurück.
# Wir extrahieren die zweite Spalte (Index 1) für das Ausfallrisiko.
probabilities = best_model.predict_proba(X_test_new)[:, 1]

# 4. DataFrame für den Export erstellen
# Die Spalte 'id' entspricht dem Index aus der X_test.csv
predictions_df = pd.DataFrame({
    'id': X_test_new.index,
    'probability': probabilities
})

# 5. Als predictions.csv speichern
predictions_df.to_csv("predictions.csv", index=False)

print("Abschnitt 8 abgeschlossen:")
print(f"- Modell auf {len(X_train_all)} Datensätzen trainiert.")
print(f"- Wahrscheinlichkeiten für {len(X_test_new)} neue Kunden berechnet.")
print("- Datei 'predictions.csv' wurde erfolgreich erstellt.")
print("\nErste 5 Zeilen der Vorhersagen:")
print(predictions_df.head())
# %%
