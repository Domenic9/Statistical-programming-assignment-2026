# Section 1: Data Wrangling and Spatial Exploration

# 1.
# Load dataset
import pandas as pd

df = pd.read_csv("Assignment/Aufgaben in Python/data_assignment/listings_berlin.csv")

# number of observations & variables
print("Dataset shape (observations, vriables):")
print(df.shape)

# Missing Values Count
missing_values = df.isna().sum()

# Missing Value Percent
missing_percent = df.isna().mean()*100

#Summary table
missing_summary = pd.DataFrame({
    "missing_values": missing_values,
    "missing_percent": missing_percent

})

# Sort by highest missing percentage
missing_summary = missing_summary.sort_values(by="missing_percent", ascending=False)

print ("\nMissing values summary:")
print(missing_summary)

# 2. 
# Identify top 3 variables with highest missing values

top_missing = missing_summary.head(3)

print("\nTop 3 variables with highest missing values:")
print(top_missing)

# host_response_rate --> keep missing values (Nicht jeder Host gibt RR an)
# price --> drop observation (Price ist Zielvariable)
# reviews_per_month --> keep missing values (reviews fehlen häufig)


# 3.
def clean_price(price_str):

    # Missing values
    if pd.isna(price_str):
        return None
    
    # String and clean symbols
    price_str = str(price_str).strip()
    price_str = price_str.replace("$", "")
    price_str = price_str.replace(",", "")

    # Handle empty strings
    if price_str == "":
        return None
    
    # Convert to float
    price = float(price_str)

    # Handle zero prices
    if price == 0:
        return None
    
    return price

#apply function to entire column
df["price_clean"] = df["price"].apply(clean_price)

print(df[["price", "price_clean"]].head())

# Report values
print("\nPrice statistics:")
print("Min", df["price_clean"].min())
print("Mean", df["price_clean"].mean())
print("Median", df["price_clean"].median())
print("Max", df["price_clean"].max())


# 4.
def extract_bathrooms(text):

    if pd.isna(text):
        return None
    
    text = str(text).lower()

    # handle hald bath
    if "half" in text:
        return 0.5
    
    #extract numeric values
    import re 
    match = re.search(r"\d+\.?\d*", text)

    if match:
        return float(match.group())
    
    return None

df["bathrooms"] = df["bathrooms_text"].apply(extract_bathrooms)

print(df[["bathrooms_text", "bathrooms"]].head(15))


# 5.
df["host_is_superhost"] = df["host_is_superhost"].map({"t": True, "f": False})
df["instant_bookable"] = df["instant_bookable"].map({"t": True, "f": False})

# Convert host_since to datetime
df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")

#Create host tenure in years relative to 2025
df["host_tenure_years"] = 2025 - df["host_since"].dt.year

#Check reusult
print(df[["host_is_superhost", "instant_bookable", "host_since", "host_tenure_years"]].head())

# 6. 
import numpy as np
import matplotlib.pyplot as plt

#create log_price variable
df["log_price"] = np.log(df["price_clean"])

#create two plots side by side
fig, ax = plt.subplots(1, 2, figsize=(12,5))

# histogram real prices
ax[0].hist(df["price_clean"].dropna(), bins=50)
ax[0].set_title("Distribution of Airbnb Prices in Berlin")
ax[0].set_xlabel("Price (€)")
ax[0].set_ylabel("Frequency")


# histogram log prices
ax[1].hist(df["log_price"].dropna(), bins=50)
ax[1].set_title("Distribution of Log Prices")
ax[1].set_xlabel("Log Price")
ax[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# 7.
# Summary Table by neighbourhood
neighbourhood_summary = df.groupby("neighbourhood_cleansed").agg(
    number_of_listings=("price_clean", "count"),
    median_price=("price_clean", "median"),
    mean_review_score=("review_scores_rating", "mean")
)

neighbourhood_summary["median_price"] = neighbourhood_summary["median_price"].round(2)
neighbourhood_summary["mean_review_score"] = neighbourhood_summary["mean_review_score"].round(2)

# Sort by median price descending
neighbourhood_summary = neighbourhood_summary.sort_values(
    by="median_price",
    ascending=False
)

print("\nNeighbourhood summary:")
print(neighbourhood_summary)

print("\nTop 3 most expensive neighbourhoods:")
print(neighbourhood_summary.head(3))


# 8.
# Function to compute a trimmed mean

def trimmed_mean(x, pct):
    # Remove missing values
    x = np.array(x)
    x = x[~np.isnan(x)]

    # sort values 
    x = np.sort(x)

    # Number of observations to trim each side
    n = len(x)
    trim_n = int(n * pct / 100)

    # Trim lower and upper pct%
    x_trimmed = x[trim_n : n - trim_n]

    # Compute mean of trimmed array
    return np.mean(x_trimmed)

# Apply trimmed mean to cleaned prices
trimmed_price_mean = trimmed_mean(df["price_clean"], 5)
arithmetic_mean = df["price_clean"].mean()

print("\nArithmetic mean:", round(arithmetic_mean, 2))
print("5% trimmed mean:", round(trimmed_price_mean, 2))


# 9. (evtl log_price verwenden da besser)
#Plot 1: Price by Room type

plt.figure(figsize=(8, 5))

df.boxplot(column="price_clean", by="room_type", grid=False)

plt.title("Nightly Price by Room Type")
plt.suptitle("") # removes automatic pandas subtitle
plt.xlabel("Room Type")
plt.ylabel("Price (€)")
plt.ylim(0, 500) # improves readability
plt.show()

#Plot 2: Price vs. accomodates

plt.figure(figsize=(8, 5))

plt.scatter(df["accommodates"], df["price_clean"], alpha=0.3)

plt.title("Nightly Price and Guest Capacity")
plt.xlabel("Accommodates")
plt.ylabel("Price (€)")
plt.ylim(0, 1000) # improves readability
plt.show()


# 10. 
# The exploratory analysis reveals several key patterns in the Berlin Airbnb market. 
# First, the distribution of prices is highly right-skewed, 
# indicating that most listings are moderately priced while a small number of listings are extremely expensive. 
# After applying a logarithmic transformation, the price distribution becomes more symmetric and suitable for statistical analysis.
# Room type appears to be an important determinant of price. 
# Entire homes or apartments are generally much more expensive than private or shared rooms. 
# In addition, listings that accommodate more guests tend to have higher prices, suggesting that larger properties command higher nightly rates. 
# Differences between neighbourhoods are also visible, with some areas showing considerably higher median prices. 
# Overall, property characteristics and location appear to be the most important factors associated with listing prices.
# review scores seem to have a weaker relationship with price.


# 11.
import json
import re

# Load GeoJSON file
with open("Assignment/Aufgaben in Python/data_assignment/berlin_bezirke.geojson", "r", encoding="utf-8") as f:
    berlin_geojson = json.load(f)

# Extract BZR_NAME from GeoJSON
bzr_names = []
for feature in berlin_geojson["features"]:
    name = feature["properties"]["BZR_NAME"]
    bzr_names.append(name)

bzr_df = pd.DataFrame({"BZR_NAME": bzr_names})

print("\nNumber of Bezirksregionen in GeoJSON:", len(bzr_df))

# Unique Airbnb neighbourhood names
airbnb_neighbourhoods = pd.DataFrame({
    "neighbourhood_cleansed": df["neighbourhood_cleansed"].dropna().unique()
})

print("Number of Airbnb neighbourhoods:", len(airbnb_neighbourhoods))

# Cleaning function
def clean_region_name(name):
    if pd.isna(name):
        return None

    name = str(name).lower().strip()

    # Normalize spacing
    name = " ".join(name.split())

    # Standardize common abbreviations
    name = name.replace("str.", "straße")
    name = name.replace("strasse", "straße")

    # Replace slashes with spaces
    name = name.replace("/", " ")

    # Replace hyphens with spaces
    name = name.replace("-", " ")

    # Remove duplicate spaces again
    name = " ".join(name.split())

    return name


# Apply cleaning
bzr_df["clean_name"] = bzr_df["BZR_NAME"].apply(clean_region_name)
airbnb_neighbourhoods["clean_name"] = airbnb_neighbourhoods["neighbourhood_cleansed"].apply(clean_region_name)

# Match analysis
geo_names = set(bzr_df["clean_name"])
airbnb_names = set(airbnb_neighbourhoods["clean_name"])

matched_regions = geo_names.intersection(airbnb_names)
unmatched_airbnb = airbnb_names - geo_names
unmatched_geojson = geo_names - airbnb_names

print("\nMatched regions:", len(matched_regions))
print("Unmatched Airbnb names:", len(unmatched_airbnb))
print("Unmatched GeoJSON names:", len(unmatched_geojson))

print("\nUnmatched Airbnb neighbourhood names:")
print(sorted(unmatched_airbnb))

print("\nUnmatched GeoJSON region names:")
print(sorted(unmatched_geojson))


# 12.
import plotly.express as px

# Make sure cleaned neighbourhood names exist in df
df["neighbourhood_clean"] = df["neighbourhood_cleansed"].apply(clean_region_name)

# Aggregate Airbnb data by cleaned neighbourhood name
airbnb_region_summary = df.groupby("neighbourhood_clean").agg(
    number_of_listings=("price_clean", "count"),
    median_price=("price_clean", "median")
).reset_index()

# Merge Airbnb summary with GeoJSON region names
map_df = bzr_df.merge(
    airbnb_region_summary,
    left_on="clean_name",
    right_on="neighbourhood_clean",
    how="left"
)

# Optional rounding
map_df["median_price"] = map_df["median_price"].round(2)

# Only preview a few rows
print(map_df[["BZR_NAME", "number_of_listings", "median_price"]].head())

# Create choropleth
fig = px.choropleth(
    map_df,
    geojson=berlin_geojson,
    locations="BZR_NAME",
    featureidkey="properties.BZR_NAME",
    color="median_price",
    hover_name="BZR_NAME",
    hover_data={
        "number_of_listings": True,
        "median_price": True,
        "BZR_NAME": False
    },
    color_continuous_scale="Viridis",
    title="Median Nightly Airbnb Prices by Berlin Bezirksregion"
)

fig.update_geos(fitbounds="locations", visible=False)

# Map in HTML Format da sonst nicht funktioniert
fig.write_html("berlin_choropleth.html")
print("Map saved as berlin_choropleth.html")



# 13.
import plotly.express as px

# Make sure cleaned neighbourhood names exist
df["neighbourhood_clean"] = df["neighbourhood_cleansed"].apply(clean_region_name)

# Remove old median columns if code was run before
for col in ["region_median_price", "region_median_price_x", "region_median_price_y"]:
    if col in df.columns:
        df = df.drop(columns=col)

# Median price by cleaned region
region_median = df.groupby("neighbourhood_clean")["price_clean"].median().reset_index()
region_median = region_median.rename(columns={"price_clean": "region_median_price"})

# Merge regional median back to each listing
df = df.merge(region_median, on="neighbourhood_clean", how="left")

# Create above/below median indicator
df["above_region_median"] = np.where(
    df["price_clean"] > df["region_median_price"],
    "Above regional median",
    "Below or equal to regional median"
)

# --- Base choropleth map ---
fig = px.choropleth(
    map_df,
    geojson=berlin_geojson,
    locations="BZR_NAME",
    featureidkey="properties.BZR_NAME",
    color="median_price",
    hover_name="BZR_NAME",
    hover_data={
        "number_of_listings": True,
        "median_price": True,
        "BZR_NAME": False
    },
    color_continuous_scale="Viridis",
    title="Berlin Airbnb Prices: Regional Median and Listing-Level Deviations"
)

fig.update_geos(
    fitbounds="locations",
    visible=False
)

# --- Scatter layer for listings ---
scatter = px.scatter_geo(
    df.dropna(subset=["latitude", "longitude", "price_clean", "region_median_price"]),
    lat="latitude",
    lon="longitude",
    color="above_region_median",
    color_discrete_map={
        "Above regional median": "red",
        "Below or equal to regional median": "blue"
    },
    hover_name="neighbourhood_cleansed",
    hover_data={
        "price_clean": True,
        "region_median_price": True,
        "above_region_median": True
    }
)

# Add scatter traces to choropleth figure
for trace in scatter.data:
    trace.marker.size = 4
    trace.marker.opacity = 0.5
    fig.add_trace(trace)

# Improve layout / legend
fig.update_layout(
    legend_title_text="Listing price vs. regional median"
)

# Save interactive map
fig.write_html("berlin_choropleth_with_listings.html")
print("Combined map saved as berlin_choropleth_with_listings.html")


# 14.
import plotly.express as px

# Extract area from GeoJSON
bzr_area = []
for feature in berlin_geojson["features"]:
    bzr_area.append({
        "BZR_NAME": feature["properties"]["BZR_NAME"],
        "GROESSE_m2": feature["properties"]["GROESSE_m2"]
    })

area_df = pd.DataFrame(bzr_area)

# Clean names for merging
area_df["clean_name"] = area_df["BZR_NAME"].apply(clean_region_name)

# Listing counts by cleaned region
listing_counts = df.groupby("neighbourhood_clean").agg(
    number_of_listings=("price_clean", "count")
).reset_index()

# Merge area + listing counts
density_df = area_df.merge(
    listing_counts,
    left_on="clean_name",
    right_on="neighbourhood_clean",
    how="left"
)

# Replace missing listing counts with 0
density_df["number_of_listings"] = density_df["number_of_listings"].fillna(0)

# Convert area to km²
density_df["area_km2"] = density_df["GROESSE_m2"] / 1_000_000

# Compute listing density
density_df["listing_density"] = density_df["number_of_listings"] / density_df["area_km2"]

# Optional rounding
density_df["listing_density"] = density_df["listing_density"].round(2)

# Check highest densities
density_top = density_df.sort_values("listing_density", ascending=False)

print("\nTop Bezirksregionen by listing density:")
print(density_top[["BZR_NAME", "number_of_listings", "area_km2", "listing_density"]].head(10))

# Choropleth map for listing density
fig_density = px.choropleth(
    density_df,
    geojson=berlin_geojson,
    locations="BZR_NAME",
    featureidkey="properties.BZR_NAME",
    color="listing_density",
    hover_name="BZR_NAME",
    hover_data={
        "number_of_listings": True,
        "area_km2": True,
        "listing_density": True,
        "BZR_NAME": False
    },
    color_continuous_scale="Plasma",
    title="Airbnb Listing Density by Berlin Bezirksregion (Listings per km²)"
)

fig_density.update_geos(
    fitbounds="locations",
    visible=False
)

# Save interactive map
fig_density.write_html("berlin_listing_density.html")
print("Density map saved as berlin_listing_density.html")

# 15.
#The spatial analysis reveals strong clustering of Airbnb listings in central districts of Berlin. 
# The highest listing densities are observed in areas such as Brunnenstraße Süd, Helmholtzplatz, and Prenzlauer Berg Südwest. 
# These neighborhoods are located in central and highly urbanized parts of the city, which are popular with tourists due to their nightlife, 
# cultural attractions, and accessibility. As a result, many hosts offer short-term rentals in these locations. 
# In contrast, peripheral districts show much lower listing densities, mainly because they have larger residential areas and fewer tourist attractions. 
# Comparing the density map with the price map reveals that high-density areas do not always correspond to the highest prices. 
# While central areas concentrate many listings, price levels can vary depending on property characteristics and neighborhood desirability. 
# Overall, the results highlight the spatial concentration of Airbnb activity in Berlin’s inner districts.