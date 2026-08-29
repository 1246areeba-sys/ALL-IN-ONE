"""
Prediction Analysis - Plain Python Script
==========================================
Runs the full pipeline for TWO datasets, each step in its own clearly
separated block (like a notebook cell):

    Task A: House Price Prediction   (house price  pridiction.csv)
    Task B: Spotify Popularity        (high_popularity_spotify_data.csv)

Pipeline per dataset:
    1. Imports
    2. Data Loading
    3. Data Understanding (head / tail / info / describe)
    4. Data Cleaning
    5. Data Fill & Drop
    6. Graphs
    7. Convert Data (encode + scale)
    8. Train / Test Split
    9. Train Model & Save to .pkl

Run with the virtual environment:
    venv\\Scripts\\activate
    python prediction.py
"""

# Force UTF-8 output so the Windows console can print symbols like the Rupee sign
import sys
import io
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Use a non-interactive backend so the script runs headless (graphs saved as PNG)
import matplotlib
matplotlib.use('Agg')

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)


# =====================================================================
# 2. DATA LOADING
# =====================================================================
house_df = pd.read_csv('house price  pridiction.csv')
spotify_df = pd.read_csv('high_popularity_spotify_data.csv')

print('House dataset shape :', house_df.shape)
print('Spotify dataset shape:', spotify_df.shape)


# =====================================================================
# 3. DATA UNDERSTANDING - HOUSE PRICE
# =====================================================================
print('\n===== HOUSE PRICE: head() =====')
print(house_df.head())

print('\n===== HOUSE PRICE: tail() =====')
print(house_df.tail())

print('\n===== HOUSE PRICE: info() =====')
house_df.info()

print('\n===== HOUSE PRICE: describe() =====')
print(house_df.describe(include='all'))

print('\n===== HOUSE PRICE: columns / shape / nulls =====')
print('Columns:', list(house_df.columns))
print('Shape  :', house_df.shape)
print('Nulls per column:')
print(house_df.isnull().sum())


# =====================================================================
# 3. DATA UNDERSTANDING - SPOTIFY
# =====================================================================
print('\n===== SPOTIFY: head() =====')
print(spotify_df.head())

print('\n===== SPOTIFY: tail() =====')
print(spotify_df.tail())

print('\n===== SPOTIFY: info() =====')
spotify_df.info()

print('\n===== SPOTIFY: describe() =====')
print(spotify_df.describe(include='all'))

print('\n===== SPOTIFY: columns / shape / nulls =====')
print('Columns:', list(spotify_df.columns))
print('Shape  :', spotify_df.shape)
print('Nulls per column:')
print(spotify_df.isnull().sum())


# =====================================================================
# 4. DATA CLEANING - HOUSE PRICE
# =====================================================================
def parse_price(val):
    """Convert Indian price strings like '5.25 Crore' / '3.6 Crore' to float (Crore)."""
    if pd.isna(val):
        return np.nan
    val = str(val).lower().replace(',', '')
    m = re.search(r'\d+(?:\.\d+)?', val)
    if not m:
        return np.nan
    num = float(m.group())
    if 'crore' in val:
        return num
    if 'lakh' in val or 'lac' in val:
        return num / 100.0
    return num


def first_number(val):
    """Extract the FIRST number from a text value (handles '20,115/sq.ft.')."""
    if pd.isna(val):
        return np.nan
    m = re.search(r'\d+(?:\.\d+)?', str(val).replace(',', ''))
    return float(m.group()) if m else np.nan


house = house_df.copy()
house['price_crore'] = house['price'].apply(parse_price)
house['rate_per_sqft'] = house['rate'].apply(first_number)
house['area_sqm'] = house['area'].apply(first_number)
house['bedRoom_n'] = house['bedRoom'].apply(first_number)
house['bathroom_n'] = house['bathroom'].apply(first_number)
house['balcony_n'] = house['balcony'].apply(first_number)
house['noOfFloor_n'] = house['noOfFloor'].apply(first_number)

print('\n===== HOUSE PRICE: cleaned numeric columns =====')
print(house[['price_crore', 'rate_per_sqft', 'area_sqm',
             'bedRoom_n', 'bathroom_n', 'balcony_n', 'noOfFloor_n']].head())

# Drop non-predictive text / identifier columns
text_cols_to_drop = ['property_name', 'link', 'society', 'price', 'rate', 'area', 'areaWithType',
                     'bedRoom', 'bathroom', 'balcony', 'additionalRoom', 'address', 'noOfFloor',
                     'facing', 'agePossession', 'nearbyLocations', 'description', 'furnishDetails',
                     'features', 'rating', 'property_id']
house = house.drop(columns=text_cols_to_drop)
print('Remaining house columns:', list(house.columns))


# =====================================================================
# 4. DATA CLEANING - SPOTIFY
# =====================================================================
spotify = spotify_df.copy()
drop_cols = ['track_href', 'uri', 'analysis_url', 'track_id', 'track_album_id', 'id',
             'playlist_id', 'track_name', 'track_artist', 'track_album_name', 'playlist_name',
             'track_album_release_date', 'type']
spotify = spotify.drop(columns=[c for c in drop_cols if c in spotify.columns])
print('\n===== SPOTIFY: remaining columns =====')
print(list(spotify.columns))


# =====================================================================
# 5. DATA FILL & DROP - HOUSE PRICE
# =====================================================================
house = house.dropna(subset=['price_crore'])
print('\n===== HOUSE PRICE: after dropping missing targets =====', house.shape)

num_cols = house.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if house[col].isnull().any():
        house[col] = house[col].fillna(house[col].median())

print('House nulls remaining:')
print(house.isnull().sum())


# =====================================================================
# 5. DATA FILL & DROP - SPOTIFY
# =====================================================================
spotify = spotify.dropna(subset=['track_popularity'])
print('\n===== SPOTIFY: after dropping missing targets =====', spotify.shape)

num_cols = spotify.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if spotify[col].isnull().any():
        spotify[col] = spotify[col].fillna(spotify[col].median())

cat_cols = spotify.select_dtypes(include=['object', 'str']).columns
for col in cat_cols:
    if spotify[col].isnull().any():
        spotify[col] = spotify[col].fillna(spotify[col].mode()[0])

print('Spotify nulls remaining:')
print(spotify.isnull().sum())


# =====================================================================
# 6. GRAPHS - HOUSE PRICE
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(house['price_crore'], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('House Price Distribution (Crore)')
sns.scatterplot(data=house, x='area_sqm', y='price_crore', ax=axes[0, 1])
axes[0, 1].set_title('Area vs Price')
sns.boxplot(data=house, x='bedRoom_n', y='price_crore', ax=axes[1, 0])
axes[1, 0].set_title('Bedrooms vs Price')
sns.heatmap(house.corr(numeric_only=True), annot=True, fmt='.2f', ax=axes[1, 1], cmap='coolwarm')
axes[1, 1].set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig('house_graphs.png')
plt.close()

plt.figure(figsize=(8, 5))
corr = house.corr(numeric_only=True)['price_crore'].sort_values(ascending=False)
sns.barplot(x=corr.values, y=corr.index, palette='viridis')
plt.title('Feature correlation with House Price')
plt.tight_layout()
plt.savefig('house_correlation.png')
plt.close()


# =====================================================================
# 6. GRAPHS - SPOTIFY
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(spotify['track_popularity'], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Track Popularity Distribution')
sns.scatterplot(data=spotify, x='danceability', y='track_popularity', ax=axes[0, 1])
axes[0, 1].set_title('Danceability vs Popularity')
sns.boxplot(data=spotify, x='playlist_genre', y='track_popularity', ax=axes[1, 0])
axes[1, 0].set_title('Genre vs Popularity')
sns.heatmap(spotify.corr(numeric_only=True), annot=False, ax=axes[1, 1], cmap='coolwarm')
axes[1, 1].set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig('spotify_graphs.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=spotify, x='playlist_genre', y='track_popularity', estimator=np.mean)
plt.title('Average Popularity by Genre')
plt.tight_layout()
plt.savefig('spotify_genre.png')
plt.close()


# =====================================================================
# 7. CONVERT DATA - HOUSE PRICE
# =====================================================================
y_house = house['price_crore']
X_house = house.drop(columns=['price_crore'])

house_encoders = {}
for col in X_house.select_dtypes(include=['object', 'str']).columns:
    le = LabelEncoder()
    X_house[col] = le.fit_transform(X_house[col].astype(str))
    house_encoders[col] = le

house_scaler = StandardScaler()
X_house_scaled = pd.DataFrame(house_scaler.fit_transform(X_house), columns=X_house.columns)
print('\n===== HOUSE PRICE: features shape =====', X_house_scaled.shape)


# =====================================================================
# 7. CONVERT DATA - SPOTIFY
# =====================================================================
y_spot = spotify['track_popularity']
X_spot = spotify.drop(columns=['track_popularity'])

spot_encoders = {}
for col in X_spot.select_dtypes(include=['object', 'str']).columns:
    le = LabelEncoder()
    X_spot[col] = le.fit_transform(X_spot[col].astype(str))
    spot_encoders[col] = le

spot_scaler = StandardScaler()
X_spot_scaled = pd.DataFrame(spot_scaler.fit_transform(X_spot), columns=X_spot.columns)
print('===== SPOTIFY: features shape =====', X_spot_scaled.shape)


# =====================================================================
# 8. TRAIN / TEST SPLIT - HOUSE PRICE
# =====================================================================
Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_house_scaled, y_house, test_size=0.2, random_state=42)
print('\n===== HOUSE PRICE: split =====',
      'Train', Xh_train.shape, '| Test', Xh_test.shape)


# =====================================================================
# 8. TRAIN / TEST SPLIT - SPOTIFY
# =====================================================================
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_spot_scaled, y_spot, test_size=0.2, random_state=42)
print('===== SPOTIFY: split =====',
      'Train', Xs_train.shape, '| Test', Xs_test.shape)


# =====================================================================
# 9. TRAIN MODEL & SAVE - HOUSE PRICE
# =====================================================================
house_model = RandomForestRegressor(n_estimators=100, random_state=42)
house_model.fit(Xh_train, yh_train)

h_pred = house_model.predict(Xh_test)
print('\n===== HOUSE PRICE RESULTS =====')
print('R2  :', round(r2_score(yh_test, h_pred), 4))
print('MAE :', round(mean_absolute_error(yh_test, h_pred), 4))
print('RMSE:', round(np.sqrt(mean_squared_error(yh_test, h_pred)), 4))

with open('house_price_model.pkl', 'wb') as f:
    pickle.dump({'model': house_model, 'scaler': house_scaler,
                 'encoders': house_encoders, 'features': list(X_house.columns)}, f)
print('House model saved to house_price_model.pkl')


# =====================================================================
# 9. TRAIN MODEL & SAVE - SPOTIFY
# =====================================================================
spotify_model = RandomForestRegressor(n_estimators=100, random_state=42)
spotify_model.fit(Xs_train, ys_train)

s_pred = spotify_model.predict(Xs_test)
print('\n===== SPOTIFY RESULTS =====')
print('R2  :', round(r2_score(ys_test, s_pred), 4))
print('MAE :', round(mean_absolute_error(ys_test, s_pred), 4))
print('RMSE:', round(np.sqrt(mean_squared_error(ys_test, s_pred)), 4))

with open('spotify_popularity_model.pkl', 'wb') as f:
    pickle.dump({'model': spotify_model, 'scaler': spot_scaler,
                 'encoders': spot_encoders, 'features': list(X_spot.columns)}, f)
print('Spotify model saved to spotify_popularity_model.pkl')

print('\nAll steps completed successfully.')
