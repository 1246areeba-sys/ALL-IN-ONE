"""
Reusable ML pipeline shared by the Flask web app.
Handles loading, cleaning, training and prediction for BOTH datasets.
"""
import re
import os
import io
import base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOUSE_CSV = os.path.join(BASE_DIR, 'house price  pridiction.csv')
SPOTIFY_CSV = os.path.join(BASE_DIR, 'high_popularity_spotify_data.csv')

sns.set_style('whitegrid')


# ----------------------------------------------------------------------
# Cleaning helpers
# ----------------------------------------------------------------------
def parse_price(val):
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
    if pd.isna(val):
        return np.nan
    m = re.search(r'\d+(?:\.\d+)?', str(val).replace(',', ''))
    return float(m.group()) if m else np.nan


# ----------------------------------------------------------------------
# House price pipeline
# ----------------------------------------------------------------------
def load_and_clean_house():
    df = pd.read_csv(HOUSE_CSV)
    df['price_crore'] = df['price'].apply(parse_price)
    df['rate_per_sqft'] = df['rate'].apply(first_number)
    df['area_sqm'] = df['area'].apply(first_number)
    df['bedRoom_n'] = df['bedRoom'].apply(first_number)
    df['bathroom_n'] = df['bathroom'].apply(first_number)
    df['balcony_n'] = df['balcony'].apply(first_number)
    df['noOfFloor_n'] = df['noOfFloor'].apply(first_number)
    drop = ['property_name', 'link', 'society', 'price', 'rate', 'area', 'areaWithType',
            'bedRoom', 'bathroom', 'balcony', 'additionalRoom', 'address', 'noOfFloor',
            'facing', 'agePossession', 'nearbyLocations', 'description', 'furnishDetails',
            'features', 'rating', 'property_id']
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.dropna(subset=['price_crore'])
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def train_house():
    df = load_and_clean_house()
    y = df['price_crore']
    X = df.drop(columns=['price_crore'])
    encoders = {}
    for col in X.select_dtypes(include=['object', 'str']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'r2': round(r2_score(y_test, pred), 4),
        'mae': round(mean_absolute_error(y_test, pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, pred)), 4),
    }
    # feature importance
    importances = model.feature_importances_
    feat_imp = (pd.DataFrame({'feature': X.columns, 'importance': importances})
                .sort_values('importance', ascending=False).to_dict('records'))
    return model, scaler, encoders, list(X.columns), metrics, feat_imp


def predict_house(model, scaler, encoders, features, form):
    row = {f: 0.0 for f in features}
    for f in ['rate_per_sqft', 'area_sqm', 'bedRoom_n', 'bathroom_n', 'balcony_n', 'noOfFloor_n']:
        if f in form and form[f] not in (None, ''):
            row[f] = float(form[f])
    sample = pd.DataFrame([row])[features]
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    return round(float(pred), 3)


# ----------------------------------------------------------------------
# Spotify pipeline
# ----------------------------------------------------------------------
def load_and_clean_spotify():
    df = pd.read_csv(SPOTIFY_CSV)
    drop = ['track_href', 'uri', 'analysis_url', 'track_id', 'track_album_id', 'id',
            'playlist_id', 'track_name', 'track_artist', 'track_album_name', 'playlist_name',
            'track_album_release_date', 'type']
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.dropna(subset=['track_popularity'])
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    cat_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def train_spotify():
    df = load_and_clean_spotify()
    y = df['track_popularity']
    X = df.drop(columns=['track_popularity'])
    encoders = {}
    for col in X.select_dtypes(include=['object', 'str']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'r2': round(r2_score(y_test, pred), 4),
        'mae': round(mean_absolute_error(y_test, pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, pred)), 4),
    }
    importances = model.feature_importances_
    feat_imp = (pd.DataFrame({'feature': X.columns, 'importance': importances})
                .sort_values('importance', ascending=False).to_dict('records'))
    return model, scaler, encoders, list(X.columns), metrics, feat_imp


def predict_spotify(model, scaler, encoders, features, form):
    row = {f: 0.0 for f in features}
    numeric_defaults = {
        'energy': 0.5, 'tempo': 120.0, 'danceability': 0.5, 'loudness': -8.0,
        'liveness': 0.2, 'valence': 0.5, 'speechiness': 0.1, 'time_signature': 4.0,
        'instrumentalness': 0.0, 'duration_ms': 200000.0, 'acousticness': 0.3,
        'mode': 1.0, 'key': 5.0,
    }
    for f, default in numeric_defaults.items():
        if f in features:
            row[f] = default
    # Categorical columns must NOT be coerced to float here; they are encoded below.
    cat_cols = set(encoders.keys())
    for f in form:
        if f in features and f not in cat_cols and form[f] not in (None, ''):
            try:
                row[f] = float(form[f])
            except (ValueError, TypeError):
                pass
    # encode categoricals
    for col, le in encoders.items():
        if col in features:
            val = str(form.get(col, le.classes_[0]))
            if val in list(le.classes_):
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0
    sample = pd.DataFrame([row])[features]
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    return round(float(pred), 2)


# ----------------------------------------------------------------------
# Graph helper -> base64 PNG
# ----------------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def house_graphs(df):
    figs = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['price_crore'], bins=30, kde=True, ax=ax, color='#d4af37')
    ax.set_title('House Price Distribution (Crore)')
    figs['price'] = fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='area_sqm', y='price_crore', ax=ax, color='#b8860b')
    ax.set_title('Area vs Price')
    figs['area'] = fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='bedRoom_n', y='price_crore', ax=ax, color='#d4af37')
    ax.set_title('Bedrooms vs Price')
    figs['bed'] = fig_to_base64(fig)
    return figs


def spotify_graphs(df):
    figs = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['track_popularity'], bins=30, kde=True, ax=ax, color='#d4af37')
    ax.set_title('Track Popularity Distribution')
    figs['pop'] = fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='danceability', y='track_popularity', ax=ax, color='#b8860b')
    ax.set_title('Danceability vs Popularity')
    figs['dance'] = fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='playlist_genre', y='track_popularity', ax=ax, color='#d4af37')
    ax.set_title('Genre vs Popularity')
    figs['genre'] = fig_to_base64(fig)
    return figs
