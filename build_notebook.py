"""Builder script that generates prediction_notebook.ipynb.
Each logical step is placed in its own notebook cell so the user can run
them one at a time. Covers both the House Price and Spotify Popularity datasets.
"""
import re
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def add(markdown, code):
    if markdown:
        cells.append(nbf.v4.new_markdown_cell(markdown))
    cells.append(nbf.v4.new_code_cell(code))


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
add(
    "# Prediction Analysis Notebook\n"
    "This notebook covers **two** prediction tasks, each following the same pipeline:\n"
    "1. Imports\n"
    "2. Data Loading\n"
    "3. Data Understanding (head / tail / info / describe)\n"
    "4. Data Cleaning\n"
    "5. Data Fill & Drop (handle missing values)\n"
    "6. Graphs / Visualisation\n"
    "7. Convert Data (encode categoricals, scale)\n"
    "8. Train / Test Split\n"
    "9. Train Model & Save to .pkl\n\n"
    "**Task A:** House Price Prediction  (`house price  pridiction.csv`)\n"
    "**Task B:** Spotify Song Popularity Prediction (`high_popularity_spotify_data.csv`)\n\n"
    "> Run the cells in order. Every command lives in its own cell.",
    "",
)

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
add(
    "## 1. Imports\n"
    "Import every library we need. Run this first so the rest of the notebook works.",
    "import re\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n"
    "from sklearn.linear_model import LinearRegression\n"
    "from sklearn.ensemble import RandomForestRegressor\n"
    "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n"
    "import pickle\n"
    "\n"
    "sns.set_style('whitegrid')\n"
    "pd.set_option('display.max_columns', None)\n"
    "print('All libraries imported successfully.')",
)

# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------
add(
    "## 2. Data Loading\n"
    "Load both CSV files into pandas DataFrames.",
    "house_df = pd.read_csv('house price  pridiction.csv')\n"
    "spotify_df = pd.read_csv('high_popularity_spotify_data.csv')\n"
    "\n"
    "print('House dataset shape :', house_df.shape)\n"
    "print('Spotify dataset shape:', spotify_df.shape)",
)

# ---------------------------------------------------------------------------
# 3. DATA UNDERSTANDING
# ---------------------------------------------------------------------------
add(
    "## 3. Data Understanding - House Price\n"
    "Use `head()`, `tail()`, `info()`, `describe()` and `columns` to understand the data.",
    "# Show the first 5 rows\n"
    "house_df.head()",
)
add(
    "Data Understanding - House Price (tail)\n"
    "Show the last 5 rows with `tail()`.",
    "# Show the last 5 rows\n"
    "house_df.tail()",
)
add(
    "Data Understanding - House Price (info)\n"
    "Show column types and non-null counts with `info()`.",
    "# General information about the DataFrame\n"
    "house_df.info()",
)
add(
    "Data Understanding - House Price (describe)\n"
    "Statistical summary of numeric columns with `describe()`.",
    "# Statistical description\n"
    "house_df.describe(include='all')",
)
add(
    "Data Understanding - House Price (columns & shape)\n"
    "List all columns and the dataset shape.",
    "print('Columns:', list(house_df.columns))\n"
    "print('Shape  :', house_df.shape)\n"
    "print('Nulls per column:')\n"
    "print(house_df.isnull().sum())",
)

add(
    "## 3. Data Understanding - Spotify Popularity\n"
    "Repeat the understanding steps for the Spotify dataset.",
    "# First 5 rows\n"
    "spotify_df.head()",
)
add(
    "Data Understanding - Spotify (tail)",
    "# Last 5 rows\n"
    "spotify_df.tail()",
)
add(
    "Data Understanding - Spotify (info)",
    "spotify_df.info()",
)
add(
    "Data Understanding - Spotify (describe)",
    "spotify_df.describe(include='all')",
)
add(
    "Data Understanding - Spotify (columns & nulls)",
    "print('Columns:', list(spotify_df.columns))\n"
    "print('Shape  :', spotify_df.shape)\n"
    "print('Nulls per column:')\n"
    "print(spotify_df.isnull().sum())",
)

# ---------------------------------------------------------------------------
# 4. DATA CLEANING
# ---------------------------------------------------------------------------
add(
    "## 4. Data Cleaning - House Price\n"
    "Extract numeric values from messy text columns (`price`, `rate`, `area`, `bedRoom`, etc.).",
    "# --- Helper to convert Indian price strings like '5.25 Crore' / '3.6 Crore' to float (in Crore) ---\n"
    "def parse_price(val):\n"
    "    if pd.isna(val):\n"
    "        return np.nan\n"
    "    val = str(val).lower().replace(',', '')\n"
    "    m = re.search(r'\\d+(?:\\.\\d+)?', val)\n"
    "    if not m:\n"
    "        return np.nan\n"
    "    num = float(m.group())\n"
    "    if 'crore' in val:\n"
    "        return num\n"
    "    if 'lakh' in val or 'lac' in val:\n"
    "        return num / 100.0\n"
    "    return num\n"
    "\n"
    "# --- Helper to extract the FIRST number from a text column (handles '20,115/sq.ft.') ---\n"
    "def first_number(val):\n"
    "    if pd.isna(val):\n"
    "        return np.nan\n"
    "    m = re.search(r'\\d+(?:\\.\\d+)?', str(val).replace(',', ''))\n"
    "    return float(m.group()) if m else np.nan\n"
    "\n"
    "house = house_df.copy()\n"
    "house['price_crore'] = house['price'].apply(parse_price)\n"
    "house['rate_per_sqft'] = house['rate'].apply(lambda x: first_number(x))\n"
    "house['area_sqm'] = house['area'].apply(lambda x: first_number(x))\n"
    "house['bedRoom_n'] = house['bedRoom'].apply(lambda x: first_number(x))\n"
    "house['bathroom_n'] = house['bathroom'].apply(lambda x: first_number(x))\n"
    "house['balcony_n'] = house['balcony'].apply(lambda x: first_number(x))\n"
    "house['noOfFloor_n'] = house['noOfFloor'].apply(lambda x: first_number(x))\n"
    "print('Cleaned numeric columns created:')\n"
    "print(house[['price_crore','rate_per_sqft','area_sqm','bedRoom_n','bathroom_n','balcony_n','noOfFloor_n']].head())",
)
add(
    "Data Cleaning - House Price (drop useless text columns)\n"
    "Remove columns that are free text / identifiers and not useful for modelling.",
    "text_cols_to_drop = ['property_name','link','society','price','rate','area','areaWithType',\n"
    "                      'bedRoom','bathroom','balcony','additionalRoom','address','noOfFloor',\n"
    "                      'facing','agePossession','nearbyLocations','description','furnishDetails',\n"
    "                      'features','rating','property_id']\n"
    "house = house.drop(columns=text_cols_to_drop)\n"
    "print('Remaining house columns:', list(house.columns))",
)

add(
    "## 4. Data Cleaning - Spotify Popularity\n"
    "Drop URL/ID/text columns that are not predictive and keep audio features + target.",
    "spotify = spotify_df.copy()\n"
    "# Drop non-predictive text / identifier columns\n"
    "drop_cols = ['track_href','uri','analysis_url','track_id','track_album_id','id',\n"
    "             'playlist_id','track_name','track_artist','track_album_name','playlist_name',\n"
    "             'track_album_release_date','type']\n"
    "spotify = spotify.drop(columns=[c for c in drop_cols if c in spotify.columns])\n"
    "print('Remaining spotify columns:', list(spotify.columns))",
)

# ---------------------------------------------------------------------------
# 5. DATA FILL & DROP
# ---------------------------------------------------------------------------
add(
    "## 5. Data Fill & Drop - House Price\n"
    "Drop rows where the target (`price_crore`) is missing, then fill remaining numeric nulls with the median.",
    "# Drop rows with no target value\n"
    "house = house.dropna(subset=['price_crore'])\n"
    "print('After dropping missing targets:', house.shape)\n"
    "\n"
    "# Fill remaining numeric missing values with the column median\n"
    "num_cols = house.select_dtypes(include=[np.number]).columns\n"
    "for col in num_cols:\n"
    "    if house[col].isnull().any():\n"
    "        house[col] = house[col].fillna(house[col].median())\n"
    "\n"
    "print('Nulls remaining:')\n"
    "print(house.isnull().sum())",
)
add(
    "Data Fill & Drop - Spotify Popularity\n"
    "Drop rows missing the target, then fill numeric nulls with median and categorical nulls with mode.",
    "# Drop rows with no target\n"
    "spotify = spotify.dropna(subset=['track_popularity'])\n"
    "print('After dropping missing targets:', spotify.shape)\n"
    "\n"
    "# Fill numeric nulls with median\n"
    "num_cols = spotify.select_dtypes(include=[np.number]).columns\n"
    "for col in num_cols:\n"
    "    if spotify[col].isnull().any():\n"
    "        spotify[col] = spotify[col].fillna(spotify[col].median())\n"
    "\n"
    "# Fill categorical nulls with mode\n"
    "cat_cols = spotify.select_dtypes(include=['object','str']).columns\n"
    "for col in cat_cols:\n"
    "    if spotify[col].isnull().any():\n"
    "        spotify[col] = spotify[col].fillna(spotify[col].mode()[0])\n"
    "\n"
    "print('Nulls remaining:')\n"
    "print(spotify.isnull().sum())",
)

# ---------------------------------------------------------------------------
# 6. GRAPHS
# ---------------------------------------------------------------------------
add(
    "## 6. Graphs - House Price\n"
    "Visualise the target distribution and relationships with numeric features.",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
    "\n"
    "sns.histplot(house['price_crore'], bins=30, kde=True, ax=axes[0,0])\n"
    "axes[0,0].set_title('House Price Distribution (Crore)')\n"
    "\n"
    "sns.scatterplot(data=house, x='area_sqm', y='price_crore', ax=axes[0,1])\n"
    "axes[0,1].set_title('Area vs Price')\n"
    "\n"
    "sns.boxplot(data=house, x='bedRoom_n', y='price_crore', ax=axes[1,0])\n"
    "axes[1,0].set_title('Bedrooms vs Price')\n"
    "\n"
    "sns.heatmap(house.corr(numeric_only=True), annot=True, fmt='.2f', ax=axes[1,1], cmap='coolwarm')\n"
    "axes[1,1].set_title('Correlation Matrix')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()",
)
add(
    "Graphs - House Price (correlation with target)",
    "plt.figure(figsize=(8, 5))\n"
    "corr = house.corr(numeric_only=True)['price_crore'].sort_values(ascending=False)\n"
    "sns.barplot(x=corr.values, y=corr.index, palette='viridis')\n"
    "plt.title('Feature correlation with House Price')\n"
    "plt.tight_layout()\n"
    "plt.show()",
)
add(
    "## 6. Graphs - Spotify Popularity\n"
    "Visualise the popularity distribution and correlations of audio features.",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
    "\n"
    "sns.histplot(spotify['track_popularity'], bins=30, kde=True, ax=axes[0,0])\n"
    "axes[0,0].set_title('Track Popularity Distribution')\n"
    "\n"
    "sns.scatterplot(data=spotify, x='danceability', y='track_popularity', ax=axes[0,1])\n"
    "axes[0,1].set_title('Danceability vs Popularity')\n"
    "\n"
    "sns.boxplot(data=spotify, x='playlist_genre', y='track_popularity', ax=axes[1,0])\n"
    "axes[1,0].set_title('Genre vs Popularity')\n"
    "\n"
    "sns.heatmap(spotify.corr(numeric_only=True), annot=False, ax=axes[1,1], cmap='coolwarm')\n"
    "axes[1,1].set_title('Correlation Matrix')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()",
)
add(
    "Graphs - Spotify (popularity by genre)",
    "plt.figure(figsize=(8, 5))\n"
    "sns.barplot(data=spotify, x='playlist_genre', y='track_popularity', estimator=np.mean)\n"
    "plt.title('Average Popularity by Genre')\n"
    "plt.tight_layout()\n"
    "plt.show()",
)

# ---------------------------------------------------------------------------
# 7. CONVERT DATA
# ---------------------------------------------------------------------------
add(
    "## 7. Convert Data - House Price\n"
    "Separate features/target, encode any remaining categoricals, and scale numeric features.",
    "# Define target and features\n"
    "y_house = house['price_crore']\n"
    "X_house = house.drop(columns=['price_crore'])\n"
    "\n"
    "# Encode any remaining object columns\n"
    "house_encoders = {}\n"
    "for col in X_house.select_dtypes(include=['object','str']).columns:\n"
    "    le = LabelEncoder()\n"
    "    X_house[col] = le.fit_transform(X_house[col].astype(str))\n"
    "    house_encoders[col] = le\n"
    "\n"
    "# Scale numeric features\n"
    "house_scaler = StandardScaler()\n"
    "X_house_scaled = pd.DataFrame(\n"
    "    house_scaler.fit_transform(X_house),\n"
    "    columns=X_house.columns\n"
    ")\n"
    "print('House features shape:', X_house_scaled.shape)\n"
    "X_house_scaled.head()",
)
add(
    "## 7. Convert Data - Spotify Popularity\n"
    "Encode categorical columns (`playlist_genre`, `playlist_subgenre`, `key`, `mode`) and scale features.",
    "# Define target and features\n"
    "y_spot = spotify['track_popularity']\n"
    "X_spot = spotify.drop(columns=['track_popularity'])\n"
    "\n"
    "# Encode categorical columns\n"
    "spot_encoders = {}\n"
    "for col in X_spot.select_dtypes(include=['object','str']).columns:\n"
    "    le = LabelEncoder()\n"
    "    X_spot[col] = le.fit_transform(X_spot[col].astype(str))\n"
    "    spot_encoders[col] = le\n"
    "\n"
    "# Scale numeric features\n"
    "spot_scaler = StandardScaler()\n"
    "X_spot_scaled = pd.DataFrame(\n"
    "    spot_scaler.fit_transform(X_spot),\n"
    "    columns=X_spot.columns\n"
    ")\n"
    "print('Spotify features shape:', X_spot_scaled.shape)\n"
    "X_spot_scaled.head()",
)

# ---------------------------------------------------------------------------
# 8. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
add(
    "## 8. Train / Test Split - House Price\n"
    "Split the data into training (80%) and testing (20%) sets.",
    "Xh_train, Xh_test, yh_train, yh_test = train_test_split(\n"
    "    X_house_scaled, y_house, test_size=0.2, random_state=42\n"
    ")\n"
    "print('Train shape:', Xh_train.shape, '| Test shape:', Xh_test.shape)",
)
add(
    "## 8. Train / Test Split - Spotify Popularity\n"
    "Split the Spotify data into training and testing sets.",
    "Xs_train, Xs_test, ys_train, ys_test = train_test_split(\n"
    "    X_spot_scaled, y_spot, test_size=0.2, random_state=42\n"
    ")\n"
    "print('Train shape:', Xs_train.shape, '| Test shape:', Xs_test.shape)",
)

# ---------------------------------------------------------------------------
# 9. TRAIN MODEL & SAVE TO PKL
# ---------------------------------------------------------------------------
add(
    "## 9. Train Model & Save - House Price\n"
    "Train a Random Forest regressor, evaluate it, and save the model + scaler to a `.pkl` file.",
    "house_model = RandomForestRegressor(n_estimators=100, random_state=42)\n"
    "house_model.fit(Xh_train, yh_train)\n"
    "\n"
    "h_pred = house_model.predict(Xh_test)\n"
    "print('House R2  :', round(r2_score(yh_test, h_pred), 4))\n"
    "print('House MAE :', round(mean_absolute_error(yh_test, h_pred), 4))\n"
    "print('House RMSE:', round(np.sqrt(mean_squared_error(yh_test, h_pred)), 4))\n"
    "\n"
    "# Save model, scaler and feature columns\n"
    "with open('house_price_model.pkl', 'wb') as f:\n"
    "    pickle.dump({'model': house_model,\n"
    "                  'scaler': house_scaler,\n"
    "                  'encoders': house_encoders,\n"
    "                  'features': list(X_house.columns)}, f)\n"
    "print('House model saved to house_price_model.pkl')",
)
add(
    "## 9. Train Model & Save - Spotify Popularity\n"
    "Train a Random Forest regressor for song popularity, evaluate it, and save to `.pkl`.",
    "spotify_model = RandomForestRegressor(n_estimators=100, random_state=42)\n"
    "spotify_model.fit(Xs_train, ys_train)\n"
    "\n"
    "s_pred = spotify_model.predict(Xs_test)\n"
    "print('Spotify R2  :', round(r2_score(ys_test, s_pred), 4))\n"
    "print('Spotify MAE :', round(mean_absolute_error(ys_test, s_pred), 4))\n"
    "print('Spotify RMSE:', round(np.sqrt(mean_squared_error(ys_test, s_pred)), 4))\n"
    "\n"
    "# Save model, scaler and feature columns\n"
    "with open('spotify_popularity_model.pkl', 'wb') as f:\n"
    "    pickle.dump({'model': spotify_model,\n"
    "                  'scaler': spot_scaler,\n"
    "                  'encoders': spot_encoders,\n"
    "                  'features': list(X_spot.columns)}, f)\n"
    "print('Spotify model saved to spotify_popularity_model.pkl')",
)
add(
    "## 10. (Bonus) Load & Use Saved Model\n"
    "Example of loading a saved `.pkl` model and making a prediction on new data.",
    "# Load the saved house price model\n"
    "with open('house_price_model.pkl', 'rb') as f:\n"
    "    saved = pickle.load(f)\n"
    "\n"
    "# Build a small sample using the SAME feature order/columns\n"
    "sample = pd.DataFrame([{c: 0 for c in saved['features']}])\n"
    "sample_scaled = saved['scaler'].transform(sample)\n"
    "prediction = saved['model'].predict(sample_scaled)\n"
    "print('Sample prediction (Crore):', prediction)",
)

nb['cells'] = cells
with open('prediction_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Notebook created with', len(cells), 'cells.')
