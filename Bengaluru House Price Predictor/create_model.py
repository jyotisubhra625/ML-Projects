import pandas as pd
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

print("Creating artifacts directory...")
print("Loading data...")
# Load the dataset
data = pd.read_csv("data/Bengaluru_House_Data.csv")

print("Cleaning and preparing data as per the notebook...")
# --- Start: Data Cleaning and Feature Engineering from Notebook ---
data.drop(columns=['area_type','availability','society','balcony'],inplace=True)
data['location'] = data['location'].fillna('Sarjapur  Road')
data['size'] = data['size'].fillna('2 BHK')
data['bath'] = data['bath'].fillna(data['bath'].median())
data['bhk'] = data['size'].str.split().str.get(0).astype(int)

def convertRange(x):
    # This function needs to handle string inputs from the raw CSV
    temp = str(x).split('-')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1])) / 2
    try:
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convertRange)
data.dropna(inplace=True) # Drop rows where total_sqft could not be converted

data['price_per_sqft'] = data['price'] * 100000 / data['total_sqft']

data['location'] = data['location'].apply(lambda x: x.strip())
location_count = data['location'].value_counts()
location_count_less_15 = location_count[location_count <= 15]
data['location'] = data['location'].apply(lambda x: 'other' if x in location_count_less_15 else x)

data = data[((data['total_sqft']/data['bhk']) >= 300)]

def remove_outlier_sqft(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_output = pd.concat([df_output, gen_df], ignore_index=True)
    return df_output

data = remove_outlier_sqft(data)

def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby("location"):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0],
            }
        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats["count"] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < (stats["mean"])].index.values,
                )
    return df.drop(exclude_indices, axis="index")

data = bhk_outlier_remover(data)

data.drop(columns=['size','price_per_sqft'],inplace=True)

import os
os.makedirs("artifacts", exist_ok=True)
data.to_csv("artifacts/Cleaned_data.csv", index=False)
print("Cleaned data saved to Cleaned_data.csv")
# --- End: Data Cleaning ---

print("Preparing data for training...")
# Define features (X) and target (y)
X = data.drop('price', axis=1)
y = data['price'] # Using direct price as per the notebook

# Create the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['total_sqft', 'bath', 'bhk']), # The order matters for prediction if not using pandas df
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), ['location'])
    ],
    remainder='passthrough'
)

# Create the full pipeline
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor', Ridge())]) # Using Ridge as it was the last one trained in the notebook

print("Training the model...")
pipe.fit(X, y)

print("Saving the new model to artifacts/RidgeModel.pkl...")
with open('artifacts/RidgeModel.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Saving column information to artifacts/columns.json...")
# The columns for the model are the original ones from the cleaned data
model_columns = [col.lower() for col in X.columns]
# The locations are the unique values in the 'location' column from the cleaned data
locations = sorted(X['location'].unique().tolist())
with open('artifacts/columns.json', 'w') as f:
    json.dump({'data_columns': model_columns, 'locations': locations}, f)

print("\nNew RidgeModel.pkl has been created successfully!")