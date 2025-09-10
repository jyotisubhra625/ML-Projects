import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    """
    Estimates the price of a home in Bengaluru given the location, sqft, bhk, and bath.
    """
    import pandas as pd
    # The model pipeline expects a pandas DataFrame with the same column names
    # it was trained on. The ColumnTransformer inside the pipeline will handle
    # the one-hot encoding of the 'location' column.
    # The order of columns should match what the model was trained on.
    # Based on create_model.py, the order is ['location', 'total_sqft', 'bath', 'bhk']
    # as X = data.drop('price', axis=1) and 'price' is the last column.
    # The pipeline's ColumnTransformer was defined with ['total_sqft', 'bath', 'bhk']
    # and then 'location', so passing a DataFrame handles the column matching.
    predict_df = pd.DataFrame([{'location': location, 'total_sqft': float(sqft), 'bath': int(bath), 'bhk': int(bhk)}])
    
    return round(__model.predict(predict_df)[0], 2)

def get_location_names():
    """
    Returns the location names.
    """
    return __locations

def load_saved_artifacts():
    """
    Loads the saved model and column information from disk.
    """
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    try:
        with open("./artifacts/columns.json", 'r') as f:
            columns_data = json.load(f)
            # 'data_columns' are the columns the model was trained on.
            __data_columns = columns_data.get('data_columns')
            # 'locations' are the unique location names for the dropdown.
            __locations = columns_data.get('locations')
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading columns.json: {e}. Please run create_model.py to generate artifacts.")
        __locations = []
        __data_columns = []

    try:
        with open("./artifacts/RidgeModel.pkl", 'rb') as f:
            __model = pickle.load(f)
    except FileNotFoundError:
        print("Error: 'RidgeModel.pkl' not found. Please run create_model.py to generate artifacts.")
        __model = None

    print("loading saved artifacts...done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2)) # other location