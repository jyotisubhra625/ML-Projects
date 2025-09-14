# Bengaluru House Price Predictor

This is a web application that predicts house prices in Bengaluru, India. The prediction is based on location, total square feet, number of bathrooms, and number of bedrooms (BHK).

The project uses a Ridge Regression model trained on a cleaned dataset derived from Kaggle's "Bangalore House Price" dataset.

## Project Structure

```
├── artifacts/
│   ├── Cleaned_data.csv
│   ├── columns.json
│   └── RidgeModel.pkl
├── data/
│   └── Bengaluru_House_Data.csv
├── templates/
│   └── index.html
├── venv/
├── app.py
├── create_model.py
├── requirements.txt
└── util.py
```

## How to Run

1.  **Set up the environment:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the model:**
    This script will clean the raw data, train a Ridge regression model, and save the model and column information into the `artifacts/` directory.
    ```bash
    python create_model.py
    ```

4.  **Run the Flask server:**
    ```bash
    python app.py
    ```

5.  **Open the application:**
    Navigate to `http://127.0.0.1:5001` in your web browser."# ML-Projects" 
"# ML-Projects" 
