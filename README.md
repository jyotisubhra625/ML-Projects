# 🤖 ML Projects

A collection of end-to-end Machine Learning projects — each featuring a trained model, preprocessing pipeline, and a deployable web app built with **Streamlit** or **Flask**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)

---

## 📂 Projects

### 🏠 [Bengaluru House Price Predictor](./Bengaluru%20House%20Price%20Predictor/)
> Predicts residential property prices in Bengaluru based on location, BHK, bathrooms, and area.

| Detail | Info |
|---|---|
| **Algorithm** | Ridge Regression |
| **Framework** | Flask |
| **Dataset** | Bengaluru House Price Data (Kaggle) |
| **Features** | Location, square footage, BHK, bathrooms |

- Data cleaned and outlier-removed through a preprocessing pipeline
- Model saved as `RidgeModel.pkl` with feature columns in `columns.json`
- Full web UI with a dynamic dropdown for 200+ Bengaluru localities

---

### 🩺 [Diabetes Predictor](./Diabetes%20Predictor/)
> Predicts whether a patient is likely to have diabetes based on health metrics.

| Detail | Info |
|---|---|
| **Algorithm** | Neural Network (Keras/TensorFlow) |
| **Framework** | Flask |
| **Dataset** | Pima Indians Diabetes Dataset |
| **Features** | Glucose, BMI, insulin, age, blood pressure, etc. |

- Deep learning model trained with Keras and saved as `diabetes_model.h5`
- Also includes a traditional ML pickle model `diabetes.pkl`
- Clean HTML/CSS frontend for easy input

---

### 🏏 [IPL Win Predictor](./IPL%20Win%20Predictor/)
> Predicts the probability of winning an IPL match in real-time given the current match situation.

| Detail | Info |
|---|---|
| **Algorithm** | Logistic Regression (Pipeline) |
| **Framework** | Streamlit |
| **Dataset** | IPL matches & deliveries CSVs |
| **Features** | Teams, city, target score, overs, wickets, runs left |

- Uses historical IPL data (matches + deliveries) to train a win probability model
- Outputs live win probability for both teams

---

### 💻 [Laptop Price Predictor](./Laptop%20Price%20Predictor/)
> Predicts the price of a laptop based on its specs — brand, RAM, processor, storage, display, and more.

| Detail | Info |
|---|---|
| **Algorithm** | Random Forest / Gradient Boosting (Pipeline) |
| **Framework** | Streamlit |
| **Dataset** | Laptop Price Dataset (Kaggle) |
| **Features** | Brand, type, RAM, weight, touchscreen, IPS, GPU, CPU, HDD/SSD |

- Full feature engineering pipeline for categorical + numerical inputs
- Interactive Streamlit UI with dropdown selectors for all specs

---

### 🎬 [Movie Recommendation Predictor](./Movie%20Recommendation%20Predictor/)
> Content-based movie recommender that suggests 5 similar movies with live posters from TMDB.

| Detail | Info |
|---|---|
| **Algorithm** | Cosine Similarity on CountVectorizer tags |
| **Framework** | Streamlit |
| **Dataset** | TMDB 5000 Movie Dataset (Kaggle) |
| **Features** | Title, genres, keywords, cast, crew, overview |

- Precomputed cosine similarity matrix for fast recommendations
- Live movie poster fetching via the TMDB API
- Model files compressed in `important_files.7z` — **must be extracted before running**

---

## ⚙️ Getting Started

### Clone the Repository

```bash
git clone https://github.com/jyotisubhra625/ML-Projects.git
cd ML-Projects
```

### Running a Specific Project

Navigate into any project folder and follow the steps below.

**1. Create and activate a virtual environment:**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

**2. Install that project's dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run the app:**

- **Streamlit projects** (IPL, Laptop, Movie Recommender):
  ```bash
  streamlit run app.py
  ```
- **Flask projects** (House Price, Diabetes):
  ```bash
  python app.py
  ```

> ⚠️ **Movie Recommender only**: You must first extract `important_files.7z` into the project folder so `movie_dict.pkl` and `similarity.pkl` are present before running.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python 3.9+ |
| **ML Libraries** | scikit-learn, Keras/TensorFlow, NumPy, Pandas |
| **Web Frameworks** | Streamlit, Flask |
| **APIs** | TMDB API |
| **Tools** | Jupyter Notebook, Pickle |

---

## 📁 Repository Structure

```
ML-Projects/
│
├── Bengaluru House Price Predictor/
│   ├── app.py
│   ├── create_model.py
│   ├── util.py
│   ├── requirements.txt
│   ├── artifacts/
│   │   ├── RidgeModel.pkl
│   │   ├── columns.json
│   │   └── Cleaned_data.csv
│   └── templates/index.html
│
├── Diabetes Predictor/
│   ├── app.py
│   ├── diabetes_model.h5
│   ├── diabetes.pkl
│   ├── requirements.txt
│   └── templates/index.html
│
├── IPL Win Predictor/
│   ├── app.py
│   ├── pipe.pkl
│   ├── matches.csv
│   └── deliveries.csv
│
├── Laptop Price Predictor/
│   ├── app.py
│   ├── pipe.pkl
│   ├── df.pkl
│   └── laptop_data.csv
│
├── Movie Recommendation Predictor/
│   ├── app.py
│   ├── requirements.txt
│   ├── important_files.7z       ← extract this first!
│   └── movie-recommender-system.ipynb
│
└── README.md
```

---

## 🙌 Author

**Subhrajyoti Das** — [@jyotisubhra625](https://github.com/jyotisubhra625)

> Built as part of a hands-on Machine Learning learning journey 🚀
