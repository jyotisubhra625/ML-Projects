# 🎬 Movie Recommendation System

A content-based movie recommender built with **Streamlit** and **scikit-learn**. Select any movie from a dataset of 5000 TMDB movies and instantly get 5 similar recommendations with posters fetched live from the TMDB API.

---

## ✨ How It Works

1. Movie metadata (title, genres, keywords, cast, crew, overview) from the TMDB 5000 dataset is preprocessed and vectorized using **CountVectorizer**.
2. **Cosine similarity** is computed between all movie vectors and saved as `similarity.pkl`.
3. When you pick a movie, the app looks up the top 5 most similar movies by cosine distance and fetches their posters from the TMDB API.

---

## 📁 Project Structure

```
Movie Recommendation Predictor/
│
├── app.py                          # Streamlit web app
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── movie-recommender-system.ipynb  # Data preprocessing & model notebook
│
├── important_files.7z              # Compressed model files (extract before running!)
│   ├── movie_dict.pkl              #   → Preprocessed movie DataFrame
│   └── similarity.pkl              #   → Cosine similarity matrix
│
└── tmdb_5000_movies_and_credits.7z # Raw dataset (optional, for retraining)
```

---

## ⚙️ Setup & Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/jyotisubhra625/ML-Projects.git
cd "ML-Projects/Machine Learning/Movie Recommendation Predictor"
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it:
- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Extract the Model Files ⚠️

> **This step is required.** The pickle files are compressed to keep the repo size manageable.

Extract `important_files.7z` into the **root of the project folder** so that both files sit alongside `app.py`:

```
Movie Recommendation Predictor/
├── app.py
├── movie_dict.pkl    ✅ must be here
└── similarity.pkl    ✅ must be here
```

You can use [7-Zip](https://www.7-zip.org/) (Windows) or `7z x important_files.7z` in the terminal.

### Step 5 — Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit |
| **ML Model** | scikit-learn (CountVectorizer + Cosine Similarity) |
| **Movie Posters** | TMDB API |
| **Data Processing** | Pandas, NumPy |
| **Language** | Python 3.9+ |

---

## 📊 Dataset

Uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

The full preprocessing pipeline is documented in `movie-recommender-system.ipynb`.

---

## 🙌 Acknowledgements

- Movie data from [The Movie Database (TMDB)](https://www.themoviedb.org/)
- Dataset sourced from [Kaggle](https://www.kaggle.com/)

---

> Built with ❤️ using Streamlit and scikit-learn
