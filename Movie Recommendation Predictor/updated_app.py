import os
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# File Paths & Constants
MOVIE_DICT_PATH = Path('model/movie_dict.pkl')
TFIDF_MATRIX_PATH = Path('model/tfidf_matrix.npz')
DATASET_PATH = Path('TMDB_movie_dataset_v11.csv')
PLACEHOLDER_POSTER = 'https://via.placeholder.com/500x750?text=No+Poster'
TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'
MAX_ROWS_FOR_RECOMMENDATIONS = 20000
RECOMMENDATION_LIMIT = 5

HOVER_CSS = '''
<style>
.movie-card {
    transition: transform 0.25s ease-in-out;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    width: 100%;
}
.movie-card:hover {
    transform: scale(1.08);
}
.movie-card-container {
    text-align: center;
}
.movie-card-link {
    text-decoration: none;
    color: inherit;
}
.movie-card-title {
    margin: 6px 0 2px;
    font-weight: 600;
}
.movie-card-caption {
    margin-top: 2px;
    color: #999;
    font-size: 0.85rem;
}
</style>
'''

# Initialize Session State
if 'selected_recommendation' not in st.session_state:
    st.session_state.selected_recommendation = None
if 'recommended' not in st.session_state:
    st.session_state.recommended = []
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False


def get_tmdb_api_key():
    try:
        return st.secrets['tmdb']['api_key']
    except Exception:
        return os.getenv('TMDB_API_KEY', '')


@st.cache_data
def fetch_poster(movie_id, api_key, poster_path_db=None):
    """Fetch a poster URL without making unnecessary API calls."""
    if poster_path_db is not None and str(poster_path_db).strip():
        return f'{TMDB_IMAGE_BASE_URL}{poster_path_db}'

    api_key = str(api_key or '').strip()
    if not api_key:
        return PLACEHOLDER_POSTER

    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'
    try:
        response = requests.get(url, timeout=5)
        if response.ok:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f'{TMDB_IMAGE_BASE_URL}{poster_path}'
    except Exception:
        pass
    return PLACEHOLDER_POSTER


def display_poster(image_url, width):
    st.image(image_url, width=width)
    if image_url == PLACEHOLDER_POSTER:
        st.caption('Poster unavailable')


def clean_words(text):
    return re.findall(r'\w+', str(text).lower())


def jaccard_similarity(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def title_similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


@st.cache_data
def load_data():
    """Load a bounded subset of metadata so the app stays responsive on large datasets."""
    movies_df = None

    if MOVIE_DICT_PATH.exists():
        try:
            with MOVIE_DICT_PATH.open('rb') as handle:
                movies_data = pickle.load(handle)
            movies_df = pd.DataFrame(movies_data) if movies_data is not None else None
        except Exception as exc:
            st.warning(f'Could not load pickle metadata: {exc}')

    if movies_df is None or movies_df.empty:
        if DATASET_PATH.exists():
            try:
                movies_df = pd.read_csv(
                    DATASET_PATH,
                    usecols=['title', 'movie_id', 'tags', 'poster_path'],
                    nrows=MAX_ROWS_FOR_RECOMMENDATIONS,
                    low_memory=False,
                )
            except Exception:
                movies_df = pd.read_csv(DATASET_PATH, nrows=MAX_ROWS_FOR_RECOMMENDATIONS, low_memory=False)

    if movies_df is None or movies_df.empty:
        st.error('No movie metadata found. Please make sure the CSV or pickle file exists.')
        return pd.DataFrame()

    movies_df = movies_df.copy()

    if 'title' in movies_df.columns:
        movies_df['title'] = movies_df['title'].fillna('').astype(str).str.strip()
    if 'tags' in movies_df.columns:
        movies_df['tags'] = movies_df['tags'].fillna('').astype(str)
    if 'poster_path' in movies_df.columns:
        movies_df['poster_path'] = movies_df['poster_path'].fillna('')
    if 'movie_id' not in movies_df.columns:
        movies_df['movie_id'] = range(len(movies_df))

    movies_df = movies_df[movies_df['title'].ne('')].drop_duplicates(subset=['title'], keep='first')

    if len(movies_df) > MAX_ROWS_FOR_RECOMMENDATIONS:
        movies_df = movies_df.sample(n=MAX_ROWS_FOR_RECOMMENDATIONS, random_state=42)

    return movies_df


movies = load_data()


def recommend(movie_title):
    """Compute a small set of lightweight recommendations without loading the full TF-IDF matrix."""
    if movies.empty:
        return []

    selected_mask = movies['title'].eq(movie_title)
    if not selected_mask.any():
        return []

    selected_row = movies.loc[selected_mask].iloc[0]
    selected_tags = clean_words(selected_row.get('tags', ''))
    selected_title = str(selected_row.get('title', '')).lower()

    scored_rows = []
    for idx, row in movies.loc[~selected_mask].iterrows():
        other_tags = clean_words(row.get('tags', ''))
        tag_score = jaccard_similarity(selected_tags, other_tags)
        title_score = title_similarity(selected_title, row.get('title', ''))
        combined_score = (0.75 * tag_score) + (0.25 * title_score)

        if combined_score > 0:
            scored_rows.append((idx, combined_score))

    scored_rows.sort(key=lambda item: item[1], reverse=True)
    top_rows = scored_rows[:RECOMMENDATION_LIMIT]

    recommended = []
    for idx, score in top_rows:
        row = movies.loc[idx]
        match_percentage = int(min(100, max(1, score * 100)))
        recommended.append(
            {
                'title': row['title'],
                'poster': fetch_poster(row['movie_id'], get_tmdb_api_key(), row.get('poster_path')),
                'score': match_percentage,
                'movie_id': row['movie_id'],
                'tags': row.get('tags', ''),
            }
        )

    return recommended


# Streamlit UI
st.header('Movie Recommender System')
st.info('The app is running in lightweight mode to avoid loading the full 1M-row TF-IDF matrix and keep the interface responsive.')

if movies.empty:
    st.stop()

movie_list = movies['title'].tolist()
selected_movie = st.selectbox(
    'Type or select a movie from the dropdown',
    movie_list,
    index=0,
)

selected_row = movies[movies['title'] == selected_movie].iloc[0]
selected_movie_id = selected_row['movie_id']
selected_poster = fetch_poster(
    selected_movie_id,
    get_tmdb_api_key(),
    selected_row.get('poster_path')
)

if st.button('Show Recommendation'):
    with st.spinner('Computing recommendations...'):
        st.session_state.recommended = recommend(selected_movie)
        st.session_state.show_recommendations = True

st.subheader('Selected Movie')
selected_col1, selected_col2 = st.columns([1, 2])
with selected_col1:
    display_poster(selected_poster, width=300)
with selected_col2:
    st.markdown(f'**{selected_movie}**')
    st.write('Movie ID:', selected_movie_id)
    with st.expander('Movie details'):
        st.write(selected_row.get('tags', 'No description available.'))

if st.session_state.show_recommendations:
    recommended = st.session_state.recommended
    if not recommended:
        st.warning('No recommendations found for the selected movie.')
    else:
        st.markdown('### Recommended movies')
        st.markdown(HOVER_CSS, unsafe_allow_html=True)
        cols = st.columns(RECOMMENDATION_LIMIT)
        for i, movie_info in enumerate(recommended):
            with cols[i]:
                st.markdown(
                    f'<div class="movie-card-container">'
                    f'  <img class="movie-card" src="{movie_info["poster"]}" width="180" />'
                    f'  <div class="movie-card-title">{movie_info["title"]}</div>'
                    f'  <div class="movie-card-caption">Match: {movie_info["score"]}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button('View details', key=f'detail_btn_{i}'):
                    st.session_state.selected_recommendation = movie_info

if st.session_state.selected_recommendation:
    info = st.session_state.selected_recommendation
    st.markdown('### Selected recommendation details')
    st.write('**Title:**', info['title'])
    st.write('**Movie ID:**', info['movie_id'])
    st.write('**Match score:**', f"{info['score']}%")
    st.write('**Tags:**', info['tags'] or 'No extra details available.')