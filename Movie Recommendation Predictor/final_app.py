"""
Movie Recommender System — Groq-powered build.

Setup:
    pip install streamlit pandas requests python-dotenv groq

Secrets (.streamlit/secrets.toml):
    TMDB_API_KEY = "your_tmdb_key_here"
    GROQ_API_KEY = "your_groq_key_here"

(Both also fall back to environment variables of the same name — e.g. a
.env file — and the app still runs, with reduced features, if either key
is missing.)
"""

import json
import os
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
except ImportError:
    Groq = None


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
MOVIE_DICT_PATH = Path('model/movie_dict.pkl')
DATASET_PATH = Path('TMDB_movie_dataset_v11.csv')
PLACEHOLDER_POSTER = 'https://via.placeholder.com/500x750?text=No+Poster'
TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'
MAX_ROWS_FOR_RECOMMENDATIONS = 50000
RECOMMENDATION_LIMIT = 5
GROQ_MODEL = 'llama-3.3-70b-versatile'

# NOTE: this is a widely shared "tutorial" key used by thousands of similar
# projects online, so TMDB frequently rate-limits it (HTTP 429). Posters are
# much more reliable once you register your own free key at
# https://www.themoviedb.org/settings/api and put it in
# .streamlit/secrets.toml as:  TMDB_API_KEY = "your_key_here"
DEFAULT_TMDB_API_KEY = '8265bd1679663a7ea12ac168da84d2e8'
try:
    TMDB_API_KEY = st.secrets['TMDB_API_KEY']
except Exception:
    TMDB_API_KEY = os.getenv('TMDB_API_KEY', DEFAULT_TMDB_API_KEY)

try:
    GROQ_API_KEY = st.secrets['GROQ_API_KEY']
except Exception:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

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
.movie-card-title {
    margin: 6px 0 2px;
    font-weight: 600;
}
.movie-card-caption {
    margin-top: 2px;
    color: #999;
    font-size: 0.85rem;
}
.movie-card-explanation {
    margin-top: 4px;
    font-size: 0.82rem;
    font-style: italic;
    color: #bbb;
}
</style>
'''

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULT_STATE = {
    'selected_recommendation': None,
    'recommended': [],
    'show_recommendations': False,
    'recommend_explanations': {},
    'vibe_results': [],
    'show_vibe_results': False,
    'vibe_used_ai': False,
    'vibe_keywords': [],
    'chat_history': [],
    '_groq_error': None,
}
for key, default in _DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# TMDB session with retry/backoff
# ---------------------------------------------------------------------------
def _build_tmdb_session():
    """A session that automatically retries on rate-limit / server errors
    with exponential backoff, instead of failing on the first 429."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,  # waits ~1.5s, 3s, 6s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET'],
        respect_retry_after_header=True,
    )
    session.mount('https://', HTTPAdapter(max_retries=retry))
    return session


_tmdb_session = _build_tmdb_session()


def _tmdb_key_fingerprint():
    """A short, non-secret stand-in for the active API key.

    st.cache_data keys its cache purely off a function's *arguments*, not
    off external globals like TMDB_API_KEY. Passing this fingerprint as an
    argument makes the cache automatically treat "new key" as "new call",
    so swapping keys re-fetches instead of reusing stale cached results.
    """
    return TMDB_API_KEY[-6:] if TMDB_API_KEY else 'none'


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id, key_fingerprint=None, poster_path_db=None):
    """Fetch a poster URL, preferring the poster path already present in
    the dataset (no network call needed) and only hitting the TMDB API
    when that's missing."""
    if poster_path_db is not None and str(poster_path_db).strip():
        return f'{TMDB_IMAGE_BASE_URL}{poster_path_db}'

    url = (
        'https://api.themoviedb.org/3/movie/{}'
        '?api_key={}&language=en-US'
    ).format(movie_id, TMDB_API_KEY)
    try:
        response = _tmdb_session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if not poster_path:
            return PLACEHOLDER_POSTER
        return f'{TMDB_IMAGE_BASE_URL}{poster_path}'
    except Exception:
        return PLACEHOLDER_POSTER


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trending(key_fingerprint=None):
    """Pulls TMDB's live 'trending this week' list, independent of our
    static dataset, so the app always has a feed that isn't purely
    recommendation-driven."""
    url = (
        'https://api.themoviedb.org/3/trending/movie/week'
        '?api_key={}&language=en-US'
    ).format(TMDB_API_KEY)
    try:
        response = _tmdb_session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        trending = []
        for item in data.get('results', [])[:8]:
            poster_path = item.get('poster_path')
            trending.append(
                {
                    'title': item.get('title') or item.get('name') or 'Untitled',
                    'poster': (
                        f'{TMDB_IMAGE_BASE_URL}{poster_path}'
                        if poster_path else PLACEHOLDER_POSTER
                    ),
                    'movie_id': item.get('id'),
                    'rating': item.get('vote_average'),
                }
            )
        return trending
    except Exception:
        return []


def display_poster(image_url, width):
    st.image(image_url, width=width)
    if image_url == PLACEHOLDER_POSTER:
        st.caption('Poster unavailable')


# ---------------------------------------------------------------------------
# Text similarity helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Data loading — pickle first, CSV fallback, tags/ids auto-built, deduped
# ---------------------------------------------------------------------------
@st.cache_data
def load_movies():
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
            csv_columns = ['id', 'title', 'overview', 'keywords', 'genres', 'poster_path']
            try:
                movies_df = pd.read_csv(
                    DATASET_PATH,
                    usecols=csv_columns,
                    nrows=MAX_ROWS_FOR_RECOMMENDATIONS,
                    low_memory=False,
                )
            except Exception:
                movies_df = pd.read_csv(DATASET_PATH, nrows=MAX_ROWS_FOR_RECOMMENDATIONS, low_memory=False)

    if movies_df is None or movies_df.empty:
        st.error('No movie metadata found. Please make sure movie_dict.pkl or the dataset CSV exists.')
        return pd.DataFrame()

    movies_df = movies_df.copy()

    if 'title' in movies_df.columns:
        movies_df['title'] = movies_df['title'].fillna('').astype(str).str.strip()

    if 'poster_path' in movies_df.columns:
        movies_df['poster_path'] = movies_df['poster_path'].fillna('')
    else:
        movies_df['poster_path'] = ''

    if 'movie_id' not in movies_df.columns:
        movies_df['movie_id'] = movies_df['id'] if 'id' in movies_df.columns else range(len(movies_df))

    if 'tags' not in movies_df.columns:
        tag_parts = [
            movies_df[col].fillna('').astype(str)
            for col in ['overview', 'keywords', 'genres']
            if col in movies_df.columns
        ]
        if tag_parts:
            tags = tag_parts[0]
            for part in tag_parts[1:]:
                tags = tags + ' ' + part
            movies_df['tags'] = tags
        else:
            movies_df['tags'] = ''
    else:
        movies_df['tags'] = movies_df['tags'].fillna('').astype(str)

    movies_df = (
        movies_df[movies_df['title'].ne('')]
        .drop_duplicates(subset=['title'], keep='first')
        .reset_index(drop=True)
    )
    return movies_df


# ---------------------------------------------------------------------------
# Groq integration
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_groq_client():
    if Groq is None:
        st.session_state['_groq_error'] = (
            'The `groq` package is not importable. Run `pip install -U groq` '
            'in the SAME Python environment that runs `streamlit run`.'
        )
        return None
    if not GROQ_API_KEY:
        st.session_state['_groq_error'] = (
            'GROQ_API_KEY resolved to an empty string. Check that '
            '.streamlit/secrets.toml (relative to where you run `streamlit '
            'run`) contains a top-level line `GROQ_API_KEY = "..."`, or that '
            'the environment variable / .env is set in the same session '
            'that launches Streamlit.'
        )
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        # Force a lightweight call so a bad/expired key fails now, not later.
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': 'ping'}],
            max_completion_tokens=5,
        )
        st.session_state['_groq_error'] = None
        return client
    except Exception as exc:
        st.session_state['_groq_error'] = f'Groq client failed to initialize: {exc}'
        return None


def groq_available():
    return get_groq_client() is not None


def _groq_generate_text(prompt, fallback=''):
    client = get_groq_client()
    if client is None:
        return fallback
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = (response.choices[0].message.content or '').strip()
        return text or fallback
    except Exception:
        return fallback


def _groq_generate_json(prompt, fallback=None):
    client = get_groq_client()
    if client is None:
        return fallback
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={'type': 'json_object'},
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return fallback


def ai_extract_vibe_keywords(query):
    """Feature 1: ask the LLM to translate a free-text mood/vibe description
    into concrete genre/theme/tone keywords, which we then match against
    the dataset's tags. Falls back to plain tokenizing of the query if the
    AI is unavailable or the call fails."""
    prompt = (
        'A user is describing a movie mood/vibe they want to watch. '
        'Extract 6-10 short genre, theme, or tone keywords implied by the '
        'request (single words or short phrases, lowercase). '
        'Return ONLY a JSON object of the form {"keywords": ["...", ...]}. '
        f'User request: "{query}"'
    )
    data = _groq_generate_json(prompt, fallback=None)
    if isinstance(data, dict) and data.get('keywords'):
        return [str(k).strip().lower() for k in data['keywords'] if str(k).strip()]
    return None


def ai_explain_recommendations(basis_titles, recommended):
    """Feature 2: batch-generate one-line, natural-language 'why you'll
    like this' explanations for a list of recommended movies."""
    if not recommended:
        return {}
    items = [
        {'title': r['title'], 'matched_tags': r.get('matched_tags', [])}
        for r in recommended
    ]
    prompt = (
        f'A user likes these movies: {", ".join(basis_titles)}. '
        'For each candidate below, write ONE short, friendly sentence '
        '(max 18 words) explaining why they might enjoy it, referencing '
        'shared themes where relevant. Vary the phrasing across sentences. '
        f'Candidates (JSON): {json.dumps(items)} '
        'Return ONLY a JSON object mapping each exact movie title to its '
        'one-sentence explanation.'
    )
    data = _groq_generate_json(prompt, fallback=None)
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def ai_chat_reply(user_message, chat_history, movies_df):
    """Feature 3: a conversational assistant that grounds its answers in a
    few relevant movies pulled from the dataset via Jaccard matching, then
    lets the LLM phrase a natural reply."""
    query_tokens = set(clean_words(user_message))
    scored = []
    if query_tokens:
        for _, row in movies_df.iterrows():
            score = jaccard_similarity(query_tokens, set(clean_words(row.get('tags', ''))))
            if score > 0:
                scored.append((score, row['title'], str(row.get('tags', ''))[:200]))
    scored.sort(key=lambda x: x[0], reverse=True)
    context_movies = scored[:6]
    context_text = '\n'.join(f'- {t}: {tags}' for _, t, tags in context_movies) or 'No close dataset matches found.'

    history_text = '\n'.join(f"{m['role']}: {m['content']}" for m in chat_history[-6:])

    prompt = (
        'You are a friendly movie recommendation assistant embedded in a '
        'Streamlit app. Use the dataset context below when it is relevant, '
        'and your own general movie knowledge otherwise. Keep replies '
        'conversational and under 120 words.\n\n'
        f'Dataset context (candidate movies and tags):\n{context_text}\n\n'
        f'Conversation so far:\n{history_text}\n\n'
        f'User: {user_message}\nAssistant:'
    )
    fallback = (
        "The chat assistant needs a working Groq API key to respond — "
        "check that GROQ_API_KEY is set in .streamlit/secrets.toml."
    )
    return _groq_generate_text(prompt, fallback=fallback)


# ---------------------------------------------------------------------------
# Core recommendation logic
# ---------------------------------------------------------------------------
def vibe_search(query, movies_df, top_n=5):
    """Matches a free-text mood/vibe description against each movie's tag
    text using Jaccard scoring, augmented with AI-extracted keywords when
    available so the match reflects the *meaning* of the request, not just
    its literal words."""
    query_tokens = set(clean_words(query))
    if not query_tokens:
        return [], False, []

    ai_keywords = ai_extract_vibe_keywords(query) if groq_available() else None
    used_ai = bool(ai_keywords)
    if ai_keywords:
        for kw in ai_keywords:
            query_tokens |= set(clean_words(kw))

    scores = []
    for idx, row in movies_df.iterrows():
        other_tags = set(clean_words(row.get('tags', '')))
        score = jaccard_similarity(query_tokens, other_tags)
        if score > 0:
            scores.append((idx, score, query_tokens & other_tags))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scores[:top_n]

    results = []
    for idx, score, shared_tags in top_candidates:
        movie_id = movies_df.loc[idx, 'movie_id']
        results.append(
            {
                'title': movies_df.loc[idx, 'title'],
                'poster': fetch_poster(movie_id, _tmdb_key_fingerprint(), movies_df.loc[idx, 'poster_path']),
                'score': round(score * 100),
                'movie_id': movie_id,
                'matched_tags': sorted(shared_tags, key=len, reverse=True)[:5],
            }
        )
    return results, used_ai, (ai_keywords or [])


def recommend(movie_titles, movies_df, top_n=RECOMMENDATION_LIMIT):
    """Recommend movies similar to a blend of 1-3 selected movies.

    Accepts either a single title (str) or a list of up to 3 titles. Tag
    profiles of all selected movies are merged (union of tokens) so the
    result reflects the combined taste of the whole group, and title
    similarity is taken as the best match against any selected title.
    """
    if isinstance(movie_titles, str):
        movie_titles = [movie_titles]

    valid_titles = [t for t in movie_titles if t in movies_df['title'].values]
    if not valid_titles:
        return []

    selected_rows = [movies_df[movies_df['title'] == t].iloc[0] for t in valid_titles]

    combined_tags = set()
    for row in selected_rows:
        combined_tags |= set(clean_words(row.get('tags', '')))

    selected_titles_clean = [row.get('title', '') for row in selected_rows]
    excluded_titles = set(valid_titles)

    scores = []
    for idx, row in movies_df.iterrows():
        if row['title'] in excluded_titles:
            continue
        other_tags = set(clean_words(row.get('tags', '')))
        shared_tags = combined_tags & other_tags
        tag_score = jaccard_similarity(combined_tags, other_tags)
        title_score = max(
            title_similarity(t, row.get('title', '')) for t in selected_titles_clean
        )
        score = 0.75 * tag_score + 0.25 * title_score
        scores.append((idx, score, shared_tags))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scores[:top_n]

    recommended = []
    for idx, score, shared_tags in top_candidates:
        movie_id = movies_df.loc[idx, 'movie_id']
        top_shared = sorted(shared_tags, key=len, reverse=True)[:5]
        recommended.append(
            {
                'title': movies_df.loc[idx, 'title'],
                'poster': fetch_poster(movie_id, _tmdb_key_fingerprint(), movies_df.loc[idx, 'poster_path']),
                'score': round(score * 100),
                'movie_id': movie_id,
                'tags': movies_df.loc[idx, 'tags'] if 'tags' in movies_df.columns else '',
                'matched_tags': top_shared,
            }
        )

    return recommended


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.header('🎬 Movie Recommender System')
movies = load_movies()

if movies.empty:
    st.stop()

with st.sidebar:
    st.subheader('Poster source (TMDB)')
    if TMDB_API_KEY == DEFAULT_TMDB_API_KEY:
        st.warning(
            'Using the shared demo TMDB key. This key is reused by many '
            'tutorial projects and gets rate-limited often. Add your own '
            'free key in `.streamlit/secrets.toml` as `TMDB_API_KEY` for '
            'reliable poster loading.'
        )
    else:
        st.success(f'Using your own TMDB key (ending in ...{TMDB_API_KEY[-4:]})')

    st.caption(
        'If posters are stuck showing "Poster unavailable" from before you '
        'added your key, use the button below to force a fresh fetch.'
    )
    if st.button('🔄 Reset poster cache'):
        fetch_poster.clear()
        st.success('Poster cache cleared — posters will be re-fetched.')
        st.rerun()

    st.divider()
    st.subheader('Gen AI (Groq)')
    if groq_available():
        st.success('Groq API connected — smart vibe search, explanations, and chat are live.')
    else:
        st.warning(
            'Groq is not active. The app still works without it, using '
            'plain keyword matching instead.'
        )
        reason = st.session_state.get('_groq_error')
        if reason:
            st.caption(f'Reason: {reason}')
        if st.button('🔄 Retry Groq connection'):
            get_groq_client.clear()
            st.rerun()

st.subheader('🔥 Trending This Week')
st.caption('Live from TMDB — not from our own dataset, refreshes hourly.')
trending = fetch_trending(_tmdb_key_fingerprint())
if trending:
    trend_cols = st.columns(len(trending))
    for col, item in zip(trend_cols, trending):
        with col:
            display_poster(item['poster'], width=110)
            st.caption(item['title'])
            if item.get('rating'):
                st.caption(f"⭐ {item['rating']:.1f}")
else:
    st.caption('Trending data unavailable right now.')

st.divider()

st.subheader('🎭 Search by Mood or Vibe')
st.caption(
    'Powered by Groq when available — it interprets the *meaning* of '
    'your description, not just the literal words.'
)
vibe_query = st.text_input(
    'Describe what you\'re in the mood for (e.g. "dark twisty thriller" or "feel-good adventure")'
)
if st.button('Search by vibe'):
    with st.spinner('Matching your vibe...'):
        results, used_ai, keywords = vibe_search(vibe_query, movies)
        st.session_state.vibe_results = results
        st.session_state.vibe_used_ai = used_ai
        st.session_state.vibe_keywords = keywords
        st.session_state.show_vibe_results = True

if st.session_state.show_vibe_results:
    vibe_results = st.session_state.vibe_results
    if not vibe_results:
        st.warning('No movies matched that description — try different words.')
    else:
        st.markdown(f'**Movies matching:** "{vibe_query}"')
        if st.session_state.vibe_used_ai and st.session_state.vibe_keywords:
            st.caption('✨ Groq interpreted this as: ' + ', '.join(st.session_state.vibe_keywords))
        vibe_cols = st.columns(len(vibe_results))
        for col, item in zip(vibe_cols, vibe_results):
            with col:
                display_poster(item['poster'], width=140)
                st.caption(item['title'])
                st.caption(f"Match: {item['score']}%")
                if item.get('matched_tags'):
                    st.caption('Matched on: ' + ', '.join(item['matched_tags']))

st.divider()

st.subheader('Pick Movies You Like')
movie_list = movies['title'].tolist()
selected_movies = st.multiselect(
    'Pick 1 to 3 movies you like — recommendations will blend all of them',
    movie_list,
    default=[movie_list[0]] if movie_list else [],
    max_selections=3,
)

if not selected_movies:
    st.info('Select at least one movie to get started.')
    st.stop()

if st.button('Show Recommendation'):
    with st.spinner('Computing recommendations...'):
        recommended = recommend(selected_movies, movies)
        st.session_state.recommended = recommended
        st.session_state.show_recommendations = True
    if groq_available():
        with st.spinner('Asking Groq why you\'ll like these...'):
            st.session_state.recommend_explanations = ai_explain_recommendations(
                selected_movies, recommended
            )
    else:
        st.session_state.recommend_explanations = {}

st.subheader('Selected Movie' if len(selected_movies) == 1 else 'Selected Movies')
selected_cols = st.columns(len(selected_movies))
for col, title in zip(selected_cols, selected_movies):
    row = movies[movies['title'] == title].iloc[0]
    with col:
        display_poster(
            fetch_poster(row['movie_id'], _tmdb_key_fingerprint(), row.get('poster_path')),
            width=200,
        )
        st.markdown(f"**{title}**")
        with st.expander('Movie details'):
            st.write(row.get('tags', 'No description available.'))

if st.session_state.show_recommendations:
    recommended = st.session_state.recommended
    if not recommended:
        st.warning('No recommendations found for the selected movie(s).')
    else:
        basis = ', '.join(selected_movies)
        st.markdown(f'### Because you liked: {basis}')
        st.markdown(HOVER_CSS, unsafe_allow_html=True)
        explanations = st.session_state.recommend_explanations
        cols = st.columns(len(recommended))
        for i, movie_info in enumerate(recommended):
            with cols[i]:
                st.markdown(
                    f'<div class="movie-card-container">'
                    f'  <img class="movie-card" src="{movie_info["poster"]}" width="180" />'
                    f'  <div class="movie-card-title">{movie_info["title"]}</div>'
                    f'  <div class="movie-card-caption">Score: {movie_info["score"]}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                explanation = explanations.get(movie_info['title'])
                if explanation:
                    st.markdown(f'<div class="movie-card-explanation">✨ {explanation}</div>', unsafe_allow_html=True)
                matched = movie_info.get('matched_tags') or []
                if matched:
                    chips = ' '.join(f'`{tag}`' for tag in matched)
                    st.caption(f'Matched on: {chips}')
                if st.button('View details', key=f'detail_btn_{i}'):
                    st.session_state.selected_recommendation = movie_info

if st.session_state.selected_recommendation:
    info = st.session_state.selected_recommendation
    st.markdown('### Selected recommendation details')
    st.write('**Title:**', info['title'])
    st.write('**Movie ID:**', info['movie_id'])
    st.write('**Score:**', f"{info['score']}%")
    if info.get('matched_tags'):
        st.write('**Matched on:**', ', '.join(info['matched_tags']))
    st.write('**Tags:**', info.get('tags') or 'No extra details available.')

st.divider()

st.subheader('💬 Ask the Movie Assistant')
st.caption('Chat with a Groq-powered assistant about movies, moods, or your recommendations.')

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_chat_input = st.chat_input('Ask something like "Suggest a sci-fi movie with a twist ending"')
if user_chat_input:
    st.session_state.chat_history.append({'role': 'user', 'content': user_chat_input})
    with st.chat_message('user'):
        st.write(user_chat_input)
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            reply = ai_chat_reply(user_chat_input, st.session_state.chat_history, movies)
        st.write(reply)
    st.session_state.chat_history.append({'role': 'assistant', 'content': reply})