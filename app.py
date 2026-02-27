import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
from numpy import load

from content_based_filtering import content_recommendation
from hybrid_recommendations import HybridRecommenderSystem


st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="music_note",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg-1: #f4f8ff;
        --bg-2: #eaf5ef;
        --ink-1: #0f1f2e;
        --ink-2: #25394d;
        --card: rgba(255, 255, 255, 0.90);
        --line: #d8e3ef;
        --accent-1: #0ea271;
        --accent-2: #0a7ea0;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(82, 195, 168, 0.22), transparent 36%),
            radial-gradient(circle at 88% 14%, rgba(96, 146, 235, 0.20), transparent 34%),
            linear-gradient(140deg, var(--bg-1) 0%, var(--bg-2) 100%);
        color: var(--ink-1);
    }

    .main .block-container {
        max-width: 1120px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stText"] {
        color: var(--ink-2);
    }

    h1, h2, h3 {
        color: var(--ink-1);
    }

    .hero {
        background: linear-gradient(
            125deg,
            rgba(255, 255, 255, 0.95) 0%,
            rgba(255, 255, 255, 0.86) 60%,
            rgba(230, 247, 243, 0.92) 100%
        );
        border: 1px solid #dce8f2;
        border-radius: 22px;
        padding: 1.8rem 1.6rem 1.5rem 1.6rem;
        box-shadow: 0 16px 40px rgba(21, 44, 63, 0.12);
        margin: 0 auto 1.2rem auto;
        text-align: center;
    }

    .hero h1 {
        margin: 0;
        color: var(--ink-1);
        font-size: 2.75rem;
        line-height: 1.15;
        font-weight: 900;
        letter-spacing: 0.01em;
    }

    .hero p {
        margin: 0.5rem auto 0 auto;
        color: var(--ink-2);
        font-size: 1.06rem;
        max-width: 820px;
    }

    .ui-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: 0 8px 22px rgba(24, 47, 67, 0.08);
        margin-bottom: 0.9rem;
    }

    .card-title {
        margin: 0 0 0.4rem 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #1c3448;
    }

    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label {
        color: #1f3a4e;
        font-weight: 700;
    }

    [data-baseweb="select"] > div {
        background: #ffffff;
        border: 1px solid #c8d8e7;
    }

    [data-baseweb="select"] * {
        color: #152b3d !important;
    }

    [data-baseweb="select"] input {
        color: #1c4f66 !important;
        -webkit-text-fill-color: #1c4f66 !important;
        font-weight: 600;
    }

    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--accent-2);
    }

    .stButton button {
        width: 100%;
        border: none;
        border-radius: 13px;
        background: linear-gradient(90deg, var(--accent-1) 0%, var(--accent-2) 100%);
        color: white;
        padding: 0.7rem 1rem;
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(10, 126, 160, 0.26);
    }

    .stButton button:hover {
        filter: brightness(1.07);
        transform: translateY(-1px);
    }

    .result-card {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #d3e2ef;
        border-radius: 14px;
        padding: 0.9rem 1rem 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }

    .result-title {
        color: #123248;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .diversity-hints {
        display: flex;
        justify-content: space-between;
        margin-top: -0.3rem;
        margin-bottom: 0.25rem;
        font-size: 0.83rem;
        color: #35586b;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================
# LOAD DATA
# ============================================
cleaned_data_path = "data/cleaned_data.csv"
songs_data = pd.read_csv(cleaned_data_path)

transformed_data = load_npz("data/transformed_data.npz")
track_ids = load("data/track_ids.npy", allow_pickle=True)
filtered_data = pd.read_csv("data/collab_filtered_data.csv")
interaction_matrix = load_npz("data/interaction_matrix.npz")
transformed_hybrid_data = load_npz("data/transformed_hybrid_data.npz")


# ============================================
# TITLE
# ============================================
st.markdown(
    """
    <div class="hero">
        <h1>Spotify Hybrid Recommendation System</h1>
        <p>Discover tracks using a smart blend of content-based and collaborative recommendation.</p>
        <p>Search by song, pick the right artist, and tune diversity for your listening style.</p>
        <p>Enjoy a clean modern interface with fast, preview-ready results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================
# PREPARE DATA FOR SEARCH
# ============================================
@st.cache_data
def prepare_data(df):
    df = df.copy()
    df["name_clean"] = df["name"].astype(str).str.lower().str.strip()
    df["artist_clean"] = df["artist"].astype(str).str.lower().str.strip()
    return df


songs_data = prepare_data(songs_data)
filtered_data = prepare_data(filtered_data)


# ============================================
# SONG SEARCH (INSTANT)
# ============================================
controls_left, controls_right = st.columns([1.45, 1.0], gap="large")

with controls_left:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Track Selection</div>', unsafe_allow_html=True)
    song_display = st.selectbox(
        "Search Song",
        options=sorted(songs_data["name"].unique()),
        index=None,
        placeholder="Type to search...",
    )

    if song_display:
        artist_options = songs_data.loc[
            songs_data["name"] == song_display, "artist"
        ].unique()

        artist_display = st.selectbox(
            "Artist",
            options=artist_options,
            index=0,
        )
    else:
        artist_display = None
        st.selectbox(
            "Artist",
            options=["Select a song first"],
            index=0,
            disabled=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================
# CLEAN VALUES FOR MODEL
# ============================================
if song_display and artist_display:
    song_name = song_display.lower().strip()
    artist_name = artist_display.lower().strip()
else:
    song_name = None
    artist_name = None


# ============================================
# SETTINGS
# ============================================
with controls_right:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Recommendation Settings</div>', unsafe_allow_html=True)
    k = st.selectbox("How many recommendations?", [5, 10, 15, 20], index=1)


# ============================================
# CHECK HYBRID AVAILABLE
# ============================================
if song_name and artist_name:
    hybrid_available = (
        (filtered_data["name_clean"] == song_name)
        & (filtered_data["artist_clean"] == artist_name)
    ).any()
else:
    hybrid_available = False

if hybrid_available:
    filtering_type = "Hybrid"

    with controls_right:
        diversity = st.slider(
            "Diversity in Recommendations",
            min_value=1,
            max_value=9,
            value=5,
        )
        st.markdown(
            '<div class="diversity-hints"><span>Personalized</span><span>Diverse</span></div>',
            unsafe_allow_html=True,
        )

    content_weight = 1 - (diversity / 10)
else:
    filtering_type = "Content"

with controls_right:
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================
# BUTTON
# ============================================
if st.button("Get Recommendations"):
    if not song_name or not artist_name:
        st.warning("Please select a song first")
        st.stop()

    st.subheader(f"Recommendations for {song_display} - {artist_display}")

    # ======================================
    # CONTENT BASED
    # ======================================
    if filtering_type == "Content":
        try:
            recommendations = content_recommendation(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=songs_data,
                transformed_data=transformed_data,
                k=k,
            )
        except Exception:
            st.error("Song not found in content model")
            st.stop()

    # ======================================
    # HYBRID
    # ======================================
    else:
        try:
            recommender = HybridRecommenderSystem(
                number_of_recommendations=k,
                weight_content_based=content_weight,
            )

            recommendations = recommender.give_recommendations(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=filtered_data,
                transformed_matrix=transformed_hybrid_data,
                track_ids=track_ids,
                interaction_matrix=interaction_matrix,
            )
        except Exception:
            st.error("Hybrid recommender failed")
            st.stop()

    # ======================================
    # SHOW RESULTS (YOUTUBE STYLE)
    # ======================================
    for i, row in recommendations.iterrows():
        name = row["name"].title()
        artist = row["artist"].title()

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if i == 0:
            st.markdown(
                '<div class="result-title">Currently Playing</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{name}** - **{artist}**")
        else:
            st.markdown(
                f'<div class="result-title">{i}. {name} - {artist}</div>',
                unsafe_allow_html=True,
            )

        if pd.notna(row["spotify_preview_url"]):
            st.audio(row["spotify_preview_url"])

        st.markdown("</div>", unsafe_allow_html=True)
