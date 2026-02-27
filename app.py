import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
from numpy import load

from content_based_filtering import content_recommendation
from hybrid_recommendations import HybridRecommenderSystem


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
st.title("🎵 Spotify Song Recommender")
st.write("Enter a song name and get similar songs 🎧")


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
song_display = st.selectbox(
    "Search Song",
    options=sorted(songs_data["name"].unique()),
    index=None,
    placeholder="Type to search..."
)

if song_display:
    artist_options = songs_data.loc[
        songs_data["name"] == song_display,
        "artist"
    ].unique()

    artist_display = st.selectbox(
        "Artist",
        options=artist_options,
        index=0
    )
else:
    artist_display = None


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
k = st.selectbox("How many recommendations?", [5, 10, 15, 20], index=1)


# ============================================
# CHECK HYBRID AVAILABLE
# ============================================
if song_name and artist_name:
    hybrid_available = (
        (filtered_data["name_clean"] == song_name) &
        (filtered_data["artist_clean"] == artist_name)
    ).any()
else:
    hybrid_available = False


if hybrid_available:
    filtering_type = "Hybrid"

    diversity = st.slider(
        "Diversity in Recommendations",
        min_value=1,
        max_value=9,
        value=5
    )

    content_weight = 1 - (diversity / 10)

else:
    filtering_type = "Content"


# ============================================
# BUTTON
# ============================================
if st.button("Get Recommendations"):

    if not song_name or not artist_name:
        st.warning("Please select a song first")
        st.stop()

    st.subheader(f"Recommendations for {song_display} – {artist_display}")

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
                k=k
            )
        except Exception as e:
            st.error("Song not found in content model")
            st.stop()

    # ======================================
    # HYBRID
    # ======================================
    else:
        try:
            recommender = HybridRecommenderSystem(
                number_of_recommendations=k,
                weight_content_based=content_weight
            )

            recommendations = recommender.give_recommendations(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=filtered_data,
                transformed_matrix=transformed_hybrid_data,
                track_ids=track_ids,
                interaction_matrix=interaction_matrix
            )
        except Exception as e:
            st.error("Hybrid recommender failed")
            st.stop()

    # ======================================
    # SHOW RESULTS (YOUTUBE STYLE)
    # ======================================
    for i, row in recommendations.iterrows():

        name = row["name"].title()
        artist = row["artist"].title()

        if i == 0:
            st.markdown("## ▶ Currently Playing")
        else:
            st.markdown(f"### {i}. {name} – {artist}")

        if pd.notna(row["spotify_preview_url"]):
            st.audio(row["spotify_preview_url"])

        st.write("---")