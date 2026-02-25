import streamlit as st
import pandas as pd
from scipy.sparse import load_npz

from content_based_filtering import content_recommendation

TRANSFORMED_DATA_PATH = "data/transformed_data.npz"
CLEANED_DATA_PATH = "data/cleaned_data.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_features(path: str):
    return load_npz(path)


def main() -> None:
    st.set_page_config(page_title="Spotify Song Recommender", page_icon=":musical_note:", layout="wide")

    st.title("Spotify Hybrid Recommender")
    st.caption("Pick a song and artist to get similar tracks with preview audio.")

    try:
        data = load_data(CLEANED_DATA_PATH)
        transformed_data = load_features(TRANSFORMED_DATA_PATH)
    except Exception as exc:
        st.error(f"Failed to load app data: {exc}")
        return

    song_query = st.text_input("Song name", placeholder="e.g., Wonderwall").strip().lower()

    if song_query:
        song_matches = data[data["name"].str.contains(song_query, na=False)]
    else:
        song_matches = data.iloc[0:0]

    if song_query and song_matches.empty:
        st.warning("No songs matched your input. Try a different title.")
        return

    selected_song = ""
    selected_artist = ""

    if not song_matches.empty:
        options = (
            song_matches[["name", "artist"]]
            .drop_duplicates()
            .sort_values(["name", "artist"])
            .reset_index(drop=True)
        )
        labels = options.apply(lambda r: f"{r['name'].title()} - {r['artist'].title()}", axis=1).tolist()
        picked_label = st.selectbox("Choose exact track", labels, index=0)
        selected = options.iloc[labels.index(picked_label)]
        selected_song = str(selected["name"])
        selected_artist = str(selected["artist"])

    k = st.selectbox("How many recommendations?", [5, 10, 15, 20], index=1)

    if st.button("Get Recommendations", type="primary"):
        if not selected_song or not selected_artist:
            st.info("Enter a song name and select a track first.")
            return

        try:
            recommendations = content_recommendation(
                selected_song,
                selected_artist,
                data,
                transformed_data,
                k,
            )
        except Exception as exc:
            st.error(f"Could not generate recommendations: {exc}")
            return

        st.subheader(f"Recommendations for {selected_song.title()} by {selected_artist.title()}")

        for idx, recommendation in recommendations.iterrows():
            rec_song = str(recommendation["name"]).title()
            rec_artist = str(recommendation["artist"]).title()
            preview_url = recommendation.get("spotify_preview_url")

            if idx == 0:
                st.markdown(f"### Now Playing: {rec_song} - {rec_artist}")
            else:
                st.markdown(f"**{idx}. {rec_song} - {rec_artist}**")

            if isinstance(preview_url, str) and preview_url.strip():
                st.audio(preview_url)
            else:
                st.caption("Preview unavailable for this track.")

            st.divider()


if __name__ == "__main__":
    main()
