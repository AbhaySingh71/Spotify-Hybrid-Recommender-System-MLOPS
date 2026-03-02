# Spotify Hybrid Recommendation System

A music recommendation project that combines **content-based filtering** and **collaborative filtering** into a single **hybrid recommender**, served through a Streamlit app and packaged with an MLOps workflow (DVC + Docker + GitHub Actions + AWS deployment).

## Overview

This project recommends songs using two complementary signals:

- **Content-based filtering**: Similarity from song metadata and audio features.
- **Collaborative filtering**: Similarity from user listening behavior (playcount interactions).
- **Hybrid scoring**: Weighted combination of both similarity scores, controlled by a diversity slider in the UI.

The app automatically falls back to content-based recommendations when collaborative signals are unavailable for a selected track.

## Features

- Streamlit interface with search, artist disambiguation, and recommendation controls.
- Audio preview playback (`spotify_preview_url`) in result cards.
- DVC pipeline for reproducible data artifacts.
- Sparse matrix representations (`.npz`) for scalable similarity computation.
- CI/CD workflow that:
  - pulls DVC data,
  - runs app smoke tests,
  - builds and pushes a Docker image to ECR,
  - triggers AWS CodeDeploy.

## Project Structure

```text
.
+-- app.py                              # Streamlit app
+-- data_cleaning.py                    # Cleans raw music metadata
+-- content_based_filtering.py          # Feature engineering + content similarity
+-- collaborative_filtering.py          # Interaction matrix + collaborative similarity
+-- hybrid_recommendations.py           # Hybrid score blending
+-- transform_filtered_data.py          # Content transform for collab-filtered subset
+-- dvc.yaml                            # Reproducible pipeline definition
+-- Dockerfile                          # Containerized app runtime
+-- .github/workflows/ci.yaml           # CI/CD pipeline
+-- appspec.yml                         # CodeDeploy app spec
+-- deploy/scripts/
    +-- install_dependencies.sh         # EC2 host setup for Docker/AWS CLI
    +-- start_docker.sh                 # Pulls image from ECR and runs container
```

## Data and Artifacts

Raw/managed datasets are tracked with DVC pointers:

- `data/Music Info.csv`
- `data/User Listening History.csv`

Pipeline-generated artifacts:

- `data/cleaned_data.csv`
- `data/transformed_data.npz`
- `transformer.joblib`
- `data/collab_filtered_data.csv`
- `data/track_ids.npy`
- `data/interaction_matrix.npz`
- `data/transformed_hybrid_data.npz`

## Recommendation Pipeline

1. **Data cleaning** (`data_cleaning.py`)
- Deduplicates by `track_id`
- Drops unused columns (`genre`, `spotify_id`)
- Normalizes text fields to lowercase

2. **Content feature transformation** (`content_based_filtering.py`)
- `CountEncoder` on `year`
- `OneHotEncoder` on `artist`, `time_signature`, `key`
- `TF-IDF` on `tags`
- `StandardScaler` + `MinMaxScaler` on numeric audio features
- Saves `transformer.joblib` and `data/transformed_data.npz`

3. **Collaborative matrix creation** (`collaborative_filtering.py`)
- Builds track-user sparse matrix from listening history (`playcount`)
- Saves matrix and aligned track IDs
- Produces collab-supported song subset

4. **Hybrid transform subset** (`transform_filtered_data.py`)
- Applies trained content transformer to the collab-supported subset
- Saves `data/transformed_hybrid_data.npz`

5. **Hybrid inference** (`hybrid_recommendations.py` + `app.py`)
- Computes cosine similarities in both spaces
- Normalizes both scores
- Combines via weighted average:

```text
hybrid_score = w_content * content_similarity + (1 - w_content) * collaborative_similarity
```

## Local Setup

### Prerequisites

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/)
- DVC with S3 remote access configured (for pulling data/artifacts)
- AWS credentials (if your DVC remote is S3)

### Install Dependencies

```bash
uv sync
```

### Pull DVC Data

```bash
uv run dvc pull
```

### Run Streamlit App

```bash
uv run streamlit run app.py --server.port 8000
```

Open: `http://localhost:8000`

### Run Tests

```bash
uv run pytest test_app.py
```

## Reproduce Data Pipeline

Run all stages:

```bash
uv run dvc repro
```

Or run individual scripts in order:

```bash
uv run python data_cleaning.py
uv run python content_based_filtering.py
uv run python collaborative_filtering.py
uv run python transform_filtered_data.py
```

## Docker

Build and run locally:

```bash
docker build -t spotify-hybrid-recsys .
docker run --rm -p 8000:8000 spotify-hybrid-recsys
```

## CI/CD and Deployment

Defined in `.github/workflows/ci.yaml`:

- Checkout code
- Setup Python + uv
- Install dependencies
- Pull DVC artifacts
- Start Streamlit and run smoke test (`test_app.py`)
- Build and push Docker image to Amazon ECR
- Upload deployment bundle to S3
- Trigger AWS CodeDeploy deployment

CodeDeploy hooks:

- `deploy/scripts/install_dependencies.sh`: installs Docker + AWS CLI on host.
- `deploy/scripts/start_docker.sh`: logs into ECR, pulls latest image, replaces container, maps port `80 -> 8000`.

## Notes

- Large data files are not expected to live directly in Git; use `dvc pull`.
- Collaborative recommendations depend on track presence in listening history; otherwise the app uses content-based mode.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
