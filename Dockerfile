# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy all required data files at once
COPY ./data/collab_filtered_data.csv \
     ./data/interaction_matrix.npz \
     ./data/track_ids.npy \
     ./data/cleaned_data.csv \
     ./data/transformed_data.npz \
     ./data/transformed_hybrid_data.npz \
     ./data/

# Copy all required Python scripts at once
COPY app.py \
     collaborative_filtering.py \
     content_based_filtering.py \
     hybrid_recommendations.py \
     data_cleaning.py \
     transform_filtered_data.py \
     ./
     
# expose the port on the container
EXPOSE 8000

# run the streamlit app
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]    