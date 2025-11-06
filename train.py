"""Training script for the Spotify Hybrid Music Recommendation System.

This script prepares content-based features from Spotify audio descriptors and
Last.fm tags, fits a nearest-neighbour model, and optionally trains a LightFM
collaborative filtering model when user interaction data is available.

Example usage
-------------

python train.py \
    --metadata data/tracks_features.csv \
    --tags data/tags.csv \
    --interactions data/lastfm_user_scrobbles.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger("train")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Spotify audio features frequently available in the dataset
NUMERIC_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms",
    "key",
    "mode",
    "time_signature",
    "popularity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid recommendation models")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/Music Info.csv"),
        help="CSV file with Spotify track metadata and audio features.",
    )
    parser.add_argument(
        "--tags",
        type=Path,
        default=None,
        help="Optional CSV with Last.fm tags (columns: track_id, tag). Tags are included in metadata file.",
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=Path("data/User Listening History.csv"),
        help="Optional CSV with Last.fm user-track interactions for collaborative filtering.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory where trained artifacts will be stored.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for LightFM training (if enabled).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Thread count for LightFM training (if enabled).",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file '{path}' not found.")
    LOGGER.info("Loading metadata from %s", path)
    metadata = pd.read_csv(path)
    if "track_id" not in metadata.columns:
        raise ValueError("Metadata file must contain a 'track_id' column.")
    
    # Standardize column names to match the rest of the system
    column_mapping = {
        "name": "track_name",
        "artist": "artist_name", 
        "spotify_preview_url": "preview_url",
        "tags": "tags_text"  # Use tags directly from the metadata file
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in metadata.columns:
            metadata[new_col] = metadata[old_col]
    
    # Ensure we have the required columns
    if "track_name" not in metadata.columns:
        metadata["track_name"] = "Unknown Track"
    if "artist_name" not in metadata.columns:
        metadata["artist_name"] = "Unknown Artist"
    if "preview_url" not in metadata.columns:
        metadata["preview_url"] = None
    if "tags_text" not in metadata.columns:
        metadata["tags_text"] = ""
    
    metadata["track_id"] = metadata["track_id"].astype(str)
    metadata = metadata.drop_duplicates(subset="track_id")
    return metadata


def load_and_merge_tags(metadata: pd.DataFrame, tags_path: Optional[Path]) -> pd.DataFrame:
    if tags_path is None:
        base_tags = metadata["tags"] if "tags" in metadata.columns else pd.Series("", index=metadata.index)
        metadata["tags_text"] = base_tags.fillna("")
        return metadata
    if not tags_path.exists():
        LOGGER.warning("Tag file %s not found; continuing without tags.", tags_path)
        base_tags = metadata["tags"] if "tags" in metadata.columns else pd.Series("", index=metadata.index)
        metadata["tags_text"] = base_tags.fillna("")
        return metadata
    LOGGER.info("Loading tags from %s", tags_path)
    tags = pd.read_csv(tags_path)
    if {"track_id", "tag"} - set(tags.columns):
        raise ValueError("Tags file must contain 'track_id' and 'tag' columns.")
    tags["track_id"] = tags["track_id"].astype(str)
    tags["tag"] = tags["tag"].astype(str)
    tag_map = tags.groupby("track_id")["tag"].agg(lambda values: " ".join(sorted(set(values))))
    metadata = metadata.merge(tag_map, how="left", left_on="track_id", right_index=True)
    metadata.rename(columns={"tag": "tags_text"}, inplace=True)
    # Fill any NaN values with empty string
    metadata["tags_text"] = metadata["tags_text"].fillna("")
    return metadata


def build_content_matrix(metadata: pd.DataFrame) -> Dict[str, object]:
    numeric_columns = [col for col in NUMERIC_FEATURES if col in metadata.columns]
    if not numeric_columns:
        raise ValueError(
            "No numeric audio feature columns were found. Ensure the metadata file "
            "contains Spotify audio features such as 'danceability' or 'tempo'."
        )
    LOGGER.info("Available columns: %s", metadata.columns.tolist())
    numeric_df = metadata[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.fillna(numeric_df.median())
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_df.values)
    numeric_sparse = sparse.csr_matrix(numeric_scaled)

    text_column = "tags_text"
    if text_column not in metadata.columns:
        raise ValueError(f"Column '{text_column}' not found in metadata. Available: {metadata.columns.tolist()}")
    text_series = metadata[text_column].fillna("")
    LOGGER.info("Text series shape: %d, empty count: %d", len(text_series), (text_series == "").sum())
    LOGGER.info("Sample text values: %s", text_series.head().tolist())
    tfidf = TfidfVectorizer(
        max_features=5000,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,  # Allow words that appear in at least 1 document
    )
    LOGGER.info("Fitting TF-IDF vectoriser on %d tracks", len(text_series))
    text_matrix = tfidf.fit_transform(text_series.tolist())

    content_matrix = sparse.hstack([numeric_sparse, text_matrix], format="csr")
    content_matrix = sparse.csr_matrix(content_matrix)
    nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    LOGGER.info("Fitting NearestNeighbors on content matrix with shape %s", content_matrix.shape)
    nn_model.fit(content_matrix)

    return {
        "numeric_columns": numeric_columns,
        "text_column": text_column,
        "scaler": scaler,
        "tfidf": tfidf,
        "content_matrix": content_matrix,
        "nn_model": nn_model,
    }


def save_content_artifacts(artifacts: Dict[str, object], model_dir: Path, metadata: pd.DataFrame) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    content_matrix: sparse.csr_matrix = artifacts["content_matrix"]  # type: ignore[assignment]
    sparse.save_npz(model_dir / "content_matrix.npz", content_matrix)
    joblib.dump(artifacts["nn_model"], model_dir / "nearest_neighbors.pkl")
    joblib.dump(
        {
            "numeric_columns": artifacts["numeric_columns"],
            "text_column": artifacts["text_column"],
            "scaler": artifacts["scaler"],
            "tfidf": artifacts["tfidf"],
        },
        model_dir / "content_vectorizer.pkl",
    )
    with (model_dir / "content_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "numeric_columns": artifacts["numeric_columns"],
                "text_column": artifacts["text_column"],
            },
            config_file,
            indent=2,
        )
    metadata.to_parquet(model_dir / "track_metadata.parquet", index=False)
    LOGGER.info("Saved content-based artifacts to %s", model_dir)


def train_collaborative_model(
    interactions_path: Optional[Path],
    metadata: pd.DataFrame,
    model_dir: Path,
    epochs: int,
    num_threads: int,
) -> None:
    """Train a collaborative filtering model using SVD matrix factorization."""
    if interactions_path is None:
        LOGGER.info("No interactions file provided; skipping collaborative training.")
        return
    if not interactions_path.exists():
        LOGGER.warning("Interactions file %s not found; skipping collaborative model.", interactions_path)
        return

    LOGGER.info("Loading interactions from %s", interactions_path)
    interactions = pd.read_csv(interactions_path)
    required_columns = {"user_id", "track_id"}
    if missing := required_columns - set(interactions.columns):
        raise ValueError(f"Interactions file missing columns: {missing}")
    
    interactions["track_id"] = interactions["track_id"].astype(str)
    interactions = interactions[interactions["track_id"].isin(metadata["track_id"])]
    if interactions.empty:
        LOGGER.warning("No overlapping tracks between interactions and metadata; skipping CF model.")
        return

    interactions["user_id"] = interactions["user_id"].astype(str)
    
    # Filter users with at least 5 interactions for better recommendations
    user_counts = interactions["user_id"].value_counts()
    active_users = user_counts[user_counts >= 5].index
    interactions = interactions[interactions["user_id"].isin(active_users)]
    
    # Filter tracks with at least 3 interactions
    track_counts = interactions["track_id"].value_counts()
    popular_tracks = track_counts[track_counts >= 3].index
    interactions = interactions[interactions["track_id"].isin(popular_tracks)]
    
    if interactions.empty:
        LOGGER.warning("No interactions remaining after filtering; skipping CF model.")
        return

    user_ids = interactions["user_id"].unique().tolist()
    item_ids = interactions["track_id"].unique().tolist()
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {track_id: idx for idx, track_id in enumerate(item_ids)}
    index_to_item_id = {idx: track_id for track_id, idx in item_id_to_index.items()}

    # Handle weight column (playcount, plays, weight)
    weight_column = None
    for candidate in ("playcount", "plays", "weight"):
        if candidate in interactions.columns:
            weight_column = candidate
            break
    
    if weight_column:
        weights = interactions[weight_column].fillna(1.0).astype(float).values
        weights = np.log1p(np.asarray(weights, dtype=float))
    else:
        weights = np.ones(len(interactions), dtype=float)

    # Create user-item matrix
    row = interactions["user_id"].map(user_id_to_index).values
    col = interactions["track_id"].map(item_id_to_index).values
    matrix = sparse.coo_matrix((weights, (row, col)), shape=(len(user_ids), len(item_ids)))
    
    LOGGER.info(
        "Training SVD on %d users x %d tracks with %d interactions",
        len(user_ids),
        len(item_ids),
        matrix.nnz,
    )
    
    # Use TruncatedSVD for matrix factorization
    n_components = min(50, min(len(user_ids), len(item_ids)) - 1)
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    matrix_csr = matrix.tocsr()
    svd_model.fit(matrix_csr)  # type: ignore
    
    # Generate item embeddings from the SVD
    item_embeddings = svd_model.components_.T  # Shape: (n_items, n_components)
    
    # Save collaborative filtering artifacts
    joblib.dump(svd_model, model_dir / "svd_model.pkl")
    np.save(model_dir / "item_embeddings.npy", item_embeddings)
    
    with (model_dir / "cf_mappings.json").open("w", encoding="utf-8") as mapping_file:
        json.dump(
            {
                "user_id_to_index": user_id_to_index,
                "item_id_to_index": item_id_to_index,
                "index_to_item_id": index_to_item_id,
            },
            mapping_file,
            indent=2,
        )
    
    LOGGER.info("Saved collaborative filtering model (SVD) to %s", model_dir)


if __name__ == "__main__":
    args = parse_args()
    metadata_df = load_metadata(args.metadata)
    metadata_df = load_and_merge_tags(metadata_df, args.tags)
    content_artifacts = build_content_matrix(metadata_df)
    save_content_artifacts(content_artifacts, args.model_dir, metadata_df)
    train_collaborative_model(args.interactions, metadata_df, args.model_dir, args.epochs, args.num_threads)
    LOGGER.info("Training pipeline completed.")
