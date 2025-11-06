"""Utility helpers for the Spotify Hybrid Music Recommendation System.

This module provides the `HybridRecommender` class that loads the trained
artifacts, performs fuzzy search, and generates hybrid recommendations that
combine content-based similarity with collaborative filtering scores when
available.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy import sparse

LOGGER = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Container for a single recommendation row."""

    track_id: str
    track_name: str
    artist_name: str
    album_name: Optional[str]
    preview_url: Optional[str]
    content_score: float
    cf_score: Optional[float]
    hybrid_score: float
    metadata: Dict[str, Optional[float]]


class ModelNotReadyError(RuntimeError):
    """Raised when the expected model artifacts are missing."""


class HybridRecommender:
    """Loads trained artifacts and serves hybrid recommendations."""

    def __init__(
        self,
        model_dir: Path | str = "models",
        min_content_similarity: float = 0.05,
        content_neighbors: int = 200,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise ModelNotReadyError(
                f"Model directory '{self.model_dir}' does not exist. Run train.py first."
            )

        metadata_path = self.model_dir / "track_metadata.parquet"
        matrix_path = self.model_dir / "content_matrix.npz"
        nn_path = self.model_dir / "nearest_neighbors.pkl"
        config_path = self.model_dir / "content_config.json"

        if not (metadata_path.exists() and matrix_path.exists() and nn_path.exists()):
            raise ModelNotReadyError(
                "Content artifacts are missing. Ensure train.py finished successfully."
            )

        self.metadata = pd.read_parquet(metadata_path)
        if "track_id" not in self.metadata.columns:
            raise ModelNotReadyError("Metadata file must include a 'track_id' column.")
        self.metadata["track_id"] = self.metadata["track_id"].astype(str)

        self.id_to_index: Dict[str, int] = {
            track_id: idx for idx, track_id in enumerate(self.metadata["track_id"].tolist())
        }
        self.index_to_id: Dict[int, str] = {
            idx: track_id for track_id, idx in self.id_to_index.items()
        }

        self.numeric_columns: List[str] = []
        self.text_column: Optional[str] = None
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as cfg_file:
                config = json.load(cfg_file)
            self.numeric_columns = config.get("numeric_columns", [])
            self.text_column = config.get("text_column")

        self.content_matrix = sparse.load_npz(matrix_path)
        self.nn_model = joblib.load(nn_path)

        vectorizer_path = self.model_dir / "content_vectorizer.pkl"
        self.content_vectorizer = None
        if vectorizer_path.exists():
            self.content_vectorizer = joblib.load(vectorizer_path)

        self.min_content_similarity = float(min_content_similarity)
        self.content_neighbors = int(content_neighbors)

        # Optional collaborative filtering artifacts
        self.svd_model = None
        self.item_id_to_index: Dict[str, int] = {}
        self.index_to_item_id: Dict[int, str] = {}
        self.item_embeddings: Optional[np.ndarray] = None

        self._load_collaborative_artifacts()
        self._build_search_corpus()

    def _load_collaborative_artifacts(self) -> None:
        """Load SVD-based collaborative filtering artifacts."""
        cf_model_path = self.model_dir / "svd_model.pkl"
        mapping_path = self.model_dir / "cf_mappings.json"
        embeddings_path = self.model_dir / "item_embeddings.npy"
        
        if not (cf_model_path.exists() and mapping_path.exists() and embeddings_path.exists()):
            LOGGER.info("Collaborative filtering artifacts not found; content-only mode enabled.")
            return

        try:
            self.svd_model = joblib.load(cf_model_path)
            self.item_embeddings = np.load(embeddings_path)
            
            with mapping_path.open("r", encoding="utf-8") as mapping_file:
                mappings = json.load(mapping_file)
            self.item_id_to_index = mappings["item_id_to_index"]
            self.index_to_item_id = {int(k): v for k, v in mappings["index_to_item_id"].items()}
            
            LOGGER.info("Loaded collaborative filtering model with %d items", len(self.item_id_to_index))
        except Exception as e:
            LOGGER.warning("Failed to load collaborative filtering artifacts: %s", e)
            self.svd_model = None
            self.item_embeddings = None
            return

        try:
            self.lightfm_model = joblib.load(cf_model_path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to load LightFM model: %s", exc)
            self.lightfm_model = None
            return

        try:
            self.svd_model = joblib.load(cf_model_path)
            self.item_embeddings = np.load(embeddings_path)
            
            with mapping_path.open("r", encoding="utf-8") as mapping_file:
                mappings = json.load(mapping_file)
            self.item_id_to_index = mappings["item_id_to_index"]
            self.index_to_item_id = {int(k): v for k, v in mappings["index_to_item_id"].items()}
            
            LOGGER.info("Loaded collaborative filtering model with %d items", len(self.item_id_to_index))
        except Exception as e:
            LOGGER.warning("Failed to load collaborative filtering artifacts: %s", e)
            self.svd_model = None
            self.item_embeddings = None

    def _build_search_corpus(self) -> None:
        track_names = self.metadata.get("track_name", pd.Series(dtype=str)).fillna("")
        artist_names = self.metadata.get("artist_name", pd.Series(dtype=str)).fillna("")
        self.search_strings: List[str] = [
            f"{track} - {artist}".strip(" -")
            for track, artist in zip(track_names, artist_names)
        ]

    @property
    def collaborative_available(self) -> bool:
        return self.svd_model is not None and self.item_embeddings is not None

    def search_tracks(self, query: str, limit: int = 10) -> pd.DataFrame:
        if not query:
            return pd.DataFrame(columns=["track_id", "track_name", "artist_name", "score"])

        results = process.extract(
            query,
            self.search_strings,
            scorer=fuzz.WRatio,
            limit=min(limit, len(self.search_strings)),
        )
        payload: List[Dict[str, object]] = []
        for match, score, index in results:
            row = self.metadata.iloc[int(index)]
            payload.append(
                {
                    "track_id": row["track_id"],
                    "track_name": row.get("track_name", ""),
                    "artist_name": row.get("artist_name", ""),
                    "score": float(score) / 100.0,
                }
            )
        return pd.DataFrame(payload)

    def recommend_from_track(
        self,
        track_id: str,
        top_n: int = 20,
        content_weight: float = 0.7,
        filters: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[RecommendationResult]:
        if track_id not in self.id_to_index:
            raise ValueError(f"Track id '{track_id}' not found in metadata.")
        content_payload = self._compute_content_neighbors(track_id, top_n * 5)
        cf_payload = self._compute_cf_neighbors(track_id)
        merged = self._merge_scores(content_payload, cf_payload, content_weight)
        merged = [entry for entry in merged if entry.track_id != track_id]
        if filters:
            merged = [entry for entry in merged if self._passes_filters(entry, filters)]
        return merged[:top_n]

    def _compute_content_neighbors(
        self,
        track_id: str,
        candidate_pool: int,
    ) -> Dict[str, float]:
        idx = self.id_to_index[track_id]
        n_neighbors = min(max(candidate_pool, 1), self.content_matrix.shape[0])
        distances, indices = self.nn_model.kneighbors(
            self.content_matrix[idx],
            n_neighbors=n_neighbors,
        )
        scores: Dict[str, float] = {}
        for distance, index in zip(distances.flatten(), indices.flatten()):
            similarity = 1.0 - float(distance)
            if similarity < self.min_content_similarity:
                continue
            candidate_track_id = self.index_to_id.get(int(index))
            if candidate_track_id is None:
                continue
            scores[candidate_track_id] = max(scores.get(candidate_track_id, 0.0), similarity)
        return scores

    def _compute_cf_neighbors(self, track_id: str) -> Dict[str, float]:
        """Compute collaborative filtering scores using SVD embeddings."""
        if not self.collaborative_available or self.item_embeddings is None:
            return {}
        
        item_index = self.item_id_to_index.get(track_id)
        if item_index is None:
            return {}
        
        # Get the embedding for the target item
        item_vec = self.item_embeddings[item_index]
        
        # Compute cosine similarities with all items
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized_embeddings = self.item_embeddings / norms
        
        query_norm = np.linalg.norm(item_vec)
        if query_norm == 0:
            return {}
        
        similarities = normalized_embeddings @ (item_vec / query_norm)
        
        scores: Dict[str, float] = {}
        for idx, score in enumerate(similarities):
            if idx == item_index:  # Skip self
                continue
            candidate_track_id = self.index_to_item_id.get(idx)
            if candidate_track_id is None:
                continue
            scores[candidate_track_id] = float(score)
        
        return scores

    def _merge_scores(
        self,
        content_scores: Dict[str, float],
        cf_scores: Dict[str, float],
        content_weight: float,
    ) -> List[RecommendationResult]:
        ids = set(content_scores) | set(cf_scores)
        if not ids:
            return []
        content_values = np.array([content_scores.get(track_id, 0.0) for track_id in ids])
        cf_values = np.array([cf_scores.get(track_id, 0.0) for track_id in ids])
        content_norm = self._normalise(content_values)
        cf_norm = self._normalise(cf_values)
        blended = content_weight * content_norm + (1.0 - content_weight) * cf_norm
        results: List[RecommendationResult] = []
        for idx, track_id in enumerate(ids):
            row = self.metadata.iloc[self.id_to_index[track_id]]
            metadata = {
                "tempo": self._safe_float(row.get("tempo")),
                "valence": self._safe_float(row.get("valence")),
                "energy": self._safe_float(row.get("energy")),
                "instrumentalness": self._safe_float(row.get("instrumentalness")),
            }
            results.append(
                RecommendationResult(
                    track_id=track_id,
                    track_name=row.get("track_name", "Unknown"),
                    artist_name=row.get("artist_name", "Unknown Artist"),
                    album_name=self._safe_string(row.get("album_name")),
                    preview_url=self._safe_string(row.get("preview_url")),
                    content_score=float(content_norm[idx]),
                    cf_score=float(cf_norm[idx]) if track_id in cf_scores else None,
                    hybrid_score=float(blended[idx]),
                    metadata=metadata,
                )
            )
        results.sort(key=lambda item: item.hybrid_score, reverse=True)
        return results

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum - minimum < 1e-9:
            return np.zeros_like(values)
        return (values - minimum) / (maximum - minimum)

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_string(value: object) -> Optional[str]:
        """Convert a value to a string, handling NaN and None."""
        try:
            if value is None:
                return None
            if isinstance(value, float) and np.isnan(value):
                return None
            str_value = str(value).strip()
            return str_value if str_value else None
        except (TypeError, ValueError):
            return None

    def _passes_filters(
        self,
        result: RecommendationResult,
        filters: Dict[str, Tuple[float, float]],
    ) -> bool:
        for feature in ("tempo", "valence", "energy", "instrumentalness"):
            if feature not in filters:
                continue
            bounds = filters[feature]
            value = result.metadata.get(feature)
            if value is None:
                continue
            if value < bounds[0] or value > bounds[1]:
                return False
        return True


def playlist_to_csv(rows: List[RecommendationResult]) -> bytes:
    """Serialise playlist rows to CSV bytes."""
    if not rows:
        return b""
    frame = pd.DataFrame(
        [
            {
                "track_id": item.track_id,
                "track_name": item.track_name,
                "artist_name": item.artist_name,
                "album_name": item.album_name,
                "preview_url": item.preview_url,
                "hybrid_score": item.hybrid_score,
                "content_score": item.content_score,
                "cf_score": item.cf_score,
            }
            for item in rows
        ]
    )
    return frame.to_csv(index=False).encode("utf-8")


def playlist_to_m3u(rows: List[RecommendationResult]) -> str:
    """Create an M3U playlist string using preview URLs when available."""
    header = "#EXTM3U"
    lines: List[str] = [header]
    for item in rows:
        title = f"{item.artist_name} - {item.track_name}".strip()
        url = item.preview_url or ""
        lines.append(f"#EXTINF:-1,{title}")
        lines.append(url)
    return "\n".join(lines)
