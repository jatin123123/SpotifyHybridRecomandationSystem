"""Streamlit application for the Spotify Hybrid Music Recommendation System."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    HybridRecommender,
    ModelNotReadyError,
    RecommendationResult,
    playlist_to_csv,
    playlist_to_m3u,
)

st.set_page_config(
    page_title="🎵 Spotify Hybrid Recommender", 
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_audio_features_chart(rows: list[RecommendationResult]) -> go.Figure:
    """Create a radar chart showing audio features of recommended tracks."""
    if not rows:
        return go.Figure()
    
    # Extract audio features
    features = ["energy", "valence", "tempo", "instrumentalness"]
    avg_features = {}
    
    for feature in features:
        values = [item.metadata.get(feature) for item in rows if item.metadata.get(feature) is not None]
        if values:
            # Convert to numpy array for proper calculation
            values_array = np.array([v for v in values if v is not None])
            if feature == "tempo":
                # Normalize tempo to 0-1 scale for visualization
                avg_features[feature] = np.mean(values_array) / 200.0
            else:
                avg_features[feature] = np.mean(values_array)
        else:
            avg_features[feature] = 0
    
    # Create radar chart
    categories = ["Energy", "Mood", "Tempo", "Instrumental"]
    values = [avg_features["energy"], avg_features["valence"], 
              avg_features["tempo"], avg_features["instrumentalness"]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Playlist Profile',
        line=dict(color='#1DB954')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig


def create_score_distribution_chart(rows: list[RecommendationResult]) -> go.Figure:
    """Create a chart showing the distribution of recommendation scores."""
    if not rows:
        return go.Figure()
    
    scores = [item.hybrid_score for item in rows]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Track {i+1}" for i in range(len(scores))],
        y=scores,
        marker_color='#1DB954',
        name='Similarity Score'
    ))
    
    fig.update_layout(
        title="Recommendation Confidence",
        xaxis_title="",
        yaxis_title="Score",
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    
    return fig


@st.cache_resource(show_spinner=True)
def load_recommender(model_path: Path = Path("models")) -> HybridRecommender:
    """Load and cache the recommender instance across Streamlit sessions."""
    return HybridRecommender(model_dir=model_path)


def build_filters(recommender: HybridRecommender) -> Dict[str, Tuple[float, float]]:
    filters: Dict[str, Tuple[float, float]] = {}
    
    with st.sidebar:
        st.header("🎛️ Controls")
        st.write("All audio features are included in recommendations automatically.")
    
    return filters


def render_playlist(rows: list[RecommendationResult]) -> None:
    if not rows:
        st.warning("No tracks matched the current filters. Try expanding the ranges.")
        return
    
    # Add analytics section with charts
    st.subheader("� Playlist Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio Features Profile")
        audio_chart = create_audio_features_chart(rows)
        st.plotly_chart(audio_chart, use_container_width=True)
    
    with col2:
        st.subheader("Recommendation Scores")
        score_chart = create_score_distribution_chart(rows)
        st.plotly_chart(score_chart, use_container_width=True)
    
    st.divider()
    st.subheader("🎵 Recommended Tracks")
    
    # Check if any preview URLs are available
    has_previews = any(item.preview_url for item in rows)
    if has_previews:
        st.success("🎵 **Audio previews available!** Click play buttons below to listen to 30-second clips.")
    else:
        st.info("ℹ️ Audio previews are not available for these tracks.")
    
    for rank, item in enumerate(rows, start=1):
        with st.container():
            # Display track name and artist prominently
            st.markdown(f"### {rank}. {item.track_name}")
            st.markdown(f"**Artist:** {item.artist_name}")
            if item.album_name:
                st.markdown(f"**Album:** {item.album_name}")
            
            # Audio preview (if available) - place prominently
            if item.preview_url and isinstance(item.preview_url, str) and item.preview_url.startswith("http"):
                st.audio(item.preview_url, format="audio/mp3")
            
            # Show audio features
            tags = []
            if item.metadata.get("tempo") is not None:
                tags.append(f"🎵 Tempo: {item.metadata['tempo']:.0f} BPM")
            if item.metadata.get("valence") is not None:
                tags.append(f"😊 Valence: {item.metadata['valence']:.2f}")
            if item.metadata.get("energy") is not None:
                tags.append(f"⚡ Energy: {item.metadata['energy']:.2f}")
            if item.metadata.get("instrumentalness") is not None:
                tags.append(f"🎸 Instrumentalness: {item.metadata['instrumentalness']:.2f}")
            
            if tags:
                st.caption(" | ".join(tags))
            
            # Show recommendation scores with better explanation
            col1, col2 = st.columns([2, 1])
            with col1:
                score_bits = [f"**Final Score: {item.hybrid_score:.3f}**"]
                score_bits.append(f"Content: {item.content_score:.3f}")
                if item.cf_score is not None:
                    score_bits.append(f"Collaborative: {item.cf_score:.3f}")
                else:
                    score_bits.append("Collaborative: N/A")
                st.caption(" | ".join(score_bits))
            
            with col2:
                if item.cf_score is not None:
                    st.caption("🔄 Hybrid recommendation")
                else:
                    st.caption("🎯 Content-based only")
            
            st.divider()


def main() -> None:
    st.title("🎵 Spotify Hybrid Music Recommendation System")
    st.markdown(
        """
        **Discover your next favorite songs** using our advanced hybrid recommendation engine that combines:
        
        - 🎯 **Content-Based Filtering**: Analyzes audio features like energy, valence, tempo, and instrumentalness
        - 👥 **Collaborative Filtering**: Learns from user listening patterns and preferences 
        - 🔄 **Hybrid Approach**: Intelligently blends both methods for superior recommendations
        """
    )

    try:
        recommender = load_recommender()
        demo_mode = False
    except ModelNotReadyError as exc:
        st.warning("⚠️ **Demo Mode**: Pre-trained models not available")
        st.info("""
        **To use the full system:**
        1. 📁 Upload your music dataset (CSV files)
        2. 🚀 Run `python train.py` to train the models
        3. 🎵 Start making recommendations!
        
        **For now, explore the demo interface below:**
        """)
        
        # Create a demo interface
        demo_mode = True
        show_demo_interface()
        return

    # Sidebar controls (only if models are loaded)
    with st.sidebar:
        st.header("⚙️ Recommendation Settings")
        
        # Show system capabilities
        if recommender.collaborative_available:
            st.success("✅ Hybrid mode: Content + Collaborative filtering")
            content_weight = st.slider(
                "Content vs Collaborative Weight", 
                0.0, 1.0, value=0.7, step=0.05,
                key="content_weight_slider",
                help="0.0 = Pure collaborative filtering, 1.0 = Pure content-based"
            )
        else:
            st.warning("⚠️ Content-only mode (no user interaction data)")
            content_weight = 1.0
        
        playlist_size = st.slider("Playlist size", 5, 50, value=20, step=1, key="playlist_size_slider")
    
    filters = build_filters(recommender)

    # Show explanation of the approach
    with st.expander("🧠 How Our Hybrid System Works"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Content-Based Filtering")
            st.write("""
            - Analyzes audio features (energy, valence, tempo, etc.)
            - Uses TF-IDF for text features like genres and tags
            - Finds songs with similar musical characteristics
            - Works even for new songs with no user data
            """)
        
        with col2:
            st.subheader("👥 Collaborative Filtering")
            if recommender.collaborative_available:
                st.write("""
                - Learns from 9.7M+ user listening histories
                - Uses matrix factorization (SVD) to find patterns
                - Discovers songs liked by users with similar taste
                - Provides serendipitous recommendations
                """)
            else:
                st.write("""
                - Would analyze user listening patterns
                - Currently disabled (no interaction data)
                - Enable by providing user-track interaction data
                - Requires user_id, track_id columns
                """)

    # Search section
    st.subheader("🔍 Find Your Seed Track")
    search_input = st.text_input(
        "Search for a track or artist",
        placeholder="Enter track name or artist...",
        help="Type a song title or artist name to find your starting point"
    )

    if search_input:
        results_df = recommender.search_tracks(search_input, limit=10)
        if not results_df.empty:
            # Create a clean selection interface
            track_options = []
            for _, result in results_df.iterrows():
                match_score = f"({result['score']:.0%} match)"
                track_options.append(f"{result['track_name']} - {result['artist_name']} {match_score}")
            
            selected_track = st.selectbox("Select a track:", track_options)
            
            if selected_track and st.button("🎯 Generate Recommendations", type="primary"):
                with st.spinner("Generating your personalized playlist..."):
                    selected_idx = track_options.index(selected_track)
                    selected_result = results_df.iloc[selected_idx]
                    
                    recommendations = recommender.recommend_from_track(
                        selected_result['track_id'],
                        top_n=playlist_size,
                        content_weight=content_weight,
                        filters=filters,
                    )
                    
                    if recommendations:
                        st.success(f"✨ Generated {len(recommendations)} recommendations!")
                        
                        # Show recommendation method explanation
                        if recommender.collaborative_available:
                            content_pct = int(content_weight * 100)
                            collab_pct = int((1 - content_weight) * 100)
                            st.info(f"🔄 **Hybrid Method**: {content_pct}% content-based + {collab_pct}% collaborative filtering")
                        else:
                            st.info("🎯 **Content-Based Method**: Recommendations based on audio features and metadata")
                        
                        render_playlist(recommendations)
                        
                        # Export options
                        st.subheader("💾 Export Playlist")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv_data = playlist_to_csv(recommendations)
                            st.download_button(
                                "📄 Download as CSV",
                                csv_data,
                                file_name="spotify_recommendations.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            m3u_data = playlist_to_m3u(recommendations)
                            st.download_button(
                                "🎵 Download as M3U",
                                m3u_data,
                                file_name="spotify_recommendations.m3u",
                                mime="audio/x-mpegurl"
                            )
                    else:
                        st.warning("No recommendations found with current filters. Try adjusting the audio feature ranges.")
        else:
            st.info("No tracks found. Try a different search term.")


def show_demo_interface():
    """Show a demo interface when models are not available"""
    
    st.subheader("🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Sample Data Structure")
        st.code("""
# Music Info.csv (required)
track_id,track_name,artist_name,preview_url,energy,valence,tempo
spotify_123,Bohemian Rhapsody,Queen,https://preview.url,0.8,0.5,144
spotify_456,Hotel California,Eagles,https://preview.url,0.7,0.6,141

# User Listening History.csv (optional)
user_id,track_id,playcount
user_001,spotify_123,45
user_002,spotify_456,23
        """, language="csv")
        
    with col2:
        st.markdown("### 🛠️ Setup Instructions")
        st.markdown("""
        1. **Prepare Data Files:**
           - Create `data/Music Info.csv` with your music metadata
           - Optionally add `data/User Listening History.csv`
        
        2. **Train Models:**
           ```bash
           python train.py
           ```
        
        3. **Launch App:**
           ```bash
           streamlit run app.py
           ```
        """)
    
    st.subheader("🎯 System Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        **🎵 Content Filtering**
        - Audio feature analysis
        - TF-IDF text processing
        - Nearest neighbors search
        - Genre & tag matching
        """)
    
    with feature_col2:
        st.markdown("""
        **👥 Collaborative Filtering**
        - SVD matrix factorization
        - User behavior patterns
        - Similarity clustering
        - Serendipitous discovery
        """)
    
    with feature_col3:
        st.markdown("""
        **🔄 Hybrid Approach**
        - Weighted score blending
        - Adjustable preferences
        - Cold start handling
        - Export capabilities
        """)
    
    st.subheader("📈 Technical Architecture")
    st.image("https://via.placeholder.com/800x300/1DB954/FFFFFF?text=Hybrid+Recommendation+System+Architecture", 
             caption="System combines content-based filtering with collaborative filtering for optimal recommendations")
    
    st.subheader("🌟 Sample Recommendations")
    st.markdown("""
    Once your system is trained, you'll see recommendations like:
    
    | Rank | Song | Artist | Similarity Score | Method |
    |------|------|--------|-----------------|---------|
    | 1 | Paradise City | Guns N' Roses | 0.892 | 🔄 Hybrid |
    | 2 | Sweet Child O' Mine | Guns N' Roses | 0.845 | 🎯 Content |
    | 3 | November Rain | Guns N' Roses | 0.821 | 👥 Collaborative |
    """)
    
    # Add GitHub link
    st.markdown("---")
    st.markdown("### 🔗 Links")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("[📚 GitHub Repository](https://github.com/jatin123123/SpotifyHybridRecomandationSystem)")
    
    with col2:
        st.markdown("[📖 Documentation](https://github.com/jatin123123/SpotifyHybridRecomandationSystem#readme)")
    
    with col3:
        st.markdown("[🐛 Report Issues](https://github.com/jatin123123/SpotifyHybridRecomandationSystem/issues)")


if __name__ == "__main__":
    main()
