# 🎵 Spotify Hybrid Music Recommendation System

A sophisticated music recommendation engine that combines **content-based filtering** and **collaborative filtering** to deliver personalized music suggestions. Built with Python, Streamlit, and advanced machine learning techniques.

## ✨ Features

- 🎯 **Hybrid Recommendation Engine**: Combines content-based and collaborative filtering
- 🎵 **Audio Preview Integration**: Real Spotify 30-second previews 
- 📊 **Advanced Analytics**: Visual insights into recommendation patterns
- 🔍 **Smart Search**: Fuzzy search with match scoring
- 💾 **Playlist Export**: CSV and M3U format support
- 🎛️ **Interactive UI**: Clean, modern Streamlit interface
- ⚡ **Fast Performance**: Optimized with sparse matrices and caching

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/jatin123123/SpotifyHybridRecomandationSystem.git
cd SpotifyHybridRecomandationSystem
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your data** (see Data Setup section below)

4. **Train the models**
```bash
python train.py
```

5. **Run the application**
```bash
streamlit run app.py
```

## 📊 Data Setup

### Required Data Files

Place these CSV files in the `data/` directory:

#### 1. `Music Info.csv` - Track Metadata
Required columns:
- `track_id` - Unique identifier
- `track_name` - Song title
- `artist_name` - Artist name
- `preview_url` - Spotify preview URL (optional)
- Audio features: `energy`, `valence`, `tempo`, `instrumentalness`, etc.
- `tags_text` - Genre/tag information

#### 2. `User Listening History.csv` - User Interactions (Optional)
Required columns:
- `user_id` - User identifier
- `track_id` - Track identifier (must match Music Info)
- `playcount` or `plays` - Listen count (optional)

### Data Sources

- **Spotify Web API**: For track metadata and audio features
- **Last.fm Dataset**: For user listening histories
- **Million Song Dataset**: Alternative source for music data

## 🔧 How It Works

### Content-Based Filtering
- Analyzes audio features (energy, valence, tempo, etc.)
- Uses TF-IDF vectorization for text features (genres, tags)
- Employs k-nearest neighbors for similarity matching

### Collaborative Filtering  
- Matrix factorization using Truncated SVD
- Learns user preference patterns from listening history
- Generates item embeddings for similarity computation

### Hybrid Approach
- Intelligently combines both filtering methods
- Weighted scoring system (adjustable via UI)
- Handles cold-start problems for new tracks

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │────│  Training Layer  │────│   Serving Layer │
│                 │    │                  │    │                 │
│ • Music Info    │    │ • Content Matrix │    │ • Streamlit UI  │
│ • User History  │    │ • SVD Model      │    │ • Search Engine │
│ • Audio Features│    │ • Embeddings     │    │ • Recommender   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
SpotifyHybridRecomandationSystem/
├── app.py              # Streamlit web application
├── train.py            # Model training pipeline  
├── utils.py            # Recommendation engine core
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── data/              # Data files (not included)
│   ├── Music Info.csv
│   └── User Listening History.csv
└── models/            # Trained models (generated)
    ├── content_matrix.npz
    ├── svd_model.pkl
    └── ...
```

## 🎛️ Configuration

### Training Parameters

Modify `train.py` arguments:
- `--epochs`: Training epochs for collaborative filtering
- `--num-threads`: Parallel processing threads
- `--model-dir`: Output directory for trained models

### Recommendation Tuning

Adjust in the UI:
- **Content vs Collaborative Weight**: Balance between recommendation types
- **Playlist Size**: Number of recommendations to generate
- **Audio Filters**: Fine-tune by energy, mood, tempo, etc.

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web app deployment
- **Docker**: Containerized deployment
- **AWS/GCP**: Cloud hosting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Spotify Web API** for audio features and preview URLs
- **Last.fm** for user interaction datasets
- **Streamlit** for the amazing web framework
- **scikit-learn** for machine learning algorithms

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/jatin123123/SpotifyHybridRecomandationSystem/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/jatin123123/SpotifyHybridRecomandationSystem/discussions)

---

⭐ **Star this repo** if you find it helpful!

Built with ❤️ for music lovers and data scientists.
