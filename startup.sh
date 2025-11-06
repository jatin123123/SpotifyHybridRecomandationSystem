#!/bin/bash

# Azure App Service startup script for Streamlit
echo "Starting Spotify Recommendation System..."

# Create required directories
mkdir -p data
mkdir -p models

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Set environment variables for Streamlit
export STREAMLIT_SERVER_PORT=${PORT:-8000}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start Streamlit app
echo "Starting Streamlit on port $STREAMLIT_SERVER_PORT..."
streamlit run app.py --server.port=$STREAMLIT_SERVER_PORT --server.address=$STREAMLIT_SERVER_ADDRESS --server.headless=true