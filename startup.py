import os
import sys
import subprocess

def main():
    """Startup script for Azure App Service"""
    
    # Set environment variables
    port = os.environ.get('PORT', '8000')
    os.environ['STREAMLIT_SERVER_PORT'] = port
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    print(f"Starting Spotify Recommendation System on port {port}...")
    
    # Create required directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Start Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true'
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == '__main__':
    main()