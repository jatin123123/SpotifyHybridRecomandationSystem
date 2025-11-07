import os
import sys
import subprocess

def main():
    """Startup script for Azure App Service"""
    
    # Set environment variables
    port = os.environ.get('PORT', '8000')
    
    # Azure App Service specific environment variables
    os.environ['STREAMLIT_SERVER_PORT'] = port
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    print(f"Starting Spotify Recommendation System on port {port}...")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Create required directories
    try:
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        print("Created required directories")
    except Exception as e:
        print(f"Error creating directories: {e}")
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("ERROR: app.py not found!")
        sys.exit(1)
    
    # Start Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', str(port),
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()