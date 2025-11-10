"""
Deployment entry point for Fake Logo Detection Suite.
This file exists to match the deployment configuration in .replit
and simply runs the main Streamlit application.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main app
from src.app_streamlit import main

if __name__ == "__main__":
    main()
