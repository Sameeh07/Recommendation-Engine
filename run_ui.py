"""
Entry point for Streamlit UI.

"""

import os
import sys
import subprocess

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    print("Starting SHL Recommendation UI...")

    print()
    
    # Set PYTHONPATH so streamlit subprocess can find modules
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    
    ui_path = os.path.join(PROJECT_ROOT, "ui", "streamlit_app.py")
    
    # Use python -m streamlit to ensure correct environment
    subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path], env=env)
