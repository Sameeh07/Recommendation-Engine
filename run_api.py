"""
Entry point for Flask API.

"""

import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from api.app import app

if __name__ == "__main__":
    print("Starting SHL Recommendation API...")
    print()
    app.run(host="0.0.0.0", port=5000, debug=True)
