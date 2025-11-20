from pathlib import Path
import sys

# Define the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Add the base directory to the system path
sys.path.append(str(BASE_DIR))