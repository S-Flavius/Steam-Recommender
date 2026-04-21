import os

from dotenv import load_dotenv

load_dotenv()

# API Keys and User IDs
STEAM_API_KEY = os.getenv("STEAM_API_KEY")
STEAM_ID = os.getenv("STEAM_ID")
CEDB_USER_ID = os.getenv("CEDB_USER_ID")

# Database
DB_FILE = f"{STEAM_ID}.games.db"

# Recommendation settings
NUM_CATEGORIES = 8
GAMES_PER_CATEGORY = 10
MIN_PLAYTIME = 60
CAROUSEL_SIZE = 15

# Tags that indicate a "chill" game
CHILL_TAGS = {
    "casual", "relaxing", "cozy", "cute", "puzzle", "idler", "clicker",
    "incremental", "walking_simulator", "life_sim", "farming_sim",
    "colorful", "wholesome",
}
