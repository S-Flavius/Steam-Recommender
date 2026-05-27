import os

from dotenv import load_dotenv

load_dotenv()

# API Keys and User IDs
STEAM_API_KEY = os.getenv("STEAM_API_KEY")
STEAM_ID = os.getenv("STEAM_ID")
CEDB_USER_ID = os.getenv("CEDB_USER_ID")

# Database
def get_db_file():
    sid = os.getenv("STEAM_ID")
    if sid:
        return f"{sid}.games.db"
    return "games.db"

DB_FILE = get_db_file()

# Recommendation settings
NUM_CATEGORIES = 50
GAMES_PER_CATEGORY = 15
MIN_PLAYTIME = 60
CAROUSEL_SIZE = 1500
IGNORE_DURATION_DAYS = 30
UP_NEXT_DURATION_DAYS = 90

# Tags that indicate a "chill" game
CHILL_TAGS = {
    "casual", "relaxing", "cozy", "cute", "puzzle", "idler", "clicker",
    "incremental", "walking_simulator", "life_sim", "farming_sim",
    "colorful", "wholesome",
}
