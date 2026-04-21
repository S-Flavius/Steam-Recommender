# Steam Recommender

An AI-powered game recommendation system for your Steam backlog. Uses TF-IDF and K-Means clustering to analyze your gaming preferences and suggest what to play next.

## Features

- **Steam Library Sync**: Automatically fetches your owned games from Steam
- **Smart Recommendations**: Uses machine learning to match games to your taste profile
- **Difficulty Integration**: Pulls difficulty data from CEDB (completionist.me) for achievement hunters
- **Multiple Categories**: Games are clustered into thematic groups plus curated sections:
  - Top Games (best overall matches)
  - Hard Games (challenging titles)
  - Chill Games (relaxing, casual experiences)
  - Recently Played Unfinished
  - Forgotten Games (unplayed for 1+ year)
- **Explanation System**: See why each game was recommended
- **Rating System**: Rate games 0-10 to improve future recommendations
- **Game Management**: Mark games as finished, ignored, banned, or "up next"

## Setup

### Prerequisites

- Python 3.8+
- Steam API key
- Steam account with public game library

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/steam-recommender.git
   cd steam-recommender
   ```

2. Install dependencies:
   ```bash
   pip install flask pandas numpy scikit-learn requests python-dotenv
   ```

3. Create a `.env` file with your credentials:
   ```
   STEAM_API_KEY=your_steam_api_key_here
   STEAM_ID=your_steam_id_here
   CEDB_USER_ID=your_cedb_user_id_here  # Optional
   ```

### Getting Your Steam API Key

1. Go to https://steamcommunity.com/dev/apikey
2. Log in and register for a key
3. Copy the key to your `.env` file

### Finding Your Steam ID

1. Go to your Steam profile
2. The URL will be `steamcommunity.com/id/YOURID` or `steamcommunity.com/profiles/YOURNUMERICID`
3. If you have a custom URL, use https://steamid.io to find your numeric ID

## Usage

1. Start the server:
   ```bash
   python app.py
   ```

2. Open http://localhost:5000 in your browser

3. Rate games in the carousel using the slider (0-10)

4. Click "Train AI & Re-Cluster" to generate recommendations

5. Use the buttons on each game card:
   - **Finish**: Mark as completed
   - **Up Next**: Boost priority for 3 months
   - **Ban**: Permanently hide
   - **Ignore**: Hide for 30 days

## Project Structure

```
steam-recommender/
├── app.py              # Flask routes and main entry point
├── config.py           # Configuration and constants
├── database.py         # SQLite database management
├── sync.py             # Steam API and external data sync
├── recommender.py      # ML recommendation engine
├── templates/
│   └── index.html      # Main UI template
├── static/
│   └── style.css       # Styles
├── .env                # Your API keys (not committed)
└── {STEAM_ID}.games.db # SQLite database (auto-created)
```

## Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `NUM_CATEGORIES` | 8 | Number of ML-generated clusters |
| `GAMES_PER_CATEGORY` | 10 | Games shown per column |
| `MIN_PLAYTIME` | 60 | Minimum minutes played to be a candidate |
| `CAROUSEL_SIZE` | 15 | Unrated games shown in carousel |

## How It Works

1. **Profile Building**: Your rated games are weighted by rating and Steam review score agreement
2. **TF-IDF Vectorization**: Game tags, names, and developers are converted to feature vectors
3. **Cosine Similarity**: Backlog games are scored against your taste profile
4. **K-Means Clustering**: Games are grouped into thematic categories
5. **Curated Sections**: Additional rule-based columns highlight specific game types

## License

MIT
