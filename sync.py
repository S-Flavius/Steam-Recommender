import math
import time

import requests
from tqdm import tqdm

from config import STEAM_API_KEY, STEAM_ID, CEDB_USER_ID
from database import get_db


def sync_steam_library():
    """Fetch owned games from Steam API and update the database."""
    url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": STEAM_API_KEY,
        "steamid": STEAM_ID,
        "format": "json",
        "include_appinfo": True,
    }

    res = requests.get(url, params=params)
    if res.status_code == 200:
        games = res.json().get("response", {}).get("games", [])
        conn = get_db()
        for g in games:
            conn.execute('''
                         INSERT INTO games (appid, name, playtime, last_played)
                         VALUES (?, ?, ?, ?)
                         ON CONFLICT(appid) DO UPDATE SET playtime    = excluded.playtime,
                                                          last_played = excluded.last_played
                         ''', (
                g["appid"],
                g.get("name", "Unknown"),
                g.get("playtime_forever", 0),
                g.get("rtime_last_played", 0),
            ))
        conn.commit()
        conn.close()


def sync_cedb_difficulties():
    """Fetch difficulty tiers from CEDB (completionist.me) and update games."""
    if not CEDB_USER_ID:
        return

    conn = get_db()
    c = conn.cursor()

    # Check if we've synced recently (within 7 days)
    c.execute("SELECT value FROM metadata WHERE key = 'last_cedb_sync'")
    row = c.fetchone()
    now = time.time()
    if row and now - float(row['value']) < 604800:  # 7 days in seconds
        conn.close()
        return

    res = requests.get(f"https://cedb.me/api/user/{CEDB_USER_ID}/games")
    if res.status_code == 200:
        updates = []
        for item in res.json():
            game = item.get('game', {})
            if str(game.get('platform')).lower() == 'steam':
                updates.append((
                    f"T{game.get('tier')} (Challenge)",
                    int(game.get('platformId')),
                ))
        c.executemany("UPDATE games SET difficulty = ? WHERE appid = ?", updates)
        c.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_cedb_sync', ?)",
            (str(now),)
        )
        conn.commit()
    conn.close()


def get_game_data(appid, force_refresh=False):
    """
    Fetch game tags and Steam score from SteamSpy.
    Returns cached data if available, otherwise fetches and stores it.

    Args:
        appid: The Steam app ID
        force_refresh: If True, always fetch fresh data from SteamSpy
    """
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM games WHERE appid = ?", (appid,))
    row = c.fetchone()

    if not force_refresh and row and row['tags']:
        conn.close()
        return dict(row)

    # Fetch from SteamSpy
    try:
        res = requests.get(f"https://steamspy.com/api.php?request=appdetails&appid={appid}")
        if res.status_code == 200:
            data = res.json()
        else:
            data = {}
    except Exception:
        data = {}

    # Parse tags into ordered string (first 30 in API order)
    raw_tags = data.get("tags", {})
    if isinstance(raw_tags, dict):
        tag_keys = list(raw_tags.keys())[:30]  # Take first 30 in order
        tags = " ".join([t.replace('-', '_').replace(' ', '_').lower() for t in tag_keys])
    else:
        tags = ""

    # Extract extra data to save in DB
    name = data.get("name", "")
    developer = data.get("developer", "")
    publisher = data.get("publisher", "")

    # Calculate Steam score using a Wilson score interval
    pos, neg = data.get("positive", 0), data.get("negative", 0)
    total = pos + neg
    if total > 0:
        steam_score = (pos / total - (pos / total - 0.5) * (2 ** -math.log10(total + 1))) * 10
    else:
        steam_score = 5.0

    # Try updating the new columns along with tags
    try:
        c.execute("""
            UPDATE games 
            SET tags = ?, 
                steam_score = ?, 
                tags_updated = ?,
                name = CASE WHEN name IS NULL OR name = 'Unknown' THEN ? ELSE name END,
                developer = ?,
                publisher = ?
            WHERE appid = ?
        """, (tags, steam_score, int(time.time()), name, developer, publisher, appid))
    except Exception:
        # Fallback if the columns aren't there for some reason
        c.execute("UPDATE games SET tags = ?, steam_score = ?, tags_updated = ? WHERE appid = ?",
                  (tags, steam_score, int(time.time()), appid))
        
    conn.commit()
    c.execute("SELECT * FROM games WHERE appid = ?", (appid,))
    updated_row = c.fetchone()
    conn.close()
    return dict(updated_row) if updated_row else {}


def is_100_percent_completed(appid):
    """Check if a game has 100% achievement completion via Steam API."""
    conn = get_db()
    c = conn.cursor()

    c.execute(
        "SELECT achievements_completed, finished, playtime FROM games WHERE appid = ?",
        (appid,)
    )
    row = c.fetchone()

    if row and (row['achievements_completed'] or row['finished']):
        conn.close()
        return True

    if row and row['playtime'] == 0:
        conn.close()
        return False

    params = {"appid": appid, "key": STEAM_API_KEY, "steamid": STEAM_ID}

    try:
        res = requests.get(
            "http://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v0001/",
            params=params,
        ).json()

        if "playerstats" in res and res["playerstats"].get("success"):
            achs = res["playerstats"].get("achievements", [])

            if not achs:
                conn.close()
                return False

            # Calculate achievement progress
            total = len(achs)
            unlocked = sum(1 for a in achs if a.get("achieved", 0) == 1)

            # Update achievement counts
            c.execute(
                "UPDATE games SET achievements_total = ?, achievements_unlocked = ? WHERE appid = ?",
                (total, unlocked, appid)
            )
            conn.commit()

            if all(a.get("achieved", 0) == 1 for a in achs):
                c.execute("""
                    UPDATE games
                    SET achievements_completed = 1, finished = 1
                    WHERE appid = ?
                """, (appid,))
                conn.commit()
                conn.close()
                return True
    except Exception:
        pass

    conn.close()
    return False


def sync_game_tags():
    """Fetch and update tags for all games that haven't been updated in the last week."""
    conn = get_db()
    c = conn.cursor()

    # Get games that need tag updates (older than 1 week, never updated, or missing developer column metadata, or empty tags)
    one_week_ago = int(time.time()) - 604800
    try:
        c.execute("""
                  SELECT appid
                  FROM games
                  WHERE tags_updated IS NULL
                     OR tags_updated < ?
                     OR developer IS NULL
                     OR tags = ''
                  """, (one_week_ago,))
    except Exception:
        c.execute("""
                  SELECT appid
                  FROM games
                  WHERE tags_updated IS NULL
                     OR tags_updated < ?
                     OR tags = ''
                  """, (one_week_ago,))

    appids_to_update = [row['appid'] for row in c.fetchall()]

    if not appids_to_update:
        conn.close()
        return

    total_games = len(appids_to_update)
    print(f"Updating tags for {total_games} games...")

    for appid in tqdm(
            appids_to_update,
            desc="🎮 Fetching tags",
            unit="game",
            ncols=100,
            bar_format='{desc}: {percentage:.1f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            colour='cyan'
    ):
        try:
            # This will fetch and update tags if needed
            get_game_data(appid, force_refresh=True)
        except Exception as e:
            tqdm.write(f"Failed to update tags for appid {appid}: {e}")

    print("Tag sync complete!")
    conn.close()
