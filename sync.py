import math
import time

import requests

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
                ON CONFLICT(appid) DO UPDATE SET
                    playtime    = excluded.playtime,
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


def get_game_data(appid):
    """
    Fetch game tags and Steam score from SteamSpy.
    Returns cached data if available, otherwise fetches and stores it.
    """
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT tags, steam_score FROM games WHERE appid = ?", (appid,))
    row = c.fetchone()

    if row and row['tags']:
        conn.close()
        return row['tags'], row['steam_score']

    # Fetch from SteamSpy
    try:
        res = requests.get(f"https://steamspy.com/api.php?request=appdetails&appid={appid}")
        if res.status_code == 200:
            data = res.json()
        else:
            data = {}
    except Exception:
        data = {}

    # Parse tags
    raw_tags = data.get("tags", {})
    if isinstance(raw_tags, dict):
        tags = " ".join(list(raw_tags.keys())).replace("-", "_")
    else:
        tags = ""

    # Add game name as a tag
    name = data.get("name", "").replace(" ", "_").lower()
    if name:
        tags += f" {name}"

    # Add developer and publisher as tags
    developer = data.get("developer", "")
    publisher = data.get("publisher", "")
    if developer:
        tags += f" {developer.replace(' ', '_')}"
    if publisher and publisher != developer:
        tags += f" {publisher.replace(' ', '_')}"

    # Calculate Steam score using Wilson score interval
    pos, neg = data.get("positive", 0), data.get("negative", 0)
    total = pos + neg
    if total > 0:
        steam_score = (pos / total - (pos / total - 0.5) * (2 ** -math.log10(total + 1))) * 10
    else:
        steam_score = 5.0

    c.execute("UPDATE games SET tags = ?, steam_score = ? WHERE appid = ?", (tags, steam_score, appid))
    conn.commit()
    conn.close()
    return tags, steam_score


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
