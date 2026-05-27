import math
import time

import requests
from tqdm import tqdm

from config import STEAM_API_KEY, STEAM_ID, CEDB_USER_ID
from database import get_db, get_metadata


def sync_steam_library():
    """Fetch owned games from Steam API and update the database."""
    url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    steam_id = get_metadata('STEAM_ID', STEAM_ID)
    params = {
        "key": STEAM_API_KEY,
        "steamid": steam_id,
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
    cedb_user_id = get_metadata('CEDB_USER_ID', CEDB_USER_ID)
    if not cedb_user_id:
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

    res = requests.get(f"https://cedb.me/api/user/{cedb_user_id}/games")
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
        data = res.json() if res.status_code == 200 else {}
    except Exception:
        data = {}

    # Parse tags into ordered string with weights (first 30 in API order)
    raw_tags = data.get("tags", {})
    tag_list = []
    if isinstance(raw_tags, dict):
        tag_items = list(raw_tags.items())[:30]
        for t, count in tag_items:
            tag_name = t.replace('-', '_').replace(' ', '_').lower()
            tag_list.append((tag_name, count))
    
    tags_str = " ".join([f"{t}:{count}" for t, count in tag_list])

    name = data.get("name", "")
    developer_str = data.get("developer", "")
    publisher_str = data.get("publisher", "")

    # Calculate Steam score using a Wilson score interval
    pos, neg = data.get("positive") or 0, data.get("negative") or 0
    total = pos + neg
    if total > 0:
        steam_score = (pos / total - (pos / total - 0.5) * (2 ** -math.log10(total + 1))) * 10
    else:
        steam_score = 5.0

    # Update main games table
    c.execute("""
        UPDATE games 
        SET tags = ?, 
            steam_score = ?, 
            tags_updated = ?,
            name = CASE WHEN name IS NULL OR name = 'Unknown' THEN ? ELSE name END,
            developer = ?,
            publisher = ?
        WHERE appid = ?
    """, (tags_str, steam_score, int(time.time()), name, developer_str, publisher_str, appid))

    # Normalized storage
    _update_normalized_data(c, appid, tag_list, developer_str, publisher_str)
        
    conn.commit()
    c.execute("SELECT * FROM games WHERE appid = ?", (appid,))
    updated_row = c.fetchone()
    conn.close()
    return dict(updated_row) if updated_row else {}


def _update_normalized_data(cursor, appid, tag_list, developer_str, publisher_str):
    """Update normalized tables for tags, developers, and publishers."""
    # 1. Tags
    for tag_name, count in tag_list:
        cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
        cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
        tag_id = cursor.fetchone()[0]
        cursor.execute("INSERT OR REPLACE INTO game_tags (appid, tag_id, count) VALUES (?, ?, ?)",
                       (appid, tag_id, count))

    # 2. Developers
    if developer_str:
        devs = [d.strip() for d in developer_str.split(',')]
        for dev_name in devs:
            cursor.execute("INSERT OR IGNORE INTO developers (name) VALUES (?)", (dev_name,))
            cursor.execute("SELECT id FROM developers WHERE name = ?", (dev_name,))
            dev_id = cursor.fetchone()[0]
            cursor.execute("INSERT OR REPLACE INTO game_developers (appid, developer_id) VALUES (?, ?)",
                           (appid, dev_id))

    # 3. Publishers
    if publisher_str:
        pubs = [p.strip() for p in publisher_str.split(',')]
        for pub_name in pubs:
            cursor.execute("INSERT OR IGNORE INTO publishers (name) VALUES (?)", (pub_name,))
            cursor.execute("SELECT id FROM publishers WHERE name = ?", (pub_name,))
            pub_id = cursor.fetchone()[0]
            cursor.execute("INSERT OR REPLACE INTO game_publishers (appid, publisher_id) VALUES (?, ?)",
                           (appid, pub_id))


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
