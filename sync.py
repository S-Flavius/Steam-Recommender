import math
import time

import requests
from tqdm import tqdm

from config import STEAM_API_KEY, STEAM_ID, CEDB_USER_ID
from database import engine, Game, Developer, Publisher, Tag, GameDeveloper, GamePublisher, GameTag, get_metadata, set_metadata, get_db
from sqlmodel import Session, select, or_


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
        with Session(engine) as session:
            for g in games:
                appid = g['appid']
                name = g.get('name', 'Unknown')
                playtime = g.get('playtime_forever', 0)
                last_played = g.get('rtime_last_played', 0)
                
                game = session.get(Game, appid)
                if game:
                    game.playtime = playtime
                    game.last_played = last_played
                    game.name = name
                else:
                    game = Game(appid=appid, name=name, playtime=playtime, last_played=last_played)
                session.add(game)
            session.commit()


def sync_cedb_difficulties():
    """Fetch difficulty tiers from CEDB (completionist.me) and update games."""
    cedb_user_id = get_metadata('CEDB_USER_ID', CEDB_USER_ID)
    if not cedb_user_id:
        return

    # Check if we've synced recently (within 7 days)
    last_sync = get_metadata('last_cedb_sync')
    now = time.time()
    if last_sync and now - float(last_sync) < 604800:  # 7 days in seconds
        return

    res = requests.get(f"https://cedb.me/api/user/{cedb_user_id}/games")
    if res.status_code == 200:
        with Session(engine) as session:
            for item in res.json():
                game_data = item.get('game', {})
                if str(game_data.get('platform')).lower() == 'steam':
                    appid = int(game_data.get('platformId'))
                    diff = f"T{game_data.get('tier')} (Challenge)"
                    game = session.get(Game, appid)
                    if game:
                        game.difficulty = diff
                        session.add(game)
            session.commit()
        set_metadata('last_cedb_sync', str(now))


def get_game_data(appid, force_refresh=False):
    """
    Fetch game tags and Steam score from SteamSpy.
    Returns cached data if available, otherwise fetches and stores it.
    """
    with Session(engine) as session:
        game = session.get(Game, appid)
        if not force_refresh and game and game.tags:
            return game.dict()

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
    with Session(engine) as session:
        game = session.get(Game, appid)
        if game:
            game.tags = tags_str
            game.steam_score = steam_score
            game.tags_updated = int(time.time())
            if not game.name or game.name == 'Unknown':
                game.name = name
            game.developer = developer_str
            game.publisher = publisher_str
            session.add(game)
            session.commit()
            
            # For normalized data, we still need a cursor-like helper or refactor it too
            conn = get_db()
            _update_normalized_data(conn.cursor(), appid, tag_list, developer_str, publisher_str)
            conn.commit()
            conn.close()
            
            session.refresh(game)
            return game.dict()
    return {}


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
    with Session(engine) as session:
        game = session.get(Game, appid)
        if not game:
            return False

        if game.achievements_completed or game.finished:
            return True

        if game.playtime == 0:
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
                return False

            # Calculate achievement progress
            total = len(achs)
            unlocked = sum(1 for a in achs if a.get("achieved", 0) == 1)
            completed = all(a.get("achieved", 0) == 1 for a in achs)

            # Update achievement counts
            with Session(engine) as session:
                game = session.get(Game, appid)
                if game:
                    game.achievements_total = total
                    game.achievements_unlocked = unlocked
                    if completed:
                        game.achievements_completed = True
                        game.finished = True
                    session.add(game)
                    session.commit()
            return completed
    except Exception:
        pass

    return False


def sync_game_tags():
    """Fetch and update tags for all games that haven't been updated in the last week."""
    # Get games that need tag updates (older than 1 week, never updated, or missing developer column metadata, or empty tags)
    one_week_ago = int(time.time()) - 604800
    
    with Session(engine) as session:
        statement = select(Game.appid).where(
            or_(
                Game.tags_updated == None,
                Game.tags_updated < one_week_ago,
                Game.developer == None,
                Game.tags == ''
            )
        )
        appids_to_update = session.exec(statement).all()

    if not appids_to_update:
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
