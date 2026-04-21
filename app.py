import math
import os
import sqlite3
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, request, render_template_string, jsonify
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backlog import STEAM_ID

# --- Configuration ---
load_dotenv()
STEAM_API_KEY = os.getenv("STEAM_API_KEY")
STEAM_ID = os.getenv("STEAM_ID")
CEDB_USER_ID = os.getenv("CEDB_USER_ID")

NUM_CATEGORIES = 8
GAMES_PER_CATEGORY = 10
MIN_PLAYTIME = 60
CAROUSEL_SIZE = 15
DB_FILE = f"{STEAM_ID}.games.db"

CHILL_TAGS = {"casual", "relaxing", "cozy", "cute", "puzzle", "idler", "clicker", "incremental", "walking_simulator",
              "life_sim", "farming_sim", "colorful", "wholesome", }
app = Flask(__name__)


# --- Database Management ---
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
              CREATE TABLE IF NOT EXISTS games
              (
                  appid                  INTEGER PRIMARY KEY,
                  name                   TEXT,
                  playtime               INTEGER DEFAULT 0,
                  last_played            INTEGER DEFAULT 0,
                  rating                 INTEGER DEFAULT 0,
                  ignored                BOOLEAN DEFAULT 0,
                  finished               BOOLEAN DEFAULT 0,
                  difficulty             TEXT    DEFAULT 'Easy',
                  tags                   TEXT    DEFAULT NULL,
                  steam_score            REAL    DEFAULT NULL,
                  achievements_completed BOOLEAN DEFAULT 0
              )
              ''')
    try:
        c.execute('ALTER TABLE games ADD COLUMN finished BOOLEAN DEFAULT 0')
    except:
        pass

    try:
        c.execute('ALTER TABLE games ADD COLUMN ignore_until INTEGER DEFAULT 0')
    except:
        pass

    try:
        c.execute('ALTER TABLE games ADD COLUMN temp_rating INTEGER DEFAULT NULL')
    except:
        pass

    try:
        c.execute('ALTER TABLE games ADD COLUMN temp_rating_until INTEGER DEFAULT NULL')
    except:
        pass

    c.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
    conn.commit()
    conn.close()


init_db()


# --- Syncing Logic ---
def sync_steam_library():
    url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {"key": STEAM_API_KEY, "steamid": STEAM_ID, "format": "json", "include_appinfo": True}
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
                         ''', (g["appid"], g.get("name", "Unknown"), g.get("playtime_forever", 0),
                               g.get("rtime_last_played", 0)))
        conn.commit()
        conn.close()


def cleanup_expired_temp_ratings():
    """Remove expired temporary ratings."""
    conn = get_db()
    now = int(time.time())
    conn.execute("UPDATE games SET temp_rating = NULL, temp_rating_until = NULL WHERE temp_rating_until < ?", (now,))
    conn.commit()
    conn.close()


def sync_cedb_difficulties():
    if not CEDB_USER_ID: return
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT value FROM metadata WHERE key = 'last_cedb_sync'")
    row = c.fetchone()
    now = time.time()
    if row and now - float(row['value']) < 604800:
        conn.close()
        return

    res = requests.get(f"https://cedb.me/api/user/{CEDB_USER_ID}/games")
    if res.status_code == 200:
        updates = []
        for item in res.json():
            game = item.get('game', {})
            if str(game.get('platform')).lower() == 'steam':
                updates.append((f"T{game.get('tier')} (Challenge)", int(game.get('platformId'))))
        c.executemany("UPDATE games SET difficulty = ? WHERE appid = ?", updates)
        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_cedb_sync', ?)", (str(now),))
        conn.commit()
    conn.close()


def get_game_data(appid):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT tags, steam_score FROM games WHERE appid = ?", (appid,))
    row = c.fetchone()
    if row and row['tags']:
        conn.close()
        return row['tags'], row['steam_score']

    try:
        res = requests.get(f"https://steamspy.com/api.php?request=appdetails&appid={appid}")
        if res.status_code == 200:
            data = res.json()
        else:
            data = {}
    except:
        data = {}

    raw_tags = data.get("tags", {})
    if isinstance(raw_tags, dict):
        tags = " ".join(list(raw_tags.keys())).replace("-", "_")
    else:
        tags = ""

    name = data.get("name", "").replace(" ", "_").lower()
    if name:
        tags += f" {name}"

    developer = data.get("developer", "")
    publisher = data.get("publisher", "")
    if developer:
        tags += f" {developer.replace(' ', '_')}"
    if publisher and publisher != developer:
        tags += f" {publisher.replace(' ', '_')}"

    pos, neg = data.get("positive", 0), data.get("negative", 0)
    total = pos + neg
    s_db = (pos / total - (pos / total - 0.5) * (2 ** -math.log10(total + 1))) * 10 if total > 0 else 5.0
    c.execute("UPDATE games SET tags = ?, steam_score = ? WHERE appid = ?", (tags, s_db, appid))
    conn.commit()
    conn.close()
    return tags, s_db


def is_100_percent_completed(appid):
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT achievements_completed, finished, playtime FROM games WHERE appid = ?", (appid,))
    row = c.fetchone()

    if row and (row['achievements_completed'] or row['finished']):
        conn.close()
        return True

    if row and row['playtime'] == 0:
        conn.close()
        return False

    params = {"appid": appid, "key": STEAM_API_KEY, "steamid": STEAM_ID}

    try:
        res = requests.get("http://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v0001/",
                           params=params).json()

        if "playerstats" in res and res["playerstats"].get("success"):
            achs = res["playerstats"].get("achievements", [])

            if not achs:
                conn.close()
                return False

            if all(a.get("achieved", 0) == 1 for a in achs):
                c.execute("""
                          UPDATE games
                          SET achievements_completed = 1,
                              finished               = 1
                          WHERE appid = ?
                          """, (appid,))
                conn.commit()
                conn.close()
                return True

    except:
        pass

    conn.close()
    return False


# --- Explanation Generator ---
def build_explanation(candidate, rated_db_games, vectorizer, tfidf_matrix, rated_start_idx):
    """
    Build a human-readable explanation for why a game was recommended.
    Returns a list of short reason strings.
    """
    reasons = []

    cand_tags = set(candidate.get('tags', '').split())
    steam_score = candidate.get('steam_score')
    difficulty = candidate.get('difficulty', 'Easy')
    rating = candidate.get('rating', 0)
    finished = candidate.get('finished', 0)
    playtime = candidate.get('playtime', 0)

    # --- 1. Similar to games you liked ---
    similar_to = []
    for i, rg in enumerate(rated_db_games):
        if rg['rating'] < 5:
            continue
        rg_tags = set(rg.get('tags', '').split())
        overlap = cand_tags & rg_tags
        if len(overlap) >= 2:
            similar_to.append((rg['name'], rg['rating'], len(overlap)))

    similar_to.sort(key=lambda x: (x[2], x[1]), reverse=True)
    if similar_to:
        top = similar_to[:2]
        names = " & ".join(f"<b>{g[0]}</b>" for g in top)
        reasons.append(f"🎮 Similar to {names}")

    # --- 2. Steam score signal ---
    if steam_score is not None:
        if steam_score >= 8.5:
            reasons.append(f"⭐ Highly rated on Steam ({steam_score:.1f}/10)")
        elif steam_score >= 7.0:
            reasons.append(f"👍 Well received on Steam ({steam_score:.1f}/10)")
        elif steam_score < 5.0:
            reasons.append(f"⚠️ Mixed reviews ({steam_score:.1f}/10)")

    # --- 3. Challenge / difficulty signal ---
    if difficulty and difficulty != 'Easy':
        reasons.append(f"💀 Challenging: {difficulty}")

    # --- 4. You've played it before but never finished ---
    if rating > 0 and not finished:
        reasons.append(f"🔁 You played this (rated {rating}/10) but never finished it")

    # --- 5. Long playtime already invested ---
    if playtime > 600 and not finished:
        hours = round(playtime / 60, 1)
        reasons.append(f"⏱️ You have {hours}h in this — worth finishing?")

    # --- 6. Tag highlights (top shared tags with your profile) ---
    # Find top tags across all liked games
    liked_tags = {}
    for rg in rated_db_games:
        if rg['rating'] >= 7:
            for t in rg.get('tags', '').split():
                liked_tags[t] = liked_tags.get(t, 0) + 1

    highlight_tags = [t for t in cand_tags if liked_tags.get(t, 0) >= 2]
    highlight_tags.sort(key=lambda t: liked_tags.get(t, 0), reverse=True)
    if highlight_tags:
        tag_str = ", ".join(t.replace("_", " ").title() for t in highlight_tags[:3])
        reasons.append(f"🏷️ Matches your taste in: {tag_str}")

    if not reasons:
        reasons.append("🤖 Matches your overall taste profile")

    return reasons


def render_game_card(r, rated_db_games, vectorizer, tfidf, rated_start_idx):
    """Render a single game card (used by both persistent and dynamic columns)."""
    replay_flag = " 🔁" if (r.get('rating', 0) > 0 and not r.get('finished', 0)) else ""
    reasons = build_explanation(r, rated_db_games, vectorizer, tfidf, rated_start_idx)
    why_html = "".join(f"<span>{reason}</span>" for reason in reasons)

    # Determine which finish button to show
    finish_btn = f'<button class="icon-btn btn-finish" onclick="finishGame({r["appid"]}, this)">✅</button>' if not r.get(
        'finished',
        0) else f'<button class="icon-btn btn-unfinish" onclick="updateGame({r["appid"]}, \'unfinish\', this)">↩️</button>'

    return f'''
            <div class="game-card">
                <div class="btn-group">
                    {finish_btn}
                    <button class="icon-btn btn-up-next" onclick="updateGame({r['appid']}, 'up_next', this)">⏭️</button>
                    <button class="icon-btn btn-ban" onclick="updateGame({r['appid']}, 'ban', this)">⛔</button>
                    <button class="icon-btn btn-ignore" onclick="updateGame({r['appid']}, 'ignore', this)">🚫</button>
                </div>
                <img src="https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{r['appid']}/header.jpg">
                <div style="padding: 5px;">
                    <b>{r['name']}{replay_flag}</b><br>
                    <span class="match-score">
                        Match: {round(r['match_score'], 1)}%
                    </span>
                    <div style="display:flex; align-items:center; gap:6px; margin-top:5px;">
                        <span style="font-size:0.75em; color:#8899aa;">Rating:</span>
                        <input type="range" min="0" max="10" value="{int(r['rating'])}"
                               style="flex:1; accent-color:#66c0f4;"
                               onchange="rateCard({r['appid']}, this)">
                        <span style="font-weight:bold; color:#66c0f4; min-width:14px;">{int(r['rating'])}</span>
                    </div>
                    <button class="why-toggle" onclick="toggleWhy(this)">▼ Why?</button>
                    <div class="why-box">{why_html}</div>
                </div>
            </div>'''


def build_persistent_sections(df_backlog, rated_db_games, vectorizer, tfidf, rated_start_idx):
    """
    Build the HTML for the always-present categories:
    🏆 Top Games, 💀 Hard Games, 🌿 Chill Games, 🔄 Recently Played Unfinished, ⏳ Forgotten Games.
    Returns (html_string, set_of_appids_already_shown).
    """
    sections = []
    shown = set()
    now = time.time()

    def make_column(title, rows):
        if rows.empty:
            return ""
        html = f'<div class="column"><div class="col-title">{title}</div>'
        for _, r in rows.iterrows():
            html += render_game_card(r.to_dict(), rated_db_games, vectorizer, tfidf, rated_start_idx)
            shown.add(int(r['appid']))
        html += '</div>'
        return html

    # 🏆 Top Games — best overall matches
    top_rows = df_backlog.head(GAMES_PER_CATEGORY)
    sections.append(make_column("🏆 Top Games", top_rows))

    # 💀 Hard Games — anything not 'Easy' difficulty
    hard_mask = df_backlog['difficulty'].fillna('Easy').str.lower() != 'easy'
    hard_rows = df_backlog[hard_mask].head(GAMES_PER_CATEGORY)
    sections.append(make_column("💀 Hard Games", hard_rows))

    # 🌿 Chill Games — tag-based
    def is_chill(tag_str):
        if not tag_str:
            return False
        tokens = {t.lower() for t in tag_str.split()}
        return bool(tokens & CHILL_TAGS)

    chill_mask = df_backlog['tags'].apply(is_chill)
    chill_rows = df_backlog[chill_mask].head(GAMES_PER_CATEGORY)
    sections.append(make_column("🌿 Chill Games", chill_rows))

    # 🔄 Recently Played Unfinished — unfinished games played within last 30 days
    recent_mask = (df_backlog['finished'] == 0) & (df_backlog['last_played'] > now - 30 * 24 * 3600)
    recent_rows = df_backlog[recent_mask].sort_values('last_played', ascending=False).head(GAMES_PER_CATEGORY)
    sections.append(make_column("🔄 Recently Played Unfinished", recent_rows))

    # ⏳ Forgotten Games — unfinished games not played in over 1 year
    forgotten_mask = (df_backlog['finished'] == 0) & (df_backlog['last_played'] < now - 365 * 24 * 3600)
    forgotten_rows = df_backlog[forgotten_mask].sort_values('last_played', ascending=True).head(GAMES_PER_CATEGORY)
    sections.append(make_column("⏳ Forgotten Games", forgotten_rows))

    return "".join(sections), shown


# --- HTML/UI ---
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Backlog Recommender</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #121a25; color: #c7d5e0; padding: 20px; }
        .carousel-container { display: flex; overflow-x: auto; gap: 15px; padding-bottom: 20px; border-bottom: 2px solid #2a475e; min-height: 250px; }
        .rate-card { position: relative; background: #1b2838; padding: 10px; border-radius: 8px; min-width: 200px; text-align: center; transition: opacity 0.3s; }
        .rate-card img { width: 100%; border-radius: 4px; }
        .btn-group { position: absolute; top: 5px; right: 5px; display: flex; gap: 5px; }
        .icon-btn { background: rgba(0,0,0,0.7); border: none; border-radius: 50%; cursor: pointer; color: white; padding: 5px; font-size: 0.9em; }
        .icon-btn:hover { transform: scale(1.1); }
        .btn-finish:hover { background: #a3e635; }
        .btn-unfinish:hover { background: #66c0f4; }
        .btn-up-next:hover { background: #ff6b35; }
        .btn-ban:hover { background: #ff3b3b; }
        .btn-ignore:hover { background: #f1c40f; }
        .btn-submit { display: block; width: 300px; margin: 20px auto; padding: 15px; background: #66c0f4; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; }
        .btn-submit:hover { background: #4b6982; color: #fff; }
        .board { display: flex; gap: 15px; overflow-x: auto; padding-top: 20px; }
        .column { flex: 0 0 280px; background: #1b2838; padding: 15px; border-radius: 8px; border-top: 4px solid #66c0f4; }
        .game-card { background: #2a475e; padding: 8px; border-radius: 6px; margin-bottom: 8px; position: relative; }
        .game-card img { width: 100%; border-radius: 4px; }
        .match-score { color: #a3e635; font-weight: bold; font-size: 0.9em; }
        .loading-text { text-align: center; display: none; margin-top: 10px; font-style: italic; color: #66c0f4; }

        /* Explanation styles */
        .why-toggle { background: none; border: 1px solid #4b6982; color: #66c0f4; border-radius: 4px;
                      font-size: 0.75em; cursor: pointer; padding: 2px 7px; margin-top: 4px; }
        .why-toggle:hover { background: #1b2838; }
        .why-box { display: none; margin-top: 6px; background: #1b2838; border-radius: 5px;
                   padding: 6px 8px; font-size: 0.78em; line-height: 1.6; color: #b8c9d9; }
        .why-box.open { display: block; }
        .why-box span { display: block; }
    </style>
</head>
<body>
    <h1>Rate & Categorize Your Backlog</h1>
    <div class="carousel-container" id="carousel">{{ carousel_html | safe }}</div>

    <button class="btn-submit" id="trainBtn" onclick="generateRecommendations()">Train AI & Re-Cluster</button>
    <div class="loading-text" id="loading">AI is crunching the data... Please wait...</div>

    <div id="results" class="board"></div>

    <script>
        function updateGame(appid, action, btn, value=null) {
            let card = btn.closest('.rate-card') || btn.closest('.game-card');
            card.style.opacity = '0.5';

            const payload = {appid: appid, action: action};
            if (value !== null) payload.value = value;

            fetch('/update_game', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(() => {
                if (action !== 'rate') {
                    card.style.opacity = '0';
                    setTimeout(() => card.remove(), 300);
                } else {
                    card.style.opacity = '1';
                }
            });
        }

        function rateCard(appid, slider) {
            const val = parseInt(slider.value);
            slider.nextElementSibling.textContent = val;
            updateGame(appid, 'rate', slider, val);
        }

        function finishGame(appid, btn) {
            let card = btn.closest('.game-card');
            let ratingSlider = card.querySelector('input[type="range"]');
            let ratingValue = ratingSlider ? parseInt(ratingSlider.value) : 0;
            updateGame(appid, 'finish', btn, ratingValue);
        }

        function toggleWhy(btn) {
            const box = btn.nextElementSibling;
            box.classList.toggle('open');
            btn.textContent = box.classList.contains('open') ? '▲ Why?' : '▼ Why?';
        }

        function generateRecommendations() {
            const trainBtn = document.getElementById('trainBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');

            trainBtn.disabled = true;
            loading.style.display = 'block';
            results.innerHTML = '';

            let ratings = {};
            document.querySelectorAll('.rate-slider').forEach(s => {
                let val = parseInt(s.value);
                if (val > 0) {
                    ratings[s.getAttribute('data-appid')] = val;
                }
            });

            fetch('/recommend', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(ratings)
            })
            .then(res => res.json())
            .then((data) => {
                trainBtn.disabled = false;
                loading.style.display = 'none';
                results.innerHTML = data.results_html;
                document.getElementById('carousel').innerHTML = data.carousel_html;
            })
            .catch(err => {
                console.error("AI Training Failed:", err);
                trainBtn.disabled = false;
                loading.style.display = 'none';
                alert("Something went wrong with the AI training.");
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    sync_steam_library()
    sync_cedb_difficulties()
    cleanup_expired_temp_ratings()
    return render_template_string(HTML_PAGE, carousel_html=get_carousel_html())


@app.route('/update_game', methods=['POST'])
def update_game():
    data = request.json
    conn = get_db()

    appid = data['appid']
    action = data['action']

    now = int(time.time())
    thirty_days = 30 * 24 * 60 * 60

    if action == 'ignore':
        conn.execute("""
                     UPDATE games
                     SET ignore_until = ?
                     WHERE appid = ?
                     """, (now + thirty_days, appid))

    elif action == 'ban':
        conn.execute("""
                     UPDATE games
                     SET ignored      = 1,
                         ignore_until = 0
                     WHERE appid = ?
                     """, (appid,))

    elif action == 'unban':
        conn.execute("""
                     UPDATE games
                     SET ignored = 0
                     WHERE appid = ?
                     """, (appid,))

    elif action == 'finish':
        rating = data.get('value', 0)
        conn.execute("""
                     UPDATE games
                     SET finished = 1,
                         rating   = ?
                     WHERE appid = ?
                     """, (rating, appid))

    elif action == 'unfinish':
        conn.execute("""
                     UPDATE games
                     SET finished = 0
                     WHERE appid = ?
                     """, (appid,))

    elif action == 'rate':
        rating = data.get('value', 0)
        conn.execute("UPDATE games SET rating = ?, temp_rating = NULL, temp_rating_until = NULL WHERE appid = ?",
                     (rating, appid))

    elif action == 'up_next':
        # Set temporary rating of 10 that expires in 3 months
        three_months = 90 * 24 * 60 * 60
        conn.execute("""
                     UPDATE games
                     SET temp_rating       = 10,
                         temp_rating_until = ?
                     WHERE appid = ?
                     """, (now + three_months, appid))

    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route('/recommend', methods=['POST'])
def recommend():
    session_ratings = request.json
    conn = get_db()

    # 1. Save ratings from the carousel
    for aid, score in session_ratings.items():
        conn.execute("UPDATE games SET rating = ?, temp_rating = NULL, temp_rating_until = NULL WHERE appid = ?",
                     (score, aid))
    conn.commit()

    # 2. Rated games (profile)
    rated_db_games = [dict(r) for r in conn.execute("SELECT * FROM games WHERE rating > 0 AND ignored = 0").fetchall()]

    # Apply temporary ratings
    now = int(time.time())
    for g in rated_db_games:
        if g.get('temp_rating') and g.get('temp_rating_until', 0) > now:
            g['rating'] = g['temp_rating']

    # 3. Candidate pool
    all_candidates = [dict(r) for r in conn.execute("""
                                                    SELECT *
                                                    FROM games
                                                    WHERE ignored = 0
                                                      AND (ignore_until = 0 OR ignore_until < ?)
                                                      AND finished = 0
                                                      AND playtime >= ?
                                                    """, (int(time.time()), MIN_PLAYTIME)).fetchall()]
    backlog = []
    for g in all_candidates:
        backlog.append(g)
        if len(backlog) >= 150:
            break

    if not rated_db_games:
        conn.close()
        return jsonify({"results_html": "<h2>Please rate some games first!</h2>", "carousel_html": get_carousel_html()})

    vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
    all_tags_list = []

    # --- PROFILE BUILDING ---
    for g in rated_db_games:
        t, s_db = get_game_data(g['appid'])
        g['tags'] = t

        diff = -math.fabs(g['rating'] - s_db)
        g['weight'] = (g['rating'] * 1.1 + diff)

        all_tags_list.append(t)

    # --- BACKLOG PREP ---
    for g in backlog:
        t, s_db = get_game_data(g['appid'])
        g['steam_score'] = s_db  # store for explanation

        if g['difficulty'] and g['difficulty'] != 'Easy':
            t += f" {g['difficulty'].replace(' ', '_')}"

        g['tags'] = t

        # Apply temporary rating if active
        if g.get('temp_rating') and g.get('temp_rating_until', 0) > now:
            g['rating'] = g['temp_rating']

        if g['rating'] > 0 and not g['finished']:
            g['priority_boost'] = 1.25;
        else:
            g['priority_boost'] = 1.0

        all_tags_list.append(t)

    df_backlog = pd.DataFrame(backlog)
    tfidf = vectorizer.fit_transform(all_tags_list)

    rated_start_idx = 0  # rated games come first in all_tags_list

    # --- USER VECTOR ---
    user_vec = np.zeros((1, tfidf.shape[1]))
    for g in rated_db_games:
        if g['tags']:
            game_vec = vectorizer.transform([g['tags']]).toarray()
            user_vec += (game_vec * g['weight'])

    # --- MATCHING ---
    backlog_vecs = vectorizer.transform(df_backlog['tags'])
    df_backlog['match_score'] = cosine_similarity(backlog_vecs, user_vec).flatten() * 80
    df_backlog['match_score'] *= df_backlog['priority_boost']

    # Add rating bonus to prevent low-rated games from overtaking high-rated ones
    df_backlog['match_score'] += df_backlog['rating'] * 2.0

    # Normalize internal scores to 0-100 for fairness
    df_backlog['match_score'] = np.clip(df_backlog['match_score'], 0, 100)

    df_backlog = df_backlog.sort_values(by='match_score', ascending=False)

    if df_backlog.empty:
        conn.close();
        return jsonify(
            {"results_html": "<h2>No matches found. Try rating more games!</h2>", "carousel_html": get_carousel_html()})

    # --- CLUSTERING ---
    num_clusters = min(NUM_CATEGORIES, len(df_backlog) // 2) or 1
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_backlog['cluster'] = kmeans.fit_predict(vectorizer.transform(df_backlog['tags']))

    terms = vectorizer.get_feature_names_out()
    names = {}
    for i in range(num_clusters):
        top_indices = kmeans.cluster_centers_.argsort()[:, ::-1][i, :2]
        names[i] = " & ".join([terms[idx] for idx in top_indices]).title()

    df_backlog['category'] = df_backlog['cluster'].map(names)

    # --- HTML ---
    # 1. Persistent categories first
    persistent_html, shown_appids = build_persistent_sections(df_backlog, rated_db_games, vectorizer, tfidf,
                                                              rated_start_idx)

    # 2. Dynamic cluster columns (skip games already shown in persistent sections
    #    to keep these columns fresh — they can still reappear if a column would be empty otherwise)
    res_html = persistent_html
    for cat, group in df_backlog.groupby('category'):
        filtered = group[~group['appid'].isin(shown_appids)]
        # Fallback: if filtering emptied the column, use the original group
        if filtered.empty:
            filtered = group
        res_html += f'<div class="column"><div class="col-title">{cat}</div>'
        for _, r in filtered.head(GAMES_PER_CATEGORY).iterrows():
            res_html += render_game_card(r.to_dict(), rated_db_games, vectorizer, tfidf, rated_start_idx)
        res_html += '</div>'

    conn.close()
    return jsonify({"results_html": res_html, "carousel_html": get_carousel_html()})


def get_carousel_html():
    conn = get_db()
    games = conn.execute("""
                         SELECT *
                         FROM games
                         WHERE rating = 0
                           AND ignored = 0
                           AND (ignore_until = 0 OR ignore_until < ?)
                           AND (temp_rating IS NULL OR temp_rating_until < ?)
                         ORDER BY CASE WHEN finished = 1 THEN 0 ELSE 1 END,
                                  playtime DESC
                         LIMIT ?
                         """, (int(time.time()), int(time.time()), CAROUSEL_SIZE)).fetchall()
    html_parts = []
    for g in games:
        rating = g['rating'] or 0
        flag = " ✅ Finished" if g['finished'] else ""
        part = f'''
        <div class="rate-card" data-appid="{g['appid']}">
            <div class="btn-group">
                <button class="icon-btn btn-finish" onclick="updateGame({g['appid']}, 'finish', this)">✅</button>
                <button class="icon-btn btn-up-next" onclick="updateGame({g['appid']}, 'up_next', this)">⏭️</button>
                <button class="icon-btn btn-ban" onclick="updateGame({g['appid']}, 'ban', this)">⛔</button>
                <button class="icon-btn btn-ignore" onclick="updateGame({g['appid']}, 'ignore', this)">🚫</button>
            </div>
            <img src="https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{g['appid']}/header.jpg">
            <p style="font-size: 0.9em; margin: 5px 0;">{g['name']}{flag}</p>
            <input type="range" class="rate-slider" data-appid="{g['appid']}" min="0" max="10" value="{rating}" oninput="this.nextElementSibling.innerText = this.value">
            <span style="font-weight: bold; color: #66c0f4;">{rating}</span>
        </div>'''
        html_parts.append(part)
    return "".join(html_parts)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
