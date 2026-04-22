import math
import threading
import time

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    NUM_CATEGORIES,
    GAMES_PER_CATEGORY,
    MIN_PLAYTIME,
    CAROUSEL_SIZE,
)
from database import get_db, init_db, cleanup_expired_temp_ratings
from sync import sync_steam_library, sync_cedb_difficulties, sync_game_tags, get_game_data
from recommender import render_game_card, build_persistent_sections

app = Flask(__name__)

# Initialize database on startup
init_db()

# Background sync tracking
_sync_lock = threading.Lock()
_sync_in_progress = False


def _background_sync():
    """Run sync tasks in the background."""
    global _sync_in_progress
    
    with _sync_lock:
        if _sync_in_progress:
            return
        _sync_in_progress = True
    
    try:
        sync_steam_library()
        sync_cedb_difficulties()
        sync_game_tags()
    finally:
        with _sync_lock:
            _sync_in_progress = False


def get_carousel_html():
    """Build HTML for the rating carousel of unrated games."""
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
        flag = " (Finished)" if g['finished'] else ""
        part = f'''
        <div class="rate-card" data-appid="{g['appid']}">
            <div class="btn-group">
                <button class="icon-btn btn-finish" onclick="updateGame({g['appid']}, 'finish', this)">Finish</button>
                <button class="icon-btn btn-up-next" onclick="updateGame({g['appid']}, 'up_next', this)">Up Next</button>
                <button class="icon-btn btn-ban" onclick="updateGame({g['appid']}, 'ban', this)">Ban</button>
                <button class="icon-btn btn-ignore" onclick="updateGame({g['appid']}, 'ignore', this)">Ignore</button>
            </div>
            <img src="https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{g['appid']}/header.jpg">
            <p style="font-size: 0.9em; margin: 5px 0;">{g['name']}{flag}</p>
            <input type="range" class="rate-slider" data-appid="{g['appid']}" min="0" max="10" value="{rating}" oninput="this.nextElementSibling.innerText = this.value">
            <span style="font-weight: bold; color: #66c0f4;">{rating}</span>
        </div>'''
        html_parts.append(part)
    conn.close()
    return "".join(html_parts)


@app.route('/')
def index():
    """Main page - sync library and show rating carousel."""
    # Start background sync
    threading.Thread(target=_background_sync, daemon=True).start()
    
    cleanup_expired_temp_ratings()
    return render_template('index.html', carousel_html=get_carousel_html())


@app.route('/update_game', methods=['POST'])
def update_game():
    """Handle game state updates (rate, ignore, ban, finish, etc.)."""
    data = request.json
    conn = get_db()

    appid = data['appid']
    action = data['action']

    now = int(time.time())
    thirty_days = 30 * 24 * 60 * 60

    if action == 'ignore':
        conn.execute("""
            UPDATE games SET ignore_until = ? WHERE appid = ?
        """, (now + thirty_days, appid))

    elif action == 'ban':
        conn.execute("""
            UPDATE games SET ignored = 1, ignore_until = 0 WHERE appid = ?
        """, (appid,))

    elif action == 'unban':
        conn.execute("""
            UPDATE games SET ignored = 0 WHERE appid = ?
        """, (appid,))

    elif action == 'finish':
        rating = data.get('value', 0)
        conn.execute("""
            UPDATE games SET finished = 1, rating = ? WHERE appid = ?
        """, (rating, appid))

    elif action == 'unfinish':
        conn.execute("""
            UPDATE games SET finished = 0 WHERE appid = ?
        """, (appid,))

    elif action == 'rate':
        rating = data.get('value', 0)
        conn.execute(
            "UPDATE games SET rating = ?, temp_rating = NULL, temp_rating_until = NULL WHERE appid = ?",
            (rating, appid)
        )

    elif action == 'up_next':
        # Set temporary rating of 10 that expires in 3 months
        three_months = 90 * 24 * 60 * 60
        conn.execute("""
            UPDATE games SET temp_rating = 10, temp_rating_until = ? WHERE appid = ?
        """, (now + three_months, appid))

    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate ML-based game recommendations."""
    session_ratings = request.json
    conn = get_db()

    # 1. Save ratings from the carousel
    for aid, score in session_ratings.items():
        conn.execute(
            "UPDATE games SET rating = ?, temp_rating = NULL, temp_rating_until = NULL WHERE appid = ?",
            (score, aid)
        )
    conn.commit()

    # 2. Rated games (profile)
    rated_db_games = [
        dict(r) for r in conn.execute(
            "SELECT * FROM games WHERE rating > 0 AND ignored = 0"
        ).fetchall()
    ]

    # Apply temporary ratings
    now = int(time.time())
    for g in rated_db_games:
        if g.get('temp_rating') and g.get('temp_rating_until', 0) > now:
            g['rating'] = g['temp_rating']

    # 3. Candidate pool
    all_candidates = [
        dict(r) for r in conn.execute("""
            SELECT *
            FROM games
            WHERE ignored = 0
              AND (ignore_until = 0 OR ignore_until < ?)
              AND finished = 0
              AND playtime >= ?
        """, (int(time.time()), MIN_PLAYTIME)).fetchall()
    ]

    backlog = all_candidates[:150]

    if not rated_db_games:
        conn.close()
        return jsonify({
            "results_html": "<h2>Please rate some games first!</h2>",
            "carousel_html": get_carousel_html()
        })

    vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
    all_tags_list = []

    # Profile building
    for g in rated_db_games:
        t, s_db = get_game_data(g['appid'])
        g['tags'] = t
        diff = -math.fabs(g['rating'] - s_db)
        g['weight'] = (g['rating'] * 1.1 + diff)
        all_tags_list.append(t)

    # Backlog prep
    for g in backlog:
        t, s_db = get_game_data(g['appid'])
        g['steam_score'] = s_db

        if g['difficulty'] and g['difficulty'] != 'Easy':
            t += f" {g['difficulty'].replace(' ', '_')}"

        g['tags'] = t

        # Apply temporary rating if active
        if g.get('temp_rating') and g.get('temp_rating_until', 0) > now:
            g['rating'] = g['temp_rating']

        g['priority_boost'] = 1.25 if (g['rating'] > 0 and not g['finished']) else 1.0
        all_tags_list.append(t)

    df_backlog = pd.DataFrame(backlog)
    tfidf = vectorizer.fit_transform(all_tags_list)
    rated_start_idx = 0

    # User vector
    user_vec = np.zeros((1, tfidf.shape[1]))
    for g in rated_db_games:
        if g['tags']:
            game_vec = vectorizer.transform([g['tags']]).toarray()
            user_vec += (game_vec * g['weight'])

    # Matching
    backlog_vecs = vectorizer.transform(df_backlog['tags'])
    df_backlog['match_score'] = cosine_similarity(backlog_vecs, user_vec).flatten() * 80
    df_backlog['match_score'] *= df_backlog['priority_boost']

    # Add rating bonus
    df_backlog['match_score'] += df_backlog['rating'] * 2.0
    df_backlog['match_score'] = np.clip(df_backlog['match_score'], 0, 100)
    df_backlog = df_backlog.sort_values(by='match_score', ascending=False)

    if df_backlog.empty:
        conn.close()
        return jsonify({
            "results_html": "<h2>No matches found. Try rating more games!</h2>",
            "carousel_html": get_carousel_html()
        })

    # Clustering
    num_clusters = min(NUM_CATEGORIES, len(df_backlog) // 2) or 1
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_backlog['cluster'] = kmeans.fit_predict(vectorizer.transform(df_backlog['tags']))

    terms = vectorizer.get_feature_names_out()
    names = {}
    for i in range(num_clusters):
        top_indices = kmeans.cluster_centers_.argsort()[:, ::-1][i, :2]
        names[i] = " & ".join([terms[idx] for idx in top_indices]).title()

    df_backlog['category'] = df_backlog['cluster'].map(names)

    # Build HTML
    # 1. Persistent categories first
    persistent_html, shown_appids = build_persistent_sections(
        df_backlog, rated_db_games, vectorizer, tfidf, rated_start_idx
    )

    # 2. Dynamic cluster columns
    res_html = persistent_html
    for cat, group in df_backlog.groupby('category'):
        filtered = group[~group['appid'].isin(shown_appids)]
        if filtered.empty:
            filtered = group
        res_html += f'<div class="column"><div class="col-title">{cat}</div>'
        for _, r in filtered.head(GAMES_PER_CATEGORY).iterrows():
            res_html += render_game_card(r.to_dict(), rated_db_games, vectorizer, tfidf, rated_start_idx)
        res_html += '</div>'

    conn.close()
    return jsonify({"results_html": res_html, "carousel_html": get_carousel_html()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
