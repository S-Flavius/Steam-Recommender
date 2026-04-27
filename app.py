import math
import threading
import time

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (NUM_CATEGORIES, GAMES_PER_CATEGORY, MIN_PLAYTIME, CAROUSEL_SIZE, )
from database import get_db, init_db, cleanup_expired_temp_ratings
from recommender import render_game_card, build_persistent_sections
from sync import sync_steam_library, sync_cedb_difficulties, sync_game_tags, get_game_data

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


def extract_tags_and_weights(tag_str, default_weight=0.1):
    """
    Parses a space-separated tag string, assigns weights based on position (decreasing).
    Weights decay linearly from 1.0 to default_weight over all tags.
    Returns a list of tags, and a dict of weights.
    """
    if not tag_str:
        return [], {}

    tags = []
    weights = {}

    tokens = tag_str.split()

    num_tags = len(tokens)

    for i, tag in enumerate(tokens):
        if num_tags > 1:
            weight = 1.0 - (i / (num_tags - 1)) * (1.0 - default_weight)
        else:
            weight = 1.0

        tags.append(tag)
        weights[tag] = weight

    return tags, weights


@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate ML-based game recommendations."""
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

    backlog = all_candidates[:150]

    if not rated_db_games:
        conn.close()
        return jsonify({"results_html": "<h2>Please rate some games first!</h2>", "carousel_html": get_carousel_html()})

    # Custom token pattern to keep tags like "action_rpg" and "1990's" intact
    vectorizer = TfidfVectorizer(stop_words='english', max_features=300, token_pattern=r"(?u)\S+")

    # We will build a list of space-separated tags (without weights) to fit the vectorizer
    all_tags_list = []

    # We will also keep track of the weight dictionary for each game
    game_weights = []

    # Set to track metadata tags (dev/pub) to exclude from cluster names
    meta_tags = set()

    # Profile building
    for g in rated_db_games:
        # Load extra fields dynamically if needed
        game_data = get_game_data(g['appid'])
        t = game_data.get('tags', '')
        s_db = game_data.get('steam_score', 5.0)

        g['tags'] = t
        diff = -math.fabs(float(g['rating']) - float(s_db))
        weight = float(g['rating']) * 1.1 + diff

        tags, weights = extract_tags_and_weights(t)

        # Add metadata as tags with max weight
        if game_data.get('developer'):
            dev_tag = game_data['developer'].replace(' ', '_').lower()
            tags.append(dev_tag)
            weights[dev_tag] = 1.0
            meta_tags.add(dev_tag)
        if game_data.get('publisher') and game_data.get('publisher') != game_data.get('developer'):
            pub_tag = game_data['publisher'].replace(' ', '_').lower()
            tags.append(pub_tag)
            weights[pub_tag] = 1.0
            meta_tags.add(pub_tag)

        # Reconstruct tag string for recommender functions and tfidf
        g['tags'] = " ".join(tags)
        g['weight'] = weight
        all_tags_list.append(g['tags'])
        game_weights.append({"weight": weight, "tag_weights": weights})

    # Backlog prep
    for g in backlog:
        game_data = get_game_data(g['appid'])
        t = game_data.get('tags', '')
        g['steam_score'] = game_data.get('steam_score', 5.0)

        tags, weights = extract_tags_and_weights(t)

        if g['difficulty'] and g['difficulty'] != 'Easy':
            diff_tag = str(g['difficulty']).replace(' ', '_').lower()
            tags.append(diff_tag)
            weights[diff_tag] = 1.0

        # Add metadata as tags with max weight
        if game_data.get('developer'):
            dev_tag = game_data['developer'].replace(' ', '_').lower()
            tags.append(dev_tag)
            weights[dev_tag] = 1.0
            meta_tags.add(dev_tag)
        if game_data.get('publisher') and game_data.get('publisher') != game_data.get('developer'):
            pub_tag = game_data['publisher'].replace(' ', '_').lower()
            tags.append(pub_tag)
            weights[pub_tag] = 1.0
            meta_tags.add(pub_tag)

        # Apply temporary rating if active
        if g.get('temp_rating') and g.get('temp_rating_until', 0) > now:
            g['rating'] = g['temp_rating']

        priority_boost = 1.25 if (float(g['rating']) > 0 and not g['finished']) else 1.0

        g['tags'] = " ".join(tags)
        all_tags_list.append(g['tags'])
        game_weights.append({"priority_boost": priority_boost, "tag_weights": weights})

    df_backlog = pd.DataFrame(backlog)

    # Fit the vectorizer to get the vocabulary
    vectorizer.fit(all_tags_list)
    vocab = vectorizer.vocabulary_

    # Build the custom TF-IDF matrix by multiplying the default TF-IDF values by our custom weights
    tfidf = vectorizer.transform(all_tags_list).tolil()

    for i, data in enumerate(game_weights):
        for tag, weight in data["tag_weights"].items():
            if tag in vocab:
                j = vocab[tag]
                tfidf[i, j] *= weight

    tfidf = tfidf.tocsr()

    rated_start_idx = 0

    # User vector
    user_vec = np.zeros((1, tfidf.shape[1]))
    for i, g in enumerate(rated_db_games):
        if g['tags']:
            game_vec = tfidf[i].toarray()
            user_vec += (game_vec * game_weights[i]["weight"])

    # Matching
    backlog_vecs = tfidf[len(rated_db_games):]
    df_backlog['match_score'] = cosine_similarity(backlog_vecs, user_vec).flatten() * 80

    for i in range(len(df_backlog)):
        df_backlog.loc[i, 'match_score'] *= game_weights[len(rated_db_games) + i]["priority_boost"]

    # Add rating bonus
    df_backlog['match_score'] += df_backlog['rating'].astype(float) * 2.0
    df_backlog['match_score'] = np.clip(df_backlog['match_score'], 0, 100)

    if df_backlog.empty:
        conn.close()
        return jsonify(
            {"results_html": "<h2>No matches found. Try rating more games!</h2>", "carousel_html": get_carousel_html()})

    # Clustering
    num_clusters = min(NUM_CATEGORIES, len(df_backlog) // 2) or 1
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_backlog['cluster'] = kmeans.fit_predict(backlog_vecs)

    terms = vectorizer.get_feature_names_out()
    names = {}
    for i in range(num_clusters):
        top_indices = kmeans.cluster_centers_.argsort()[:, ::-1][i, :]

        selected_terms = []
        for idx in top_indices:
            term = terms[idx]
            if term not in meta_tags and not any(char.isdigit() for char in term):
                selected_terms.append(term)
            if len(selected_terms) >= 2:
                break

        if selected_terms:
            names[i] = " & ".join([t.replace('_', ' ') for t in selected_terms]).title()
        else:
            names[i] = "Miscellaneous"

    df_backlog['category'] = df_backlog['cluster'].map(names)

    # Sort after assigning clusters so predictions align correctly with backlog_vecs
    df_backlog = df_backlog.sort_values(by='match_score', ascending=False)

    # Build HTML
    # 1. Persistent categories first
    persistent_html, shown_app_ids = build_persistent_sections(df_backlog, rated_db_games, vectorizer, tfidf,
                                                               rated_start_idx)

    # 2. Dynamic cluster columns
    res_html = persistent_html
    for cat, group in df_backlog.groupby('category'):
        filtered = group[~group['appid'].isin(shown_app_ids)]
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
