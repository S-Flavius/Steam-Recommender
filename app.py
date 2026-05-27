import os
import threading
import time

from flask import Flask, request, render_template, jsonify

from config import (NUM_CATEGORIES, GAMES_PER_CATEGORY, MIN_PLAYTIME, CAROUSEL_SIZE,
                    IGNORE_DURATION_DAYS, UP_NEXT_DURATION_DAYS)
from database import get_db, init_db, cleanup_expired_temp_ratings, get_metadata, set_metadata
from recommender import build_recommendations_html
from sync import sync_steam_library, sync_cedb_difficulties, sync_game_tags
from ui_helpers import get_carousel_html, get_unrated_count

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


@app.route('/')
def index():
    """Main page - sync library and show rating carousel."""
    init_db()  # Ensure DB and metadata are initialized
    
    # Start background sync
    threading.Thread(target=_background_sync, daemon=True).start()

    cleanup_expired_temp_ratings()
    conn = get_db()
    unrated_count = get_unrated_count(conn)
    carousel_html = get_carousel_html(conn)
    
    show_finished = int(get_metadata('SHOW_FINISHED', '0'))
    results_html = build_recommendations_html(conn, show_finished=bool(show_finished))
    
    conn.close()
    return render_template('index.html', 
                           carousel_html=carousel_html, 
                           unrated_count=unrated_count,
                           results_html=results_html)


@app.route('/update_game', methods=['POST'])
def update_game():
    """Handle game state updates (rate, ignore, ban, finish, etc.)."""
    data = request.json
    conn = get_db()

    appid = data['appid']
    action = data['action']

    now = int(time.time())

    if action == 'ignore':
        ignore_days = int(get_metadata('IGNORE_DURATION_DAYS', IGNORE_DURATION_DAYS))
        duration = ignore_days * 24 * 60 * 60
        conn.execute("""
                     UPDATE games
                     SET ignore_until = ?
                     WHERE appid = ?
                     """, (now + duration, appid))

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
        # Set temporary rating of 10 that expires in X days
        up_next_days = int(get_metadata('UP_NEXT_DURATION_DAYS', UP_NEXT_DURATION_DAYS))
        duration = up_next_days * 24 * 60 * 60
        conn.execute("""
                     UPDATE games
                     SET temp_rating       = 10,
                         temp_rating_until = ?,
                         ignore_until      = 0
                     WHERE appid = ?
                     """, (now + duration, appid))

    conn.commit()
    unrated_count = get_unrated_count(conn)
    conn.close()
    return jsonify({"success": True, "unrated_count": unrated_count})


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Get or update user configuration."""
    if request.method == 'GET':
        return jsonify({
            'NUM_CATEGORIES': int(get_metadata('NUM_CATEGORIES', NUM_CATEGORIES)),
            'GAMES_PER_CATEGORY': int(get_metadata('GAMES_PER_CATEGORY', GAMES_PER_CATEGORY)),
            'MIN_PLAYTIME': int(get_metadata('MIN_PLAYTIME', MIN_PLAYTIME)),
            'CAROUSEL_SIZE': int(get_metadata('CAROUSEL_SIZE', CAROUSEL_SIZE)),
            'IGNORE_DURATION_DAYS': int(get_metadata('IGNORE_DURATION_DAYS', IGNORE_DURATION_DAYS)),
            'UP_NEXT_DURATION_DAYS': int(get_metadata('UP_NEXT_DURATION_DAYS', UP_NEXT_DURATION_DAYS)),
            'SHOW_FINISHED': int(get_metadata('SHOW_FINISHED', '0')),
            'STEAM_ID': get_metadata('STEAM_ID', os.getenv("STEAM_ID", "")),
            'CEDB_USER_ID': get_metadata('CEDB_USER_ID', os.getenv("CEDB_USER_ID", ""))
        })

    data = request.json
    for key in ['NUM_CATEGORIES', 'GAMES_PER_CATEGORY', 'MIN_PLAYTIME', 'CAROUSEL_SIZE', 
                'IGNORE_DURATION_DAYS', 'UP_NEXT_DURATION_DAYS', 'SHOW_FINISHED', 'STEAM_ID', 'CEDB_USER_ID']:
        if key in data:
            set_metadata(key, data[key])
    
    return jsonify({"success": True})


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

    show_finished = int(get_metadata('SHOW_FINISHED', '0'))
    res_html = build_recommendations_html(conn, show_finished=bool(show_finished))
    
    unrated_count = get_unrated_count(conn)
    carousel_html = get_carousel_html(conn)
    conn.close()
    
    if res_html is None:
        return jsonify({
            "results_html": "<h2>Please rate some games first!</h2>",
            "carousel_html": carousel_html,
            "unrated_count": unrated_count
        })

    return jsonify({"results_html": res_html, "carousel_html": carousel_html, "unrated_count": unrated_count})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
