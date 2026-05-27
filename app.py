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

# Background training tracking
_train_lock = threading.Lock()
_train_thread = None
_stop_event = threading.Event()
_last_train_results = None


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
        conn.commit()
        # No more manual triggers here, frontend will call /recommend

    elif action == 'unfinish':
        conn.execute("""
                     UPDATE games
                     SET finished = 0
                     WHERE appid = ?
                     """, (appid,))
        conn.commit()
        # No more manual triggers here, frontend will call /recommend

    elif action == 'rate':
        rating = data.get('value', 0)
        conn.execute("UPDATE games SET rating = ?, temp_rating = NULL, temp_rating_until = NULL WHERE appid = ?",
                     (rating, appid))

    elif action == 'up_next':
        # Toggle Up Next: if already active, clear it; otherwise set it.
        row = conn.execute("SELECT temp_rating, temp_rating_until FROM games WHERE appid = ?", (appid,)).fetchone()
        is_active = row and row['temp_rating'] is not None and row['temp_rating_until'] > now
        
        if is_active:
            conn.execute("UPDATE games SET temp_rating = NULL, temp_rating_until = NULL WHERE appid = ?", (appid,))
        else:
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


def _background_train(show_finished, stop_event):
    """Run recommendation generation in the background."""
    global _last_train_results

    try:
        conn = get_db()
        res_html = build_recommendations_html(conn, show_finished=bool(show_finished), stop_event=stop_event)
        
        if stop_event.is_set() or res_html is None:
            conn.close()
            return

        unrated_count = get_unrated_count(conn)
        carousel_html = get_carousel_html(conn)
        conn.close()

        _last_train_results = {
            "results_html": res_html,
            "carousel_html": carousel_html,
            "unrated_count": unrated_count
        }
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        pass


@app.route('/recommend', methods=['POST'])
def recommend():
    """Trigger background re-training, canceling any existing one."""
    global _train_thread, _stop_event, _last_train_results
    
    session_ratings = request.json
    
    # Persist ratings IMMEDIATELY so they aren't lost if training is cancelled/restarted
    if session_ratings:
        conn = get_db()
        for aid, score in session_ratings.items():
            # Only update if the rating has actually changed or if it's not a temp-rated game
            # To avoid clearing temp_rating accidentally
            conn.execute("""
                UPDATE games 
                SET rating = ?, 
                    temp_rating = CASE WHEN rating = ? THEN temp_rating ELSE NULL END,
                    temp_rating_until = CASE WHEN rating = ? THEN temp_rating_until ELSE NULL END
                WHERE appid = ?
            """, (score, score, score, aid))
        conn.commit()
        conn.close()

    with _train_lock:
        _stop_event.set()
        _stop_event = threading.Event()
        _last_train_results = None
        
        show_finished = int(get_metadata('SHOW_FINISHED', '0'))
        
        _train_thread = threading.Thread(
            target=_background_train, 
            args=(show_finished, _stop_event), 
            daemon=True
        )
        _train_thread.start()
    
    return jsonify({"success": True, "message": "Training started"})


@app.route('/training_status')
def training_status():
    """Check background training status."""
    with _train_lock:
        if _train_thread and _train_thread.is_alive() and not _stop_event.is_set():
            return jsonify({"status": "in_progress"})
        
        if _last_train_results:
            return jsonify({"status": "complete", "data": _last_train_results})
        
        return jsonify({"status": "idle"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
