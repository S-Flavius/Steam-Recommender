import os
import threading
import time
import json

from flask import Flask, request, render_template, jsonify, make_response

from config import (NUM_CATEGORIES, GAMES_PER_CATEGORY, MIN_PLAYTIME, CAROUSEL_SIZE,
                    IGNORE_DURATION_DAYS, UP_NEXT_DURATION_DAYS)
from database import engine, Game, init_db, cleanup_expired_temp_ratings, get_metadata, set_metadata, get_db
from sqlmodel import Session, select, func
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
        
        # Trigger background training to update cache after sync
        show_finished = int(get_metadata('SHOW_FINISHED', '0'))
        _background_train(show_finished, _stop_event)
    finally:
        with _sync_lock:
            _sync_in_progress = False


@app.route('/')
def index():
    """Main page - sync library and show rating carousel."""
    global _train_thread, _stop_event
    # Ensure DB and metadata are initialized
    # If STEAM_ID is in environment but not in DB, init_db will sync it.
    # If it's in DB but different from environment, we might have a problem.
    # Let's check if we should update environment from DB.
    db_sid = get_metadata('STEAM_ID')
    if db_sid and db_sid != os.getenv("STEAM_ID"):
        os.environ["STEAM_ID"] = db_sid
        import database
        database.engine = database.get_engine()

    init_db()
    
    # Start background sync
    if not _sync_in_progress:
        # Check if we synced recently (within 24 hours) OR if we have no games
        last_sync = get_metadata('LAST_SYNC_TIME')
        now = time.time()
        
        has_games = False
        with Session(engine) as session:
            has_games = session.exec(select(func.count(Game.appid))).one() > 0

        if not last_sync or (now - float(last_sync)) > 86400 or not has_games:
            threading.Thread(target=_background_sync, daemon=True).start()
            if not last_sync or (now - float(last_sync)) > 86400:
                set_metadata('LAST_SYNC_TIME', str(now))

    cleanup_expired_temp_ratings()
    
    with Session(engine) as session:
        # If no games in DB at all, carousel will be empty anyway
        # but we should at least check if we can populate it.
        
        cached = get_metadata('CACHED_RESULTS')
        if cached:
            try:
                data = json.loads(cached)
                carousel_html = data.get('carousel_html', '')
                results_html = data.get('results_html', '')
                unrated_count = data.get('unrated_count', 0)
            except:
                carousel_html = ""
                results_html = ""
                unrated_count = 0
        else:
            carousel_html = ""
            results_html = ""
            unrated_count = 0
        
        # If cache is missing, we should ensure a training is triggered
        if not cached and not (_train_thread and _train_thread.is_alive()):
            show_finished = int(get_metadata('SHOW_FINISHED', '0'))
            _stop_event = threading.Event()
            _train_thread = threading.Thread(
                target=_background_train, 
                args=(show_finished, _stop_event), 
                daemon=True
            )
            _train_thread.start()
        
        # If carousel is missing (e.g. freshly cleared cache), 
        # try to get it directly from DB so the user sees SOMETHING while training.
        if not carousel_html:
             carousel_html = get_carousel_html(session)
             unrated_count = get_unrated_count(session)

    return render_template('home.html', 
                           carousel_html=carousel_html, 
                           unrated_count=unrated_count,
                           results_html=results_html)


@app.route('/library')
def library():
    """Library page."""
    return render_template('library.html')


@app.route('/stats')
def stats():
    """Stats page."""
    return render_template('stats.html')


@app.route('/update_game', methods=['POST'])
def update_game():
    """Handle game state updates (rate, ignore, ban, finish, etc.)."""
    data = request.json
    appid = data['appid']
    action = data['action']
    now = int(time.time())

    with Session(engine) as session:
        game = session.get(Game, appid)
        if not game:
            return jsonify({"success": False, "error": "Game not found"}), 404

        if action == 'ignore':
            ignore_days = int(get_metadata('IGNORE_DURATION_DAYS', IGNORE_DURATION_DAYS))
            game.ignore_until = now + (ignore_days * 24 * 60 * 60)

        elif action == 'ban':
            game.ignored = True
            game.ignore_until = 0

        elif action == 'unban':
            game.ignored = False

        elif action == 'finish':
            game.finished = True
            game.rating = data.get('value', 0)

        elif action == 'unfinish':
            game.finished = False

        elif action == 'rate':
            game.rating = data.get('value', 0)
            game.temp_rating = None
            game.temp_rating_until = None

        elif action == 'up_next':
            is_active = game.temp_rating is not None and (game.temp_rating_until or 0) > now
            if is_active:
                game.temp_rating = None
                game.temp_rating_until = None
            else:
                up_next_days = int(get_metadata('UP_NEXT_DURATION_DAYS', UP_NEXT_DURATION_DAYS))
                game.temp_rating = 10
                game.temp_rating_until = now + (up_next_days * 24 * 60 * 60)
                game.ignore_until = 0
        
        session.add(game)
        session.commit()
        unrated_count = get_unrated_count(session) # Need to update ui_helpers too
    
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
            if key == 'STEAM_ID':
                os.environ['STEAM_ID'] = str(data[key])
                # Recreate the engine and clear cache for the new database
                import database
                database.engine = database.get_engine()
                set_metadata('CACHED_RESULTS', '')
                set_metadata('LAST_SYNC_TIME', '') # Force re-sync for new ID
                
                # Trigger immediate sync and training in background
                threading.Thread(target=_background_sync, daemon=True).start()
    
    return jsonify({"success": True})


def _background_train(show_finished, stop_event):
    """Run recommendation generation in the background."""
    global _last_train_results

    try:
        with Session(engine) as session:
            res_html = build_recommendations_html(session, show_finished=bool(show_finished), stop_event=stop_event)
        
            if stop_event.is_set() or res_html is None:
                return

            unrated_count = get_unrated_count(session)
            carousel_html = get_carousel_html(session)

        _last_train_results = {
            "results_html": res_html,
            "carousel_html": carousel_html,
            "unrated_count": unrated_count
        }
        
        # Cache results for first page load
        # Use a temporary connection for set_metadata to avoid potential threading issues if needed,
        # but set_metadata already opens its own connection.
        set_metadata('CACHED_RESULTS', json.dumps(_last_train_results))
        
        # Also update global _last_train_results if we want it to be immediately available for /training_status
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Training error: {e}")


@app.route('/recommend', methods=['POST'])
def recommend():
    """Trigger background re-training, canceling any existing one."""
    global _train_thread, _stop_event, _last_train_results
    
    session_ratings = request.json
    
    # Persist ratings IMMEDIATELY so they aren't lost if training is cancelled/restarted
    if session_ratings:
        with Session(engine) as session:
            for aid, score in session_ratings.items():
                try:
                    appid_int = int(aid)
                except (ValueError, TypeError):
                    continue
                game = session.get(Game, appid_int)
                if game:
                    game.rating = int(score)
                    game.temp_rating = None
                    game.temp_rating_until = None
                    session.add(game)
            session.commit()

    with _train_lock:
        _stop_event.set()
        _stop_event = threading.Event()
        _last_train_results = None
        
        # Clear CACHED_RESULTS so that index() can show fresh carousel immediately if needed
        # Or at least indicates training is fresh
        set_metadata('CACHED_RESULTS', '')
        
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


@app.route('/sync_steam', methods=['POST'])
def sync_steam():
    """Trigger manual Steam library sync."""
    threading.Thread(target=_background_sync, daemon=True).start()
    return jsonify({"success": True})


@app.route('/library_data')
def library_data():
    """Get all games for the library view."""
    with Session(engine) as session:
        statement = select(Game).where(Game.ignored == False)
        games = session.exec(statement).all()
        return jsonify([g.model_dump() for g in games])


@app.route('/stats_data')
def stats_data():
    """Get stats for the stats view."""
    with Session(engine) as session:
        # Basic counts
        total_games = session.exec(select(func.count(Game.appid)).where(Game.ignored == False)).one()
        finished_count = session.exec(select(func.count(Game.appid)).where(Game.finished == True, Game.ignored == False)).one()
        rated_count = session.exec(select(func.count(Game.appid)).where(Game.rating > 0, Game.ignored == False)).one()
        total_playtime = session.exec(select(func.sum(Game.playtime))).one() or 0
        avg_rating = session.exec(select(func.avg(Game.rating)).where(Game.rating > 0, Game.ignored == False)).one() or 0
        
        # Rating distribution (1-10)
        rating_dist = [0] * 10
        # Use SQLModel/SQLAlchemy grouping
        from sqlmodel import case, cast, Integer
        statement = select(
            case(
                (Game.rating >= 10, 10),
                else_=cast(Game.rating, Integer)
            ).label("r"),
            func.count(Game.appid).label("count")
        ).where(Game.ignored == False, Game.rating > 0).group_by("r")
        
        results = session.exec(statement).all()
        for r_val, count in results:
            if 1 <= r_val <= 10:
                rating_dist[r_val - 1] = count

        # Top genres
        from recommender import extract_tags
        genre_counts = {}
        tags_rows = session.exec(select(Game.tags).where(Game.tags != None, Game.ignored == False)).all()
        for tag_str in tags_rows:
            tags = extract_tags(tag_str)
            for t in tags:
                genre_counts[t] = genre_counts.get(t, 0) + 1
        
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        max_genre = max([c for g, c in sorted_genres] + [1])
        top_genres = [{
            'name': g.replace('_', ' ').title(),
            'count': c,
            'percent': (c / max_genre) * 100
        } for g, c in sorted_genres]

    return jsonify({
        'total_games': total_games,
        'finished_count': finished_count,
        'rated_count': rated_count,
        'completion_rate': round((finished_count / total_games * 100), 1) if total_games > 0 else 0,
        'total_playtime': round(total_playtime / 60, 1),
        'avg_rating': round(avg_rating, 1),
        'rating_dist': rating_dist,
        'top_genres': top_genres
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
