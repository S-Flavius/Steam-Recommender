import os
import json
import pytest

# Set testing environment variable early
os.environ["TESTING"] = "true"

import database
from sqlmodel import Session, select, create_engine, SQLModel
from database import set_metadata, get_metadata, init_db, Game, Metadata
import app
import threading
import time

# Use a separate test database
TEST_DB = "test_games.db"

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Set testing environment variable early
    os.environ["TESTING"] = "true"
    
    # Override the engine in database module
    test_engine = create_engine(f"sqlite:///{TEST_DB}", connect_args={"check_same_thread": False})
    database.engine = test_engine
    # Also need to make sure app uses this engine if it's imported it
    import app as app_mod
    app_mod.engine = test_engine
    
    SQLModel.metadata.create_all(test_engine)
    yield
    # On Windows, we often can't delete the DB file if connections are still open.
    # We'll just leave it or try to dispose the engine first.
    test_engine.dispose()

@pytest.fixture
def client():
    app.app.config['TESTING'] = True
    # Ensure fresh state for each test
    with Session(database.engine) as session:
        for table in reversed(SQLModel.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
    
    init_db() # Re-initialize defaults
    
    with app.app.test_client() as client:
        yield client

def test_update_game_rate(client):
    # Add a dummy game
    with Session(database.engine) as session:
        game = Game(appid=123, name="Test Game", rating=0)
        session.add(game)
        session.commit()
    
    response = client.post('/update_game', 
                           data=json.dumps({'appid': 123, 'action': 'rate', 'value': 8}),
                           content_type='application/json')
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert data['unrated_count'] == 0
    
    with Session(database.engine) as session:
        game = session.get(Game, 123)
        assert game.rating == 8

def test_update_game_finish(client):
    # Add a dummy game
    with Session(database.engine) as session:
        game = Game(appid=124, name="Finish Me", rating=0, finished=False)
        session.add(game)
        session.commit()
    
    response = client.post('/update_game', 
                           data=json.dumps({'appid': 124, 'action': 'finish', 'value': 9}),
                           content_type='application/json')
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    
    with Session(database.engine) as session:
        game = session.get(Game, 124)
        assert game.finished is True
        assert game.rating == 9

def test_update_game_ignore(client):
    with Session(database.engine) as session:
        game = Game(appid=125, name="Ignore Me", rating=0)
        session.add(game)
        session.commit()
    
    response = client.post('/update_game', 
                           data=json.dumps({'appid': 125, 'action': 'ignore'}),
                           content_type='application/json')
    
    assert response.status_code == 200
    with Session(database.engine) as session:
        game = session.get(Game, 125)
        assert game.ignore_until > time.time()

def test_settings_get(client):
    response = client.get('/settings')
    assert response.status_code == 200
    data = response.get_json()
    assert 'STEAM_ID' in data
    assert 'NUM_CATEGORIES' in data

def test_settings_post(client):
    response = client.post('/settings', 
                           data=json.dumps({'NUM_CATEGORIES': 12, 'STEAM_ID': 'test_id'}),
                           content_type='application/json')
    assert response.status_code == 200
    assert get_metadata('NUM_CATEGORIES') == '12'
    assert get_metadata('STEAM_ID') == 'test_id'

def test_library_data(client):
    with Session(database.engine) as session:
        game = Game(appid=126, name="Library Game", rating=5)
        session.add(game)
        session.commit()
    
    response = client.get('/library_data')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) >= 1
    assert data[0]['appid'] == 126
    assert data[0]['rating'] == 5

def test_index_cache_miss(client, mocker):
    # Ensure no cache
    set_metadata('CACHED_RESULTS', '')
    
    # Mock background train to avoid heavy computation
    mock_train = mocker.patch('app._background_train')
    
    response = client.get('/')
    assert response.status_code == 200
    
    # Check that it triggered background training
    # Since it's in a thread, we might need a small delay or check if it was called
    # But wait, it's called inside index() which starts a thread. 
    # Mocking the Thread target might be better or just mocking _background_train
    
    # Wait a bit for the thread to potentially start
    time.sleep(0.5)
    mock_train.assert_called_once()

def test_index_cache_hit(client):
    # Set fake cache
    fake_data = {
        "results_html": "<div>Recommended</div>",
        "carousel_html": "<div>Carousel</div>",
        "unrated_count": 5
    }
    set_metadata('CACHED_RESULTS', json.dumps(fake_data))
    
    response = client.get('/')
    assert response.status_code == 200
    data = response.get_data(as_text=True)
    assert "Recommended" in data
    assert "Carousel" in data
    # The new Alpine.js logic uses x-text, but we still expect the initial value in the span if rendered or just check for the count
    assert "5 left to rate" in data

def test_library_route(client):
    response = client.get('/library')
    assert response.status_code == 200
    assert "Library" in response.get_data(as_text=True)

def test_stats_route(client):
    response = client.get('/stats')
    assert response.status_code == 200
    assert "Stats" in response.get_data(as_text=True)

def test_background_train_populates_cache(mocker):
    # Mock build_recommendations_html and other helpers to avoid DB/ML complexity
    mocker.patch('app.build_recommendations_html', return_value="<html>Results</html>")
    mocker.patch('app.get_unrated_count', return_value=10)
    mocker.patch('app.get_carousel_html', return_value="<html>Carousel</html>")
    
    stop_event = threading.Event()
    app._background_train(show_finished=0, stop_event=stop_event)
    
    # Check if cache was set in metadata
    cached = get_metadata('CACHED_RESULTS')
    assert cached is not None
    data = json.loads(cached)
    assert data['results_html'] == "<html>Results</html>"
    assert data['unrated_count'] == 10
    assert data['carousel_html'] == "<html>Carousel</html>"

def test_recommend_with_null_key(client):
    # This reproduces the ValueError: invalid literal for int() with base 10: 'null'
    # when the frontend sends "null" as a key in the ratings dictionary.
    response = client.post('/recommend',
                           data=json.dumps({'null': 5}),
                           content_type='application/json')
    
    # It should ideally handle it gracefully or ignore it, but currently it crashes (500)
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True

def test_settings_updates_db_file(client, mocker):
    # Test that updating STEAM_ID changes the database file used
    mocker.patch('database.get_engine', return_value=create_engine("sqlite:///:memory:"))
    
    # 1. Update STEAM_ID
    new_id = "new_steam_id"
    response = client.post('/settings', 
                           data=json.dumps({'STEAM_ID': new_id}),
                           content_type='application/json')
    assert response.status_code == 200
    assert os.environ.get("STEAM_ID") == new_id
    
    # 2. Check that database.engine was updated (mocked, but we can check if it was called)
    # Actually, we can check the effect of database.get_engine being called
    import database
    assert database.get_engine.called

def test_index_populates_carousel_even_if_cache_empty(client):
    # Ensure database is clean or at least has some games for this test
    # We can use the session setup
    from database import engine, Game, set_metadata
    from sqlmodel import Session
    
    with Session(engine) as session:
        # Add a dummy game if none exist
        if session.query(Game).count() == 0:
            session.add(Game(appid=123, name="Test Game", rating=0, ignored=False))
            session.commit()
    
    # Clear cache
    set_metadata('CACHED_RESULTS', '')
    
    response = client.get('/')
    assert response.status_code == 200
    html = response.data.decode()
    # Carousel should be populated
    assert 'rate-card' in html
    assert 'Test Game' in html

def test_recommender_handles_empty_candidates(client):
    # Test that the recommender doesn't crash if no candidates are found
    from database import engine, Game, set_metadata
    from sqlmodel import Session, select, func
    import json
    
    with Session(engine) as session:
        # 1. Ensure at least one rated game
        game = session.get(Game, 123)
        if not game:
            session.add(Game(appid=123, name="Rated Game", rating=5, ignored=False))
        else:
            game.rating = 5
            game.ignored = False
            session.add(game)
        
        # 2. Ensure NO candidates (e.g. all ignored or no tags)
        # We'll just ignore all other games for this test
        statement = select(Game).where(Game.appid != 123)
        others = session.exec(statement).all()
        for g in others:
            g.ignored = True
            session.add(g)
        session.commit()

    # Clear cache
    set_metadata('CACHED_RESULTS', '')
    
    # Trigger index which should start training
    response = client.get('/')
    assert response.status_code == 200
    
    # Wait a bit for background thread or call it manually
    from app import _background_train
    import threading
    stop_event = threading.Event()
    _background_train(0, stop_event)
    
    # Check cache
    cached = get_metadata('CACHED_RESULTS')
    assert cached != ''
    data = json.loads(cached)
    assert 'No candidate games found yet' in data['results_html']
