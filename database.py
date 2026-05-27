import sqlite3
import time

from config import DB_FILE


def get_db():
    """Get a database connection with Row factory enabled."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
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
            achievements_completed BOOLEAN DEFAULT 0,
            ignore_until           INTEGER DEFAULT 0,
            temp_rating            INTEGER DEFAULT NULL,
            temp_rating_until      INTEGER DEFAULT NULL,
            tags_updated           INTEGER DEFAULT NULL,
            achievements_total     INTEGER DEFAULT 0,
            achievements_unlocked  INTEGER DEFAULT 0,
            developer              TEXT    DEFAULT NULL,
            publisher              TEXT    DEFAULT NULL
        )
    ''')

    # Normalized tables
    c.execute('CREATE TABLE IF NOT EXISTS developers (id INTEGER PRIMARY KEY, name TEXT UNIQUE)')
    c.execute('CREATE TABLE IF NOT EXISTS publishers (id INTEGER PRIMARY KEY, name TEXT UNIQUE)')
    c.execute('CREATE TABLE IF NOT EXISTS tags (id INTEGER PRIMARY KEY, name TEXT UNIQUE)')

    c.execute('''
        CREATE TABLE IF NOT EXISTS game_developers (
            appid INTEGER,
            developer_id INTEGER,
            PRIMARY KEY (appid, developer_id),
            FOREIGN KEY (appid) REFERENCES games (appid),
            FOREIGN KEY (developer_id) REFERENCES developers (id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS game_publishers (
            appid INTEGER,
            publisher_id INTEGER,
            PRIMARY KEY (appid, publisher_id),
            FOREIGN KEY (appid) REFERENCES games (appid),
            FOREIGN KEY (publisher_id) REFERENCES publishers (id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS game_tags (
            appid INTEGER,
            tag_id INTEGER,
            count INTEGER,
            PRIMARY KEY (appid, tag_id),
            FOREIGN KEY (appid) REFERENCES games (appid),
            FOREIGN KEY (tag_id) REFERENCES tags (id)
        )
    ''')

    # Add columns if they don't exist (for backwards compatibility)
    for column, definition in [
        ('finished', 'BOOLEAN DEFAULT 0'),
        ('ignore_until', 'INTEGER DEFAULT 0'),
        ('temp_rating', 'INTEGER DEFAULT NULL'),
        ('temp_rating_until', 'INTEGER DEFAULT NULL'),
        ('tags_updated', 'INTEGER DEFAULT NULL'),
        ('achievements_total', 'INTEGER DEFAULT 0'),
        ('achievements_unlocked', 'INTEGER DEFAULT 0'),
        ('developer', 'TEXT DEFAULT NULL'),
        ('publisher', 'TEXT DEFAULT NULL'),
    ]:
        try:
            c.execute(f'ALTER TABLE games ADD COLUMN {column} {definition}')
        except sqlite3.OperationalError:
            pass  # Column already exists

    c.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
    conn.commit()
    conn.close()


def cleanup_expired_temp_ratings():
    """Remove expired temporary ratings."""
    conn = get_db()
    now = int(time.time())
    conn.execute(
        "UPDATE games SET temp_rating = NULL, temp_rating_until = NULL WHERE temp_rating_until < ?",
        (now,)
    )
    conn.commit()
    conn.close()
