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
            achievements_completed BOOLEAN DEFAULT 0
        )
    ''')

    # Add columns if they don't exist (for backwards compatibility)
    for column, definition in [
        ('finished', 'BOOLEAN DEFAULT 0'),
        ('ignore_until', 'INTEGER DEFAULT 0'),
        ('temp_rating', 'INTEGER DEFAULT NULL'),
        ('temp_rating_until', 'INTEGER DEFAULT NULL'),
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
