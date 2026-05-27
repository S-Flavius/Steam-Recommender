from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session, select
from config import DB_FILE
import sqlite3
import time
import os

class Game(SQLModel, table=True):
    __tablename__ = "games"
    appid: int = Field(primary_key=True)
    name: Optional[str] = None
    playtime: int = Field(default=0)
    last_played: int = Field(default=0)
    rating: int = Field(default=0)
    ignored: bool = Field(default=False)
    finished: bool = Field(default=False)
    difficulty: str = Field(default="Easy")
    tags: Optional[str] = None
    steam_score: Optional[float] = None
    achievements_completed: bool = Field(default=False)
    ignore_until: int = Field(default=0)
    temp_rating: Optional[int] = None
    temp_rating_until: Optional[int] = None
    tags_updated: Optional[int] = None
    achievements_total: int = Field(default=0)
    achievements_unlocked: int = Field(default=0)
    developer: Optional[str] = None
    publisher: Optional[str] = None

class Developer(SQLModel, table=True):
    __tablename__ = "developers"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)

class Publisher(SQLModel, table=True):
    __tablename__ = "publishers"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)

class Tag(SQLModel, table=True):
    __tablename__ = "tags"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)

class GameDeveloper(SQLModel, table=True):
    __tablename__ = "game_developers"
    appid: int = Field(foreign_key="games.appid", primary_key=True)
    developer_id: int = Field(foreign_key="developers.id", primary_key=True)

class GamePublisher(SQLModel, table=True):
    __tablename__ = "game_publishers"
    appid: int = Field(foreign_key="games.appid", primary_key=True)
    publisher_id: int = Field(foreign_key="publishers.id", primary_key=True)

class GameTag(SQLModel, table=True):
    __tablename__ = "game_tags"
    appid: int = Field(foreign_key="games.appid", primary_key=True)
    tag_id: int = Field(foreign_key="tags.id", primary_key=True)
    count: int

class Metadata(SQLModel, table=True):
    __tablename__ = "metadata"
    key: str = Field(primary_key=True)
    value: str

def get_engine():
    """Get the database engine, ensuring it matches the current STEAM_ID."""
    from config import get_db_file
    db_file = get_db_file()
    
    # Try to get STEAM_ID from an existing database if possible to keep it consistent
    # But wait, config.get_db_file() already checks os.getenv("STEAM_ID").
    # If the user changed it via settings, we need to make sure it's reflected.
    
    sqlite_url = f"sqlite:///{db_file}"
    return create_engine(sqlite_url, connect_args={"check_same_thread": False})

engine = get_engine()

def get_db():
    """Legacy helper for raw SQL if needed, but should prefer Session(engine)."""
    from config import get_db_file
    conn = sqlite3.connect(get_db_file())
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    if os.getenv("TESTING") == "true" and "test" not in str(engine.url).lower():
        # During module import, engine might be default. 
        # We don't want to crash on import, but we want to prevent creation.
        print(f"DEBUG: Skipping init_db for production engine {engine.url} in testing mode")
        return
        
    SQLModel.metadata.create_all(engine)
    
    # Initialize default settings
    from config import (NUM_CATEGORIES, GAMES_PER_CATEGORY, MIN_PLAYTIME, CAROUSEL_SIZE, 
                        IGNORE_DURATION_DAYS, UP_NEXT_DURATION_DAYS)
    defaults = {
        'NUM_CATEGORIES': str(NUM_CATEGORIES),
        'GAMES_PER_CATEGORY': str(GAMES_PER_CATEGORY),
        'MIN_PLAYTIME': str(MIN_PLAYTIME),
        'CAROUSEL_SIZE': str(CAROUSEL_SIZE),
        'IGNORE_DURATION_DAYS': str(IGNORE_DURATION_DAYS),
        'UP_NEXT_DURATION_DAYS': str(UP_NEXT_DURATION_DAYS),
        'SHOW_FINISHED': '0'
    }
    
    with Session(engine) as session:
        for key, value in defaults.items():
            existing = session.get(Metadata, key)
            if not existing:
                session.add(Metadata(key=key, value=value))
        
        # Sync STEAM_ID and CEDB_USER_ID from env if not already in DB
        from config import STEAM_ID, CEDB_USER_ID as CEDB_ID
        if STEAM_ID and not session.get(Metadata, 'STEAM_ID'):
            session.add(Metadata(key='STEAM_ID', value=STEAM_ID))
        if CEDB_ID and not session.get(Metadata, 'CEDB_USER_ID'):
            session.add(Metadata(key='CEDB_USER_ID', value=CEDB_ID))
            
        session.commit()


def get_metadata(key, default=None):
    """Get a metadata value from the database."""
    with Session(engine) as session:
        metadata = session.get(Metadata, key)
        return metadata.value if metadata else default


def set_metadata(key, value):
    """Set a metadata value in the database."""
    with Session(engine) as session:
        metadata = session.get(Metadata, key)
        if metadata:
            metadata.value = str(value)
        else:
            metadata = Metadata(key=key, value=str(value))
        session.add(metadata)
        session.commit()


def cleanup_expired_temp_ratings():
    """Remove expired temporary ratings."""
    with Session(engine) as session:
        now = int(time.time())
        statement = select(Game).where(Game.temp_rating_until < now)
        results = session.exec(statement)
        for game in results:
            game.temp_rating = None
            game.temp_rating_until = None
            session.add(game)
        session.commit()
