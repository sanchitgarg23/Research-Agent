"""Database and session persistence configuration for Chainlit."""

import sqlite3
import logging
from pathlib import Path
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "chainlit.db"


def init_database():
    """Initialize SQLite tables for Chainlit chat and thread persistence."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        identifier TEXT NOT NULL UNIQUE,
        createdAt TEXT NOT NULL,
        metadata TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS threads (
        id TEXT PRIMARY KEY,
        createdAt TEXT,
        name TEXT,
        userId TEXT,
        userIdentifier TEXT,
        tags TEXT,
        metadata TEXT
    );
    CREATE TABLE IF NOT EXISTS steps (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        threadId TEXT NOT NULL,
        parentId TEXT,
        streaming BOOLEAN,
        waitForAnswer BOOLEAN,
        isError BOOLEAN,
        metadata TEXT,
        tags TEXT,
        input TEXT,
        output TEXT,
        createdAt TEXT,
        start TEXT,
        end TEXT,
        generation TEXT,
        showInput TEXT,
        language TEXT
    );
    CREATE TABLE IF NOT EXISTS feedbacks (
        id TEXT PRIMARY KEY,
        forId TEXT NOT NULL,
        value INTEGER NOT NULL,
        comment TEXT
    );
    CREATE TABLE IF NOT EXISTS elements (
        id TEXT PRIMARY KEY,
        threadId TEXT,
        type TEXT,
        url TEXT,
        chainlitKey TEXT,
        name TEXT NOT NULL,
        display TEXT,
        size TEXT,
        language TEXT,
        autoPlay BOOLEAN,
        playerConfig TEXT,
        page INTEGER,
        props TEXT,
        forId TEXT,
        mime TEXT,
        objectKey TEXT
    );
    """)
    conn.commit()
    conn.close()
    logger.info(f"Chainlit SQLite persistence database initialized at: {DB_PATH}")


def get_data_layer():
    """Get the configured SQLAlchemyDataLayer instance."""
    init_database()
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{DB_PATH}")

