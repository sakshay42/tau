
import json
import sqlite3


class SQLiteStore:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                published_at TEXT,
                ingested_at TEXT NOT NULL,
                url TEXT,
                metadata TEXT
            )
            """
        )

        self.conn.commit()

    def insert(self, document):
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO documents (
                id,
                title,
                text,
                source,
                published_at,
                ingested_at,
                url,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.id,
                document.title,
                document.text,
                document.source,
                document.published_at.isoformat()
                if document.published_at else None,
                document.ingested_at.isoformat(),
                document.url,
                json.dumps(document.metadata),
            ),
        )

        self.conn.commit()

        return cursor.rowcount > 0
