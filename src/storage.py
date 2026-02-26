import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
from pathlib import Path
from typing import Optional
import config


class ChunkData:
    def __init__(self, chunk_id: str, text: str, source: str, page: int):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.page = page


class SummaryStorage:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or config.DB_PATH
        self._init_db()
        
    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT,
                page INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                chunk_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                source TEXT,
                page INTEGER
            )
        """)
        conn.commit()
        conn.close()
        
    def add_chunk(self, chunk_id: str, text: str, source: str = None, page: int = None):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, text, source, page) VALUES (?, ?, ?, ?)",
            (chunk_id, text, source, page)
        )
        conn.commit()
        conn.close()
        
    def add_summary(self, chunk_id: str, summary: str, source: str = None, page: int = None):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO summaries (chunk_id, summary, source, page) VALUES (?, ?, ?, ?)",
            (chunk_id, summary, source, page)
        )
        conn.commit()
        conn.close()
        
    def get_chunk(self, chunk_id: str) -> Optional[ChunkData]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, text, source, page FROM chunks WHERE chunk_id = ?", (chunk_id,))
        result = cursor.fetchone()
        conn.close()
        return ChunkData(result[0], result[1], result[2], result[3]) if result else None
        
    def get_summary(self, chunk_id: str) -> Optional[str]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT summary FROM summaries WHERE chunk_id = ?", (chunk_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
        
    def get_all_summaries(self) -> list:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, summary, source, page FROM summaries")
        results = cursor.fetchall()
        conn.close()
        return [SummaryData(r[0], r[1], r[2], r[3]) for r in results]
    
    def exists(self, chunk_id: str) -> bool:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM summaries WHERE chunk_id = ?", (chunk_id,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    def chunk_exists(self, chunk_id: str) -> bool:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    def get_all_chunk_ids(self) -> set:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id FROM chunks")
        results = cursor.fetchall()
        conn.close()
        return {r[0] for r in results}
    
    def count(self) -> int:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM summaries")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_chunk_for_summary(self, chunk_id: str):
        return self.get_chunk(chunk_id)
    
    def clear(self):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM summaries")
        cursor.execute("DELETE FROM chunks")
        conn.commit()
        conn.close()


class SummaryData:
    def __init__(self, chunk_id: str, summary: str, source: str, page: int):
        self.chunk_id = chunk_id
        self.summary = summary
        self.source = source
        self.page = page
