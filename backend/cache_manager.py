"""
Response Cache Manager - SQLite-based caching for fast repeated queries.

Implements a simple key-value cache using SQLite with TTL support.
"""

import sqlite3
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
import threading

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    SQLite-based response cache with TTL support.
    
    Cache Strategy:
    - Key: hash(query + model_name + str(doc_ids))
    - Value: JSON-encoded response
    - TTL: Default 24 hours
    """
    
    def __init__(self, db_path: str = "./cache.db", default_ttl_hours: int = 24):
        """
        Initialize cache with SQLite backend.
        
        Args:
            db_path: Path to SQLite database file
            default_ttl_hours: Default time-to-live in hours
        """
        self.db_path = Path(db_path)
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.lock = threading.Lock()
        self._init_db()
        logger.info(f"✅ ResponseCache initialized at {self.db_path}")
    
    def _init_db(self):
        """Create cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            # Index for faster cleanup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON response_cache(expires_at)
            """)
            conn.commit()
    
    def _generate_key(self, query: str, model_name: str, doc_ids: list = None) -> str:
        """
        Generate cache key from query, model, and document IDs.
        
        Args:
            query: User query
            model_name: Model used for generation
            doc_ids: List of document IDs (for RAG queries)
        
        Returns:
            MD5 hash as cache key
        """
        # Normalize query
        query_normalized = query.strip().lower()
        
        # Create composite key
        key_parts = [query_normalized, model_name]
        if doc_ids:
            key_parts.append(",".join(sorted(doc_ids)))
        
        composite = "|".join(key_parts)
        return hashlib.md5(composite.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        model_name: str,
        doc_ids: list = None
    ) -> Optional[dict]:
        """
        Retrieve cached response if available and not expired.
        
        Args:
            query: User query
            model_name: Model name
            doc_ids: Document IDs
        
        Returns:
            Cached response or None
        """
        key = self._generate_key(query, model_name, doc_ids)
        
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT value, expires_at FROM response_cache
                        WHERE key = ? AND expires_at > ?
                    """, (key, datetime.now()))
                    
                    row = cursor.fetchone()
                    if row:
                        logger.info(f"✅ Cache HIT for key: {key[:8]}...")
                        return json.loads(row[0])
                    else:
                        logger.info(f"❌ Cache MISS for key: {key[:8]}...")
                        return None
            except Exception as e:
                logger.error(f"Cache retrieval error: {e}")
                return None
    
    def set(
        self,
        query: str,
        model_name: str,
        response: dict,
        doc_ids: list = None,
        ttl_hours: Optional[int] = None
    ) -> bool:
        """
        Store response in cache.
        
        Args:
            query: User query
            model_name: Model name
            response: Response to cache (will be JSON-encoded)
            doc_ids: Document IDs
            ttl_hours: Custom TTL in hours
        
        Returns:
            Success status
        """
        key = self._generate_key(query, model_name, doc_ids)
        ttl = timedelta(hours=ttl_hours) if ttl_hours else self.default_ttl
        expires_at = datetime.now() + ttl
        
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO response_cache (key, value, expires_at)
                        VALUES (?, ?, ?)
                    """, (key, json.dumps(response), expires_at))
                    conn.commit()
                    logger.info(f"💾 Cached response for key: {key[:8]}... (TTL: {ttl_hours or self.default_ttl.total_seconds()/3600}h)")
                    return True
            except Exception as e:
                logger.error(f"Cache storage error: {e}")
                return False
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of deleted entries
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        DELETE FROM response_cache
                        WHERE expires_at <= ?
                    """, (datetime.now(),))
                    conn.commit()
                    deleted = cursor.rowcount
                    if deleted > 0:
                        logger.info(f"🧹 Cleaned up {deleted} expired cache entries")
                    return deleted
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                return 0
    
    def clear_all(self) -> bool:
        """
        Clear entire cache.
        
        Returns:
            Success status
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM response_cache")
                    conn.commit()
                    logger.warning("🗑️ Cache cleared completely")
                    return True
            except Exception as e:
                logger.error(f"Cache clear error: {e}")
                return False
    
    def stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN expires_at > ? THEN 1 ELSE 0 END) as valid,
                        SUM(CASE WHEN expires_at <= ? THEN 1 ELSE 0 END) as expired
                    FROM response_cache
                """, (datetime.now(), datetime.now()))
                
                row = cursor.fetchone()
                return {
                    "total_entries": row[0],
                    "valid_entries": row[1],
                    "expired_entries": row[2],
                    "db_path": str(self.db_path)
                }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"error": str(e)}
