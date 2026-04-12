"""
Persistence Layer for HAVEN Multi-Camera Tracking System
Handles GlobalID state, embeddings, and metadata across restarts.

Author: Senior MLOps Engineer
Date: 2026-02-02
"""

import sqlite3
import numpy as np
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading
import logging

logger = logging.getLogger(__name__)


class PersistenceManager:
    """
    Thread-safe persistence for GlobalID tracking system.
    
    Storage Strategy:
    - SQLite: Metadata (global_ids, timestamps, camera transitions)
    - Numpy memmap: Embeddings (memory-mapped arrays for large-scale)
    - Pickle: Configuration and auxiliary data
    """
    
    def __init__(self, persist_path: str, embedding_dim: int = 512):
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.persist_path / "haven_state.db"
        self.embeddings_path = self.persist_path / "embeddings.npy"
        self.config_path = self.persist_path / "config.pkl"
        
        self.embedding_dim = embedding_dim
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Load or create embeddings memmap
        self._init_embeddings()
        
        logger.info(f"PersistenceManager initialized at {self.persist_path}")
    
    def _init_database(self):
        """Initialize SQLite database with schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Global IDs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_ids (
                    global_id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    first_camera TEXT NOT NULL,
                    embedding_idx INTEGER NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    last_seen_at TEXT
                )
            """)
            
            # Camera appearances table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS camera_appearances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_id INTEGER NOT NULL,
                    camera TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    bbox TEXT,  -- JSON encoded
                    FOREIGN KEY (global_id) REFERENCES global_ids(global_id)
                )
            """)
            
            # Create index for fast queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_appearances_global_id 
                ON camera_appearances(global_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_appearances_camera 
                ON camera_appearances(camera, timestamp)
            """)
            
            # Metadata table (for next_global_id counter)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            # Initialize next_global_id if not exists
            cursor = conn.execute("SELECT value FROM metadata WHERE key='next_global_id'")
            if cursor.fetchone() is None:
                conn.execute("INSERT INTO metadata VALUES ('next_global_id', '1')")
            
            conn.commit()
    
    def _init_embeddings(self):
        """Initialize or load embeddings memmap."""
        if self.embeddings_path.exists():
            # Calculate shape from file size
            file_size = self.embeddings_path.stat().st_size
            if file_size == 0:
                # Corrupted or empty? Re-init
                logger.warning("Embeddings file exists but is distinct empty. Re-initializing.")
                self.embeddings_path.unlink()
                self._create_new_memmap()
                return

            # Raw binary file size = N * dim * 4 bytes (float32)
            # Verify alignment
            row_size = self.embedding_dim * 4
            if file_size % row_size != 0:
                logger.error(f"Embeddings file size {file_size} not aligned with dim {self.embedding_dim}")
                # Fallback or error? For now, error
                raise ValueError("Corrupted embeddings file size")
            
            n_rows = file_size // row_size
            
            # Load existing memmap using np.memmap (not np.load which expects .npy header)
            self.embeddings = np.memmap(
                str(self.embeddings_path),
                dtype='float32',
                mode='r+',
                shape=(n_rows, self.embedding_dim)
            )
            logger.info(f"Loaded embeddings memmap: shape={self.embeddings.shape}")
        else:
            self._create_new_memmap()

    def _create_new_memmap(self):
        """Helper to create new memmap."""
        # Create new memmap (initial size: 10000 IDs × embedding_dim)
        initial_size = 10000
        self.embeddings = np.memmap(
            str(self.embeddings_path),
            dtype='float32',
            mode='w+',
            shape=(initial_size, self.embedding_dim)
        )
        logger.info(f"Created new embeddings memmap: shape={self.embeddings.shape}")
    
    def get_next_global_id(self) -> int:
        """
        Get and increment next GlobalID (atomic operation).
        This is ONLY called by MASTER camera.
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT value FROM metadata WHERE key='next_global_id'")
                next_id = int(cursor.fetchone()[0])
                
                # Increment
                conn.execute(
                    "UPDATE metadata SET value=? WHERE key='next_global_id'",
                    (str(next_id + 1),)
                )
                conn.commit()
                
                logger.info(f"Issued new GlobalID: {next_id}")
                return next_id
    
    def register_global_id(
        self, 
        global_id: int,
        camera: str,
        embedding: np.ndarray,
        bbox: Optional[List[int]] = None
    ):
        """
        Register a new GlobalID in the system.
        
        Args:
            global_id: The GlobalID number
            camera: Camera where person first appeared
            embedding: Feature embedding (will be stored in memmap)
            bbox: Bounding box [x1, y1, x2, y2]
        """
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # Store embedding in memmap
            embedding_idx = global_id - 1  # 0-indexed
            
            # Expand memmap if needed
            if embedding_idx >= self.embeddings.shape[0]:
                self._expand_embeddings(embedding_idx + 1000)
            
            self.embeddings[embedding_idx] = embedding.astype('float32')
            self.embeddings.flush()  # Force write to disk
            
            # Store metadata in SQLite
            with sqlite3.connect(self.db_path) as conn:
                # Check if already exists (for idempotency)
                cursor = conn.execute(
                    "SELECT global_id FROM global_ids WHERE global_id=?",
                    (global_id,)
                )
                if cursor.fetchone() is None:
                    conn.execute("""
                        INSERT INTO global_ids 
                        (global_id, created_at, first_camera, embedding_idx, last_seen_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (global_id, timestamp, camera, embedding_idx, timestamp))
                
                # Log appearance
                bbox_str = str(bbox) if bbox else None
                conn.execute("""
                    INSERT INTO camera_appearances 
                    (global_id, camera, timestamp, bbox)
                    VALUES (?, ?, ?, ?)
                """, (global_id, camera, timestamp, bbox_str))
                
                conn.commit()
            
            logger.debug(f"Registered GlobalID {global_id} from {camera}")
    
    def update_appearance(
        self,
        global_id: int,
        camera: str,
        embedding: Optional[np.ndarray] = None,
        bbox: Optional[List[int]] = None
    ):
        """
        Update appearance of existing GlobalID (for EMA update).
        
        Args:
            global_id: Existing GlobalID
            camera: Camera where seen
            embedding: New embedding (optional, for EMA update)
            bbox: Bounding box
        """
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # Update embedding if provided
            if embedding is not None:
                embedding_idx = global_id - 1
                if embedding_idx < self.embeddings.shape[0]:
                    # EMA update: 0.3 * new + 0.7 * old
                    alpha = 0.3
                    old_embedding = self.embeddings[embedding_idx]
                    updated = alpha * embedding + (1 - alpha) * old_embedding
                    # Re-normalize
                    updated = updated / np.linalg.norm(updated)
                    self.embeddings[embedding_idx] = updated.astype('float32')
                    self.embeddings.flush()
            
            # Log appearance
            with sqlite3.connect(self.db_path) as conn:
                bbox_str = str(bbox) if bbox else None
                conn.execute("""
                    INSERT INTO camera_appearances 
                    (global_id, camera, timestamp, bbox)
                    VALUES (?, ?, ?, ?)
                """, (global_id, camera, timestamp, bbox_str))
                
                # Update last_seen_at
                conn.execute("""
                    UPDATE global_ids 
                    SET last_seen_at=?
                    WHERE global_id=?
                """, (timestamp, global_id))
                
                conn.commit()
    
    def get_embedding(self, global_id: int) -> Optional[np.ndarray]:
        """Get embedding for a GlobalID."""
        embedding_idx = global_id - 1
        if 0 <= embedding_idx < self.embeddings.shape[0]:
            return self.embeddings[embedding_idx].copy()
        return None
    
    def get_all_embeddings(self, active_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all embeddings and their corresponding GlobalIDs.
        
        Returns:
            embeddings: (N, embedding_dim) array
            global_ids: (N,) array of GlobalID numbers
        """
        with sqlite3.connect(self.db_path) as conn:
            if active_only:
                cursor = conn.execute("""
                    SELECT global_id, embedding_idx 
                    FROM global_ids 
                    WHERE is_active=1
                    ORDER BY global_id
                """)
            else:
                cursor = conn.execute("""
                    SELECT global_id, embedding_idx 
                    FROM global_ids 
                    ORDER BY global_id
                """)
            
            results = cursor.fetchall()
        
        if not results:
            return np.array([]), np.array([])
        
        global_ids = np.array([r[0] for r in results])
        embedding_indices = np.array([r[1] for r in results])
        
        embeddings = self.embeddings[embedding_indices]
        
        return embeddings, global_ids
    
    def get_last_seen(self, global_id: int, camera: str) -> Optional[str]:
        """Get last seen timestamp for GlobalID at specific camera."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp 
                FROM camera_appearances
                WHERE global_id=? AND camera=?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (global_id, camera))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def _expand_embeddings(self, new_size: int):
        """Expand embeddings memmap when capacity exceeded."""
        logger.info(f"Expanding embeddings memmap to {new_size}")
        
        # Create new larger memmap
        new_path = self.embeddings_path.with_suffix('.new.npy')
        new_embeddings = np.memmap(
            str(new_path),
            dtype='float32',
            mode='w+',
            shape=(new_size, self.embedding_dim)
        )
        
        # Copy old data
        old_size = self.embeddings.shape[0]
        new_embeddings[:old_size] = self.embeddings
        new_embeddings.flush()
        
        # Replace old with new
        del self.embeddings
        shutil.move(str(new_path), str(self.embeddings_path))
        
        # Reload
        self.embeddings = np.memmap(
            str(self.embeddings_path),
            dtype='float32',
            mode='r+',
            shape=(new_size, self.embedding_dim)
        )
        
        logger.info(f"Embeddings memmap expanded to {new_size}")
    
    def create_backup(self, backup_dir: Optional[str] = None):
        """Create backup of current state."""
        if backup_dir is None:
            backup_dir = self.persist_path / "backups"
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"
        backup_path.mkdir()
        
        # Backup database
        with sqlite3.connect(self.db_path) as conn:
            backup_db = sqlite3.connect(backup_path / "haven_state.db")
            conn.backup(backup_db)
            backup_db.close()
        
        # Backup embeddings
        shutil.copy(self.embeddings_path, backup_path / "embeddings.npy")
        
        logger.info(f"Backup created at {backup_path}")
        return backup_path
    
    def close(self):
        """Close resources (flush memmap and release file handle)."""
        if hasattr(self, 'embeddings'):
            try:
                self.embeddings.flush()
                # Delete reference to memmap to close file handle
                del self.embeddings
            except Exception as e:
                logger.error(f"Error closing embeddings: {e}")
        
        # Force garbage collection to release Windows file handles
        import gc
        gc.collect()
        
        logger.info("PersistenceManager closed")

    def get_statistics(self) -> Dict:
        """Get system statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Total GlobalIDs
            cursor = conn.execute("SELECT COUNT(*) FROM global_ids WHERE is_active=1")
            total_ids = cursor.fetchone()[0]
            
            # Next ID
            cursor = conn.execute("SELECT value FROM metadata WHERE key='next_global_id'")
            next_id = int(cursor.fetchone()[0])
            
            # Appearances per camera
            cursor = conn.execute("""
                SELECT camera, COUNT(*) 
                FROM camera_appearances 
                GROUP BY camera
            """)
            appearances = dict(cursor.fetchall())
        
        return {
            'total_global_ids': total_ids,
            'next_global_id': next_id,
            'appearances_per_camera': appearances,
            'embeddings_capacity': self.embeddings.shape[0] if hasattr(self, 'embeddings') else 0,
            'embeddings_used': next_id - 1
        }
