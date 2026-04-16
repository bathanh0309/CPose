import numpy as np
import logging
from typing import List, Tuple, Optional
import threading

from app.utils.runtime_config import get_runtime_section

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

logger = logging.getLogger(__name__)

_VECTOR_DB_DEFAULTS = get_runtime_section("vector_db")
SEARCH_TOP_K = int(_VECTOR_DB_DEFAULTS.get("search_top_k", 20))
MEDIUM_DATASET_THRESHOLD = int(_VECTOR_DB_DEFAULTS.get("medium_dataset_threshold", 1000))
LARGE_DATASET_THRESHOLD = int(_VECTOR_DB_DEFAULTS.get("large_dataset_threshold", 10000))
HNSW_M = int(_VECTOR_DB_DEFAULTS.get("hnsw_m", 32))
HNSW_EF_CONSTRUCTION = int(_VECTOR_DB_DEFAULTS.get("hnsw_ef_construction", 200))
HNSW_EF_SEARCH = int(_VECTOR_DB_DEFAULTS.get("hnsw_ef_search", 64))
IVF_NLIST = int(_VECTOR_DB_DEFAULTS.get("ivf_nlist", 100))
IVF_NPROBE = int(_VECTOR_DB_DEFAULTS.get("ivf_nprobe", 10))


class VectorDatabase:
    """
    FAISS-based vector database for fast similarity search.
    
    Strategy:
    - Small dataset (N < 1000): Use IndexFlatL2 (exact search)
    - Medium dataset (1000 <= N < 10000): Use IndexHNSWFlat
    - Large dataset (N >= 10000): Use IndexIVFFlat with quantization
    """
    
    def __init__(
        self, 
        embedding_dim: int,
        index_type: str = 'auto',
        metric: str = 'cosine'  # 'cosine' or 'l2'
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            index_type: 'flat', 'hnsw', 'ivf', 'auto'
            metric: Distance metric
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.lock = threading.Lock()
        
        # Storage for metadata
        self.global_ids = np.array([], dtype=np.int32)
        
        # Initialize index
        self.index = None
        self._init_index()
        
        logger.info(
            f"VectorDatabase initialized: dim={embedding_dim}, "
            f"type={index_type}, metric={metric}"
        )
    
    def _init_index(self):
        """Initialize FAISS index based on strategy."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using fallback linear search")
            self.index = None
            return
        
        # Normalize embeddings for cosine similarity
        if self.metric == 'cosine':
            # Use inner product after L2 normalization = cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # Use L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        logger.info(f"Created FAISS index: {self.index}")
    
    def add(self, embeddings: np.ndarray, global_ids: np.ndarray):
        """
        Add embeddings to index.
        
        Args:
            embeddings: (N, embedding_dim) array
            global_ids: (N,) array of GlobalID numbers
        """
        with self.lock:
            if embeddings.shape[0] == 0:
                return
            
            # Ensure float32
            embeddings = embeddings.astype('float32')
            
            # Normalize for cosine similarity
            if self.metric == 'cosine':
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)
            
            if self.index is None:
                # Fallback: store in numpy array
                if not hasattr(self, 'fallback_embeddings'):
                    self.fallback_embeddings = embeddings
                else:
                    self.fallback_embeddings = np.vstack([
                        self.fallback_embeddings, 
                        embeddings
                    ])
            else:
                # Add to FAISS index
                self.index.add(embeddings)
            
            # Store metadata
            self.global_ids = np.concatenate([self.global_ids, global_ids])
            
            # Check if need to upgrade index type
            self._maybe_upgrade_index()
            
            logger.debug(f"Added {len(global_ids)} embeddings to index")
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = SEARCH_TOP_K,
        allowed_ids: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for top-k similar embeddings.
        
        Args:
            query: (embedding_dim,) query vector
            k: Number of neighbors to return
            allowed_ids: If provided, only return results from these GlobalIDs
        
        Returns:
            distances: (k,) array of distances/similarities
            global_ids: (k,) array of GlobalID numbers
            indices: (k,) array of internal indices
        """
        with self.lock:
            if len(self.global_ids) == 0:
                return np.array([]), np.array([]), np.array([])
            
            # Ensure float32
            query = query.astype('float32').reshape(1, -1)
            
            # Normalize for cosine similarity
            if self.metric == 'cosine':
                query = query / (np.linalg.norm(query) + 1e-8)
            
            # Adjust k if larger than database size
            k = min(k, len(self.global_ids))
            
            if self.index is None:
                # Fallback linear search
                similarities = self._linear_search(query[0], k)
                indices = np.argsort(similarities)[::-1][:k]
                distances = similarities[indices]
                result_global_ids = self.global_ids[indices]
            else:
                # FAISS search
                if self.metric == 'cosine':
                    # For inner product, larger is better
                    distances, indices = self.index.search(query, k)
                    distances = distances[0]  # (k,)
                    indices = indices[0]  # (k,)
                else:
                    # For L2, smaller is better
                    distances, indices = self.index.search(query, k)
                    distances = distances[0]
                    indices = indices[0]
                
                result_global_ids = self.global_ids[indices]
            
            # Filter by allowed_ids if provided
            if allowed_ids is not None:
                allowed_set = set(allowed_ids)
                mask = np.array([gid in allowed_set for gid in result_global_ids])
                distances = distances[mask]
                result_global_ids = result_global_ids[mask]
                indices = indices[mask]
            
            return distances, result_global_ids, indices
    
    def _linear_search(self, query: np.ndarray, k: int) -> np.ndarray:
        """Fallback linear search when FAISS not available."""
        if self.metric == 'cosine':
            # Cosine similarity
            similarities = np.dot(self.fallback_embeddings, query)
        else:
            # L2 distance (convert to similarity)
            distances = np.linalg.norm(
                self.fallback_embeddings - query, 
                axis=1
            )
            similarities = 1.0 / (1.0 + distances)
        
        return similarities
    
    def _maybe_upgrade_index(self):
        """Upgrade index type based on database size."""
        if not FAISS_AVAILABLE or self.index_type != 'auto':
            return
        
        n = len(self.global_ids)
        
        # Upgrade strategy
        if n >= LARGE_DATASET_THRESHOLD and not isinstance(self.index, faiss.IndexIVFFlat):
            logger.info("Upgrading to IndexIVFFlat for large dataset")
            self._upgrade_to_ivf()
        elif n >= MEDIUM_DATASET_THRESHOLD and not isinstance(self.index, faiss.IndexHNSWFlat):
            logger.info("Upgrading to IndexHNSWFlat for medium dataset")
            self._upgrade_to_hnsw()
    
    def _upgrade_to_hnsw(self):
        """Upgrade to HNSW index for better performance."""
        if self.metric == 'cosine':
            new_index = faiss.IndexHNSWFlat(self.embedding_dim, HNSW_M)
        else:
            new_index = faiss.IndexHNSWFlat(self.embedding_dim, HNSW_M)
        
        # Set HNSW parameters
        new_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        new_index.hnsw.efSearch = HNSW_EF_SEARCH
        
        # Copy old vectors
        if self.index.ntotal > 0:
            old_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            new_index.add(old_vectors)
        
        self.index = new_index
        logger.info("Upgraded to HNSW index")
    
    def _upgrade_to_ivf(self):
        """Upgrade to IVF index for very large datasets."""
        nlist = IVF_NLIST  # Number of clusters
        
        if self.metric == 'cosine':
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            new_index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dim, 
                nlist
            )
        else:
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            new_index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dim, 
                nlist
            )
        
        # Train on existing data
        if self.index.ntotal > 0:
            old_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            new_index.train(old_vectors)
            new_index.add(old_vectors)
        
        # Set search parameters
        new_index.nprobe = IVF_NPROBE  # Number of clusters to search
        
        self.index = new_index
        logger.info("Upgraded to IVF index")
    
    def rebuild(self, embeddings: np.ndarray, global_ids: np.ndarray):
        """
        Rebuild index from scratch.
        Useful after bulk updates or when switching index type.
        
        Args:
            embeddings: (N, embedding_dim) array
            global_ids: (N,) array of GlobalID numbers
        """
        with self.lock:
            logger.info(f"Rebuilding index with {len(global_ids)} vectors")
            
            # Reinitialize index
            self._init_index()
            self.global_ids = np.array([], dtype=np.int32)
            
            # Add all vectors
            self.add(embeddings, global_ids)
            
            logger.info("Index rebuild complete")
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            'total_vectors': len(self.global_ids),
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.index).__name__ if self.index else 'None',
            'metric': self.metric,
            'faiss_available': FAISS_AVAILABLE
        }


class HybridMatcher:
    """
    Hybrid matcher that combines Vector DB search with filtering.
    """
    
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
    
    def match(
        self,
        query_embedding: np.ndarray,
        allowed_ids: Optional[List[int]] = None,
        top_k: int = SEARCH_TOP_K
    ) -> List[dict]:
        """
        Find best matches for query embedding.
        
        Args:
            query_embedding: Query vector
            allowed_ids: Filter to only these GlobalIDs
            top_k: Number of candidates to return
        
        Returns:
            List of match results sorted by score (best first)
        """
        distances, global_ids, indices = self.vector_db.search(
            query_embedding,
            k=top_k,
            allowed_ids=allowed_ids
        )
        
        results = []
        for dist, gid, idx in zip(distances, global_ids, indices):
            # Convert distance to similarity score
            if self.vector_db.metric == 'cosine':
                # Inner product is already similarity
                score = float(dist)
            else:
                # Convert L2 distance to similarity
                score = 1.0 / (1.0 + float(dist))
            
            results.append({
                'global_id': int(gid),
                'score': score,
                'distance': float(dist),
                'index': int(idx)
            })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
