#!/usr/bin/env python3
"""
FAISS vector indexing and metadata persistence.

Handles vector storage, indexing, and metadata management.
"""

import json
import sqlite3
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import os

logger = logging.getLogger(__name__)

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class VectorIndex:
    """FAISS-based vector index with metadata persistence."""
    
    def __init__(self, index_path: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector index.
        
        Args:
            index_path: Path to store index files
            embedding_model_name: Name of sentence transformer model
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.index = None
        self.metadata_db_path = self.index_path / "metadata.db"
        self.vectors_path = self.index_path / "vectors.pkl"
        # Parallel ids list persisted to keep vector order stable across incremental writes
        self.vector_ids_path = self.index_path / "vector_ids.pkl"
        
        # Initialize components
        self._load_embedding_model()
        self._load_or_create_index()
        self._setup_metadata_db()
    
    def _load_embedding_model(self):
        """Load the embedding model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("sentence-transformers not available, using stub embeddings")
            return
        
        try:
            # Resolve device (prefer CUDA → MPS → CPU), allow override via VECTOR_DEVICE
            device = 'cpu'
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                    device = 'mps'
            except Exception:
                device = 'cpu'

            override = os.getenv('VECTOR_DEVICE')
            if override:
                device = override

            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
            logger.info(f"Loaded embedding model: {self.embedding_model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        if not HAS_FAISS:
            logger.warning("FAISS not available, using in-memory storage")
            self.index = None
            return
        
        index_file = self.index_path / "index.faiss"
        
        if index_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                self.index = None
        else:
            logger.info("No existing index found, will create new one")
            self.index = None
    
    def _setup_metadata_db(self):
        """Set up SQLite database for metadata."""
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    vod_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration REAL NOT NULL,
                    chapter_id TEXT,
                    category TEXT,
                    excluded INTEGER DEFAULT 0,
                    mode TEXT,
                    text TEXT,
                    chat_text TEXT,
                    chat_rate REAL DEFAULT 0,
                    chat_rate_z REAL DEFAULT 0,
                    burst_score REAL DEFAULT 0,
                    reaction_hits TEXT,
                    section_context TEXT,
                    summary TEXT,
                    topic TEXT,
                    energy TEXT,
                    role TEXT,
                    role_confidence REAL DEFAULT 0,
                    same_topic_prev INTEGER DEFAULT 0,
                    topic_thread INTEGER DEFAULT 0,
                    topic_key TEXT,
                    link_type TEXT,
                    link_evidence TEXT,
                    confidence REAL DEFAULT 0,
                    section_id TEXT,
                    section_title TEXT,
                    section_role TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vod_id ON documents(vod_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mode ON documents(mode)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chapter_id ON documents(chapter_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_time_range ON documents(start_time, end_time)
            """)
            
            conn.commit()

            # Ensure new columns exist (auto-migration)
            cursor.execute("PRAGMA table_info(documents)")
            cols = {row[1] for row in cursor.fetchall()}  # set of names
            required = {
                ("chat_rate", "REAL DEFAULT 0"),
                ("chat_rate_z", "REAL DEFAULT 0"),
                ("burst_score", "REAL DEFAULT 0"),
                ("reaction_hits", "TEXT"),
                ("section_context", "TEXT"),
                ("summary", "TEXT"),
                ("topic", "TEXT"),
                ("energy", "TEXT"),
                ("role", "TEXT"),
                ("role_confidence", "REAL DEFAULT 0"),
                ("same_topic_prev", "INTEGER DEFAULT 0"),
                ("topic_thread", "INTEGER DEFAULT 0"),
                ("topic_key", "TEXT"),
                ("link_type", "TEXT"),
                ("link_evidence", "TEXT"),
                ("confidence", "REAL DEFAULT 0"),
            }
            for name, decl in required:
                if name not in cols:
                    cursor.execute(f"ALTER TABLE documents ADD COLUMN {name} {decl}")
            conn.commit()
            conn.close()
            logger.info("Metadata database setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup metadata database: {e}")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            Numpy array of embeddings
        """
        if not self.embedding_model:
            # Return stub embeddings
            return np.random.rand(len(texts), 768).astype(np.float32)
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            # Normalize embeddings for better similarity search
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return np.random.rand(len(texts), 768).astype(np.float32)
    
    def add_documents(self, documents: List, embedding_texts: List[str]):
        """
        Add documents to the index.
        
        Args:
            documents: List of VectorDocument objects
            embedding_texts: List of formatted text strings for embedding
        """
        if not documents:
            return
        
        logger.info(f"Adding {len(documents)} documents to index")
        
        # Create embeddings
        embeddings = self.create_embeddings(embedding_texts)
        
        # Add to FAISS index
        if HAS_FAISS:
            if self.index is None:
                # Create new index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node
                logger.info(f"Created new FAISS HNSW index with dimension {dimension}")
            
            # Add vectors to index
            self.index.add(embeddings)
            logger.info(f"Added {len(embeddings)} vectors to FAISS index")
        
        # Store vectors + ids for fallback and Retriever alignment
        self._store_vectors(embeddings, [d.id for d in documents])
        
        # Add metadata to database
        self._add_metadata_to_db(documents)
        
        # Save index
        self._save_index()
    
    def _store_vectors(self, vectors: np.ndarray, ids: List[str]):
        """Append-safe vector storage with aligned ids.

        Ensures that vectors.pkl and vector_ids.pkl always have the same length and order.
        Existing ids are preserved; new ids are appended; duplicates are updated in-place.
        """
        try:
            existing_ids: List[str] = []
            existing_vecs: Optional[np.ndarray] = None

            if self.vector_ids_path.exists() and self.vectors_path.exists():
                try:
                    with open(self.vector_ids_path, 'rb') as f:
                        existing_ids = pickle.load(f)
                    with open(self.vectors_path, 'rb') as f:
                        existing_vecs = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load existing vectors; starting fresh: {e}")
                    existing_ids = []
                    existing_vecs = None

            # Build id -> index map for fast updates
            id_to_index: Dict[str, int] = {doc_id: idx for idx, doc_id in enumerate(existing_ids)}

            # Prepare output containers
            out_ids: List[str] = list(existing_ids)
            out_vecs: List[np.ndarray] = []
            if existing_vecs is not None:
                # Split existing vectors into list to allow in-place replacement
                out_vecs = [existing_vecs[i] for i in range(existing_vecs.shape[0])]

            # Append/update with new batch
            for row_idx, doc_id in enumerate(ids):
                if doc_id in id_to_index:
                    # Update existing vector
                    out_vecs[id_to_index[doc_id]] = vectors[row_idx]
                else:
                    # Append new id/vector
                    id_to_index[doc_id] = len(out_ids)
                    out_ids.append(doc_id)
                    out_vecs.append(vectors[row_idx])

            # Persist in lock-step order
            out_mat = np.asarray(out_vecs, dtype=np.float32)
            with open(self.vectors_path, 'wb') as f:
                pickle.dump(out_mat, f)
            with open(self.vector_ids_path, 'wb') as f:
                pickle.dump(out_ids, f)
            logger.debug(f"Stored {len(out_ids)} total vectors (appended {len(ids)})")
        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
    
    def _add_metadata_to_db(self, documents: List):
        """Add document metadata to SQLite database."""
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            for doc in documents:
                cursor.execute("""
                    INSERT OR REPLACE INTO documents (
                        id, vod_id, start_time, end_time, duration,
                        chapter_id, category, excluded, mode,
                        text, chat_text,
                        chat_rate, chat_rate_z, burst_score,
                        reaction_hits, section_context,
                        summary, topic, energy,
                        role, role_confidence,
                        same_topic_prev, topic_thread, topic_key,
                        link_type, link_evidence, confidence,
                        section_id, section_title, section_role
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc.id, doc.vod_id, doc.start, doc.end, doc.len_s,
                    doc.chapter_id, doc.category, doc.excluded, doc.mode,
                    doc.text, doc.chat_text,
                    getattr(doc, 'chat_rate', 0.0), getattr(doc, 'chat_rate_z', 0.0), getattr(doc, 'burst_score', 0.0),
                    json.dumps(getattr(doc, 'reaction_hits', {})),
                    getattr(doc, 'section_context', ''),
                    '', '', '',
                    '', 0.0,
                    0, 0, '',
                    '', '', 0.0,
                    doc.section_id, doc.section_title, doc.section_role
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Added {len(documents)} documents to metadata database")
            
        except Exception as e:
            logger.error(f"Failed to add metadata to database: {e}")
    
    def _save_index(self):
        """Save FAISS index to disk."""
        if not HAS_FAISS or not self.index:
            return
        
        try:
            index_file = self.index_path / "index.faiss"
            faiss.write_index(self.index, str(index_file))
            logger.info(f"Saved FAISS index to {index_file}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def search(self, query: str, k: int = 10, filters: Optional[Dict] = None) -> List[Tuple[Dict, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            filters: Optional filters to apply
        
        Returns:
            List of (document, score) tuples
        """
        # Create query embedding (falls back to stub embeddings when model is unavailable)
        query_embedding = self.create_embeddings([query])
        
        # Search FAISS index
        if HAS_FAISS and self.index:
            scores, indices = self.index.search(query_embedding, k * 2)  # Oversample for filtering
            results = []
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                
                # Get document metadata
                doc = self._get_document_by_index(idx)
                if doc:
                    results.append((doc, float(score)))
            
            # Apply filters
            if filters:
                results = self._apply_filters(results, filters)
            
            return results[:k]
        
        else:
            # Fallback: brute force search with stored vectors
            return self._fallback_search(query_embedding[0], k, filters)
    
    def _get_document_by_index(self, index: int) -> Optional[Dict]:
        """Get document metadata by index."""
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            # Get all documents and select by index
            cursor.execute("""
                SELECT id, vod_id, start_time, end_time, duration,
                       chapter_id, category, excluded, mode,
                       text, chat_text,
                       section_id, section_title, section_role
                FROM documents
                ORDER BY created_at
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if 0 <= index < len(rows):
                return self._row_to_document(rows[index])
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document by index {index}: {e}")
            return None
    
    def _row_to_document(self, row: Tuple) -> Dict:
        """Convert database row to document dictionary."""
        return {
            'id': row[0],
            'vod_id': row[1],
            'start': row[2],
            'end': row[3],
            'len_s': row[4],
            'chapter_id': row[5],
            'category': row[6],
            'excluded': bool(row[7]),
            'mode': row[8],
            'text': row[9],
            'chat_text': row[10],
            'section_id': row[11],
            'section_title': row[12],
            'section_role': row[13]
        }
    
    def _apply_filters(self, results: List[Tuple[Dict, float]], 
                      filters: Dict) -> List[Tuple[Dict, float]]:
        """Apply filters to search results."""
        filtered = []
        
        for doc, score in results:
            # Skip excluded documents
            if doc.get('excluded', False):
                continue
            
            # Apply filters
            if 'vod_id' in filters and doc.get('vod_id') != filters['vod_id']:
                continue
            
            if 'mode' in filters and doc.get('mode') != filters['mode']:
                continue
            
            if 'chapter_id' in filters and doc.get('chapter_id') != filters['chapter_id']:
                continue
            
            if 'category' in filters and doc.get('category') != filters['category']:
                continue
            
            if 'time_range' in filters:
                start, end = filters['time_range']
                if not (start <= doc.get('start', 0) <= end):
                    continue
            
            if 'min_len_s' in filters and doc.get('len_s', 0) < filters['min_len_s']:
                continue
            
            filtered.append((doc, score))
        
        return filtered
    
    def _fallback_search(self, query_embedding: np.ndarray, k: int, 
                        filters: Optional[Dict]) -> List[Tuple[Dict, float]]:
        """Fallback search using stored vectors."""
        try:
            # Load stored vectors
            with open(self.vectors_path, 'rb') as f:
                stored_vectors = pickle.load(f)
            
            # Calculate similarities
            similarities = np.dot(stored_vectors, query_embedding)
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:k * 2]  # Oversample
            
            results = []
            for idx in top_indices:
                doc = self._get_document_by_index(idx)
                if doc:
                    results.append((doc, float(similarities[idx])))
            
            # Apply filters
            if filters:
                results = self._apply_filters(results, filters)
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = {
            'index_path': str(self.index_path),
            'embedding_model': self.embedding_model_name,
            'has_faiss': HAS_FAISS,
            'has_embedding_model': self.embedding_model is not None
        }
        
        if HAS_FAISS and self.index:
            stats['vector_count'] = self.index.ntotal
            stats['dimension'] = self.index.d
        
        # Get document count from database
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            stats['document_count'] = cursor.fetchone()[0]
            conn.close()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            stats['document_count'] = 0
        
        return stats

    def get_ids(self) -> List[str]:
        """Return the current ordered list of vector ids if persisted; else fall back to DB order.

        This is used by narrative indexers to de-duplicate before adding new documents.
        """
        try:
            if self.vector_ids_path.exists():
                with open(self.vector_ids_path, 'rb') as f:
                    ids = pickle.load(f)
                if isinstance(ids, list):
                    return [str(x) for x in ids]
        except Exception as e:
            logger.warning(f"Failed to read vector_ids.pkl: {e}")

        # Fallback to DB order
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cur = conn.cursor()
            cur.execute("SELECT id FROM documents ORDER BY created_at")
            rows = cur.fetchall()
            conn.close()
            return [str(r[0]) for r in rows]
        except Exception as e:
            logger.error(f"Failed to read ids from DB: {e}")
            return []
