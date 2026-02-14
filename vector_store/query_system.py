#!/usr/bin/env python3
"""
Query system with filtering and reranking capabilities.

Handles semantic search queries with advanced filtering and reranking.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class QueryFilters:
    """Query filters for search."""
    vod_id: Optional[str] = None
    mode: Optional[str] = None  # 'jc', 'game', 'unknown'
    chapter_id: Optional[str] = None
    category: Optional[str] = None
    time_range: Optional[Tuple[float, float]] = None  # (start, end) in seconds
    min_len_s: Optional[float] = None
    max_len_s: Optional[float] = None


@dataclass
class SearchResult:
    """Search result with metadata."""
    document: Dict[str, Any]
    score: float
    rank: int


class QuerySystem:
    """Advanced query system with filtering and reranking."""
    
    def __init__(self, vector_index):
        """
        Initialize query system.
        
        Args:
            vector_index: VectorIndex instance
        """
        self.vector_index = vector_index
    
    def search(self, query: str, k: int = 10, filters: Optional[QueryFilters] = None,
              rerank: bool = True, oversample_factor: int = 3) -> List[SearchResult]:
        """
        Perform semantic search with filtering and reranking.
        
        Args:
            query: Search query string
            k: Number of final results to return
            filters: Optional filters to apply
            rerank: Whether to apply reranking
            oversample_factor: Factor to oversample before filtering
        
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Searching for: '{query}' (k={k}, rerank={rerank})")
        
        # Convert filters to dict format
        filter_dict = self._filters_to_dict(filters) if filters else None
        
        # Oversample to account for filtering
        oversample_k = k * oversample_factor
        
        # Perform initial search
        raw_results = self.vector_index.search(query, k=oversample_k, filters=filter_dict)
        
        if not raw_results:
            logger.warning("No results found")
            return []
        
        logger.info(f"Found {len(raw_results)} initial results")
        
        # Convert to SearchResult objects
        results = []
        for i, (doc, score) in enumerate(raw_results):
            result = SearchResult(
                document=doc,
                score=score,
                rank=i + 1
            )
            results.append(result)
        
        # Apply additional filtering
        results = self._apply_advanced_filters(results, filters)
        
        # Apply reranking if requested
        if rerank and len(results) > 1:
            results = self._rerank_results(results, query)
        
        # Return top k results
        final_results = results[:k]
        
        logger.info(f"Returning {len(final_results)} final results")
        return final_results
    
    def _filters_to_dict(self, filters: QueryFilters) -> Dict[str, Any]:
        """Convert QueryFilters to dictionary format."""
        filter_dict = {}
        
        if filters.vod_id:
            filter_dict['vod_id'] = filters.vod_id
        if filters.mode:
            filter_dict['mode'] = filters.mode
        if filters.chapter_id:
            filter_dict['chapter_id'] = filters.chapter_id
        if filters.category:
            filter_dict['category'] = filters.category
        if filters.time_range:
            filter_dict['time_range'] = filters.time_range
        if filters.min_len_s:
            filter_dict['min_len_s'] = filters.min_len_s
        
        return filter_dict
    
    def _apply_advanced_filters(self, results: List[SearchResult], 
                               filters: Optional[QueryFilters]) -> List[SearchResult]:
        """Apply advanced filtering logic."""
        if not filters:
            return results
        
        filtered = []
        
        for result in results:
            doc = result.document
            
            # Skip excluded documents
            if doc.get('excluded', False):
                continue
            
            # Apply max length filter
            if filters.max_len_s and doc.get('len_s', 0) > filters.max_len_s:
                continue
            
            # Additional time range validation
            if filters.time_range:
                start, end = filters.time_range
                doc_start = doc.get('start', 0)
                doc_end = doc.get('end', 0)
                
                # Check if document overlaps with time range
                if not (doc_start < end and doc_end > start):
                    continue
            
            filtered.append(result)
        
        logger.info(f"Advanced filtering: {len(results)} -> {len(filtered)} results")
        return filtered
    
    def _rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Rerank results using text similarity and BM25-style scoring.
        
        Args:
            results: Initial search results
            query: Original query string
        
        Returns:
            Reranked results
        """
        if len(results) <= 1:
            return results
        
        logger.info(f"Reranking {len(results)} results")
        
        # Extract query terms
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        reranked = []
        for result in results:
            doc = result.document
            
            # Calculate text similarity score
            text_score = self._calculate_text_similarity(query, doc)
            
            # Calculate BM25-style keyword score
            keyword_score = self._calculate_keyword_score(query_terms, doc)
            
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(doc)
            
            # Combine scores (weighted average)
            combined_score = (
                result.score * 0.4 +      # Original vector similarity
                text_score * 0.3 +        # Text similarity
                keyword_score * 0.2 +     # Keyword matching
                engagement_score * 0.1    # Engagement metrics
            )
            
            # Create new result with combined score
            reranked_result = SearchResult(
                document=doc,
                score=combined_score,
                rank=result.rank
            )
            reranked.append(reranked_result)
        
        # Sort by combined score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        logger.info("Reranking complete")
        return reranked
    
    def _calculate_text_similarity(self, query: str, doc: Dict[str, Any]) -> float:
        """Calculate text similarity between query and document."""
        # Combine document text
        doc_text = ""
        if doc.get('text'):
            doc_text += doc['text'] + " "
        if doc.get('chat_text'):
            doc_text += doc['chat_text'] + " "
        
        doc_text = doc_text.lower().strip()
        query = query.lower()
        
        if not doc_text or not query:
            return 0.0
        
        # Simple word overlap similarity
        query_words = set(re.findall(r'\b\w+\b', query))
        doc_words = set(re.findall(r'\b\w+\b', doc_text))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(doc_words))
        return overlap / len(query_words)
    
    def _calculate_keyword_score(self, query_terms: set, doc: Dict[str, Any]) -> float:
        """Calculate BM25-style keyword score."""
        # Get document keywords
        doc_keywords = doc.get('keywords', [])
        doc_keywords_lower = [kw.lower() for kw in doc_keywords]
        
        # Count matches
        matches = 0
        for term in query_terms:
            # Direct keyword match
            if term in doc_keywords_lower:
                matches += 1
            # Partial keyword match
            else:
                for kw in doc_keywords_lower:
                    if term in kw or kw in term:
                        matches += 0.5
                        break
        
        if not query_terms:
            return 0.0
        
        return matches / len(query_terms)
    
    def _calculate_engagement_score(self, doc: Dict[str, Any]) -> float:
        """Calculate engagement score based on document metadata."""
        score = 0.0
        
        # Chat activity bonus
        chat_text = doc.get('chat_text', '')
        if chat_text:
            # Count chat messages (rough estimate)
            chat_count = len(chat_text.split())
            if chat_count > 10:
                score += 0.3
            elif chat_count > 5:
                score += 0.2
            else:
                score += 0.1
        
        # Duration bonus (prefer medium-length content)
        duration = doc.get('len_s', 0)
        if 30 <= duration <= 300:  # 30 seconds to 5 minutes
            score += 0.2
        elif 10 <= duration <= 600:  # 10 seconds to 10 minutes
            score += 0.1
        
        # Mode bonus (prefer JC content for general queries)
        mode = doc.get('mode', '')
        if mode == 'jc':
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def search_by_time_range(self, start_time: float, end_time: float, 
                           vod_id: Optional[str] = None) -> List[SearchResult]:
        """
        Search for documents within a specific time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            vod_id: Optional VOD ID filter
        
        Returns:
            List of documents in time range
        """
        filters = QueryFilters(
            vod_id=vod_id,
            time_range=(start_time, end_time)
        )
        
        # Use a generic query since we're filtering by time
        return self.search("content", k=100, filters=filters, rerank=False)
    
    def search_by_chapter(self, chapter_id: str, vod_id: Optional[str] = None,
                         query: str = "content") -> List[SearchResult]:
        """
        Search for documents within a specific chapter.
        
        Args:
            chapter_id: Chapter ID to search
            vod_id: Optional VOD ID filter
            query: Search query (default: "content")
        
        Returns:
            List of documents in chapter
        """
        filters = QueryFilters(
            vod_id=vod_id,
            chapter_id=chapter_id
        )
        
        return self.search(query, k=50, filters=filters)
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        try:
            # Search with very specific filters
            filters = QueryFilters()
            results = self.search("", k=1000, filters=filters, rerank=False)
            
            for result in results:
                if result.document.get('id') == doc_id:
                    return result.document
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document by ID {doc_id}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query system statistics."""
        return {
            'vector_index_stats': self.vector_index.get_stats(),
            'has_numpy': HAS_NUMPY
        }
