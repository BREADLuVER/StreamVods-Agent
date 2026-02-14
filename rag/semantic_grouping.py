#!/usr/bin/env python3
"""
Semantic Grouping for RAG-based Content Organization

This module uses vector similarity to group semantically related content
before applying director cut selection, preventing conversation fragmentation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from rag.retrieval import load_retriever, Retriever


def compute_semantic_similarity_matrix(bursts: List[Dict], retriever: Retriever) -> np.ndarray:
    """
    Compute pairwise semantic similarity matrix for all bursts.
    
    Args:
        bursts: List of burst data
        retriever: Vector similarity retriever
        
    Returns:
        NxN similarity matrix
    """
    n = len(bursts)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = retriever.sim(bursts[i]["id"], bursts[j]["id"])
    
    return sim_matrix


def find_semantic_clusters(bursts: List[Dict], retriever: Retriever, 
                          similarity_threshold: float = 0.4,
                          min_cluster_size: int = 2) -> List[List[int]]:
    """
    Find semantic clusters of related content using vector similarity.
    
    Args:
        bursts: List of burst data
        retriever: Vector similarity retriever
        similarity_threshold: Minimum similarity for cluster membership
        min_cluster_size: Minimum size for a valid cluster
        
    Returns:
        List of clusters, each containing burst indices
    """
    if not retriever.have_index or len(bursts) < 2:
        return []
    
    sim_matrix = compute_semantic_similarity_matrix(bursts, retriever)
    n = len(bursts)
    visited = set()
    clusters = []
    
    for i in range(n):
        if i in visited:
            continue
            
        # Start new cluster from burst i
        cluster = [i]
        visited.add(i)
        
        # Find all bursts similar to this one
        for j in range(i + 1, n):
            if j in visited:
                continue
                
            # Check if burst j is similar to any burst in current cluster
            is_similar = any(sim_matrix[j, k] >= similarity_threshold for k in cluster)
            
            if is_similar:
                cluster.append(j)
                visited.add(j)
        
        # Only keep clusters with minimum size
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    
    return clusters


def merge_semantic_clusters(bursts: List[Dict], clusters: List[List[int]]) -> List[Dict]:
    """
    Merge bursts within semantic clusters to create coherent segments.
    
    Args:
        bursts: List of burst data
        clusters: List of cluster indices
        
    Returns:
        Updated bursts with semantic grouping information
    """
    updated_bursts = []
    
    # Create cluster mapping
    burst_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for burst_idx in cluster:
            burst_to_cluster[burst_idx] = cluster_id
    
    for i, burst in enumerate(bursts):
        # Add semantic cluster information
        burst["semantic_cluster"] = burst_to_cluster.get(i, -1)
        burst["cluster_size"] = len(clusters[burst_to_cluster[i]]) if burst["semantic_cluster"] >= 0 else 1
        
        updated_bursts.append(burst)
    
    return updated_bursts


def enhanced_context_wrap_semantic(ch_bursts: List[Dict], keep: List[bool], 
                                 retriever: Retriever, 
                                 similarity_threshold: float = 0.35) -> None:
    """
    Enhanced context wrapping that uses semantic similarity to maintain coherence.
    
    Args:
        ch_bursts: List of burst data
        keep: Boolean list indicating which bursts to keep
        retriever: Vector similarity retriever
        similarity_threshold: Minimum similarity for context inclusion
    """
    if not retriever.have_index:
        return
    
    n = len(ch_bursts)
    
    for i in range(n):
        if not keep[i]:
            continue
        
        current_burst = ch_bursts[i]
        
        # Look backwards for semantically similar content
        j = i - 1
        while j >= 0:
            prev_burst = ch_bursts[j]
            
            # Check semantic similarity
            similarity = retriever.sim(current_burst["id"], prev_burst["id"])
            
            if similarity >= similarity_threshold:
                keep[j] = True
                # Continue looking backwards from this point
                j -= 1
            else:
                # If not similar, check if we should stop based on time gap
                time_gap = current_burst["start_time"] - prev_burst["end_time"]
                if time_gap > 60.0:  # 1 minute gap threshold
                    break
                j -= 1
        
        # Look forwards for semantically similar content
        j = i + 1
        while j < n:
            next_burst = ch_bursts[j]
            
            # Check semantic similarity
            similarity = retriever.sim(current_burst["id"], next_burst["id"])
            
            if similarity >= similarity_threshold:
                keep[j] = True
                # Continue looking forwards from this point
                j += 1
            else:
                # If not similar, check if we should stop based on time gap
                time_gap = next_burst["start_time"] - current_burst["end_time"]
                if time_gap > 60.0:  # 1 minute gap threshold
                    break
                j += 1


def enhanced_peak_blocks_semantic(ch_bursts: List[Dict], keep: List[bool], 
                                 retriever: Retriever,
                                 similarity_threshold: float = 0.4) -> List[Optional[str]]:
    """
    Enhanced peak block labeling that groups by semantic similarity.
    
    Args:
        ch_bursts: List of burst data
        keep: Boolean list indicating which bursts to keep
        retriever: Vector similarity retriever
        similarity_threshold: Minimum similarity for block membership
        
    Returns:
        List of peak block IDs
    """
    if not retriever.have_index:
        # Fallback to simple time-based grouping
        return _fallback_peak_blocks(ch_bursts, keep)
    
    block_ids: List[Optional[str]] = [None] * len(ch_bursts)
    block_counter = 0
    
    i = 0
    while i < len(ch_bursts):
        if not keep[i]:
            i += 1
            continue
        
        # Start new block
        block_counter += 1
        block_id = f"PB-{block_counter:03d}"
        block_ids[i] = block_id
        
        # Find all semantically similar bursts
        j = i + 1
        while j < len(ch_bursts) and keep[j]:
            # Check semantic similarity to any burst already in this block
            is_similar = False
            for k in range(i, j):
                if keep[k] and block_ids[k] == block_id:
                    similarity = retriever.sim(ch_bursts[k]["id"], ch_bursts[j]["id"])
                    if similarity >= similarity_threshold:
                        is_similar = True
                        break
            
            if is_similar:
                block_ids[j] = block_id
                j += 1
            else:
                # Check time gap as fallback
                time_gap = ch_bursts[j]["start_time"] - ch_bursts[j-1]["end_time"]
                if time_gap <= 30.0:  # 30 second gap threshold
                    block_ids[j] = block_id
                    j += 1
                else:
                    break
        
        i = j
    
    return block_ids


def _fallback_peak_blocks(ch_bursts: List[Dict], keep: List[bool]) -> List[Optional[str]]:
    """Fallback peak block labeling when retriever is not available."""
    block_ids: List[Optional[str]] = [None] * len(ch_bursts)
    block_counter = 0
    
    i = 0
    while i < len(ch_bursts):
        if not keep[i]:
            i += 1
            continue
        
        block_counter += 1
        block_id = f"PB-{block_counter:03d}"
        
        j = i
        while j < len(ch_bursts) and keep[j]:
            block_ids[j] = block_id
            j += 1
        
        i = j
    
    return block_ids


def analyze_semantic_coherence(bursts: List[Dict], retriever: Retriever) -> Dict[str, any]:
    """
    Analyze the semantic coherence of the content.
    
    Args:
        bursts: List of burst data
        retriever: Vector similarity retriever
        
    Returns:
        Analysis results
    """
    if not retriever.have_index:
        return {"error": "No retriever available"}
    
    # Find semantic clusters
    clusters = find_semantic_clusters(bursts, retriever)
    
    analysis = {
        "total_clusters": len(clusters),
        "cluster_sizes": [len(cluster) for cluster in clusters],
        "average_cluster_size": np.mean([len(cluster) for cluster in clusters]) if clusters else 0,
        "coherence_score": 0.0
    }
    
    # Calculate coherence score based on cluster quality
    if clusters:
        # Coherence = (number of well-formed clusters) / (total possible clusters)
        well_formed_clusters = sum(1 for cluster in clusters if len(cluster) >= 3)
        analysis["coherence_score"] = well_formed_clusters / len(clusters)
    
    return analysis


def main():
    """Test the semantic grouping system."""
    import json
    from pathlib import Path
    
    # Load test data
    data_file = Path("data/vector_stores/2565828611/exported_all_documents.json")
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        bursts = data.get("documents", [])
        
        # Load retriever
        retriever = load_retriever("2565828611")
        
        # Analyze semantic coherence
        analysis = analyze_semantic_coherence(bursts, retriever)
        
        print("Semantic Grouping Analysis:")
        print(f"Total clusters: {analysis.get('total_clusters', 0)}")
        print(f"Average cluster size: {analysis.get('average_cluster_size', 0):.2f}")
        print(f"Coherence score: {analysis.get('coherence_score', 0):.3f}")
        
        # Show cluster sizes
        cluster_sizes = analysis.get('cluster_sizes', [])
        if cluster_sizes:
            print(f"Cluster sizes: {sorted(cluster_sizes, reverse=True)}")


if __name__ == "__main__":
    main()
