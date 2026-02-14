"""
Grouping functions for building reaction arcs and extending groups.
"""

from typing import List, Optional

from .types import WindowDoc
from .config import ClipConfig


def build_reaction_arcs(
    docs: List[WindowDoc],
    base_groups: List[List[int]],
    max_gap_s: float = 40.0,
    max_arc_dur: float = 180.0,
) -> List[List[int]]:
    """Merge neighbouring groups (sorted by start) into longer arcs.

    We assume *base_groups* are lists of indices already sorted by start.
    Two groups *g* and *h* are merged if:
      • gap between start of *h* and end of *g* ≤ max_gap_s, AND
      • resulting arc duration ≤ max_arc_dur.
    """
    if not base_groups:
        return []

    # Sort groups by their first index start time
    base_groups = sorted(base_groups, key=lambda g: docs[g[0]].start)

    arcs: List[List[int]] = []
    cur = base_groups[0][:]
    for g in base_groups[1:]:
        cur_start = docs[cur[0]].start
        cur_end = docs[cur[-1]].end
        g_start = docs[g[0]].start
        g_end = docs[g[-1]].end

        gap = g_start - cur_end
        new_dur = g_end - cur_start

        if gap <= max_gap_s and new_dur <= max_arc_dur:
            # merge
            cur.extend(g)
        else:
            arcs.append(sorted(cur))
            cur = g[:]
    arcs.append(sorted(cur))
    return arcs


def extend_group_by_thread(docs: List[WindowDoc], group: List[int]) -> List[int]:
    """Expand group on left/right if same topic_thread or same_topic_prev continuity."""
    selected = set(group)
    left = min(group)
    right = max(group)
    
    # Expand left
    i = left - 1
    while i >= 0:
        prev = docs[i]
        cur_first = docs[min(selected)]
        if prev.chapter_id != cur_first.chapter_id:
            break
        same_thread = (prev.topic_thread is not None and cur_first.topic_thread is not None and prev.topic_thread == cur_first.topic_thread)
        if same_thread or prev.same_topic_prev:
            selected.add(i)
            i -= 1
            continue
        break
    
    # Expand right
    i = right + 1
    while i < len(docs):
        nxt = docs[i]
        cur_last = docs[max(selected)]
        if nxt.chapter_id != cur_last.chapter_id:
            break
        same_thread = (nxt.topic_thread is not None and cur_last.topic_thread is not None and nxt.topic_thread == cur_last.topic_thread)
        if same_thread or nxt.same_topic_prev:
            selected.add(i)
            i += 1
            continue
        break
    
    return sorted(selected)


def extend_group_by_semantics(
    docs: List[WindowDoc], 
    group: List[int], 
    retriever, 
    time_window: float = 60.0, 
    sim_thr: float = 0.40
) -> List[int]:
    """Extend group using semantic similarity."""
    if not getattr(retriever, "have_index", False):
        return group
    
    selected = set(group)
    # Compute group temporal bounds
    gstart = min(docs[i].start for i in group)
    gend = max(docs[i].end for i in group)
    
    # Candidates: windows within ±time_window of edges
    for i, d in enumerate(docs):
        if i in selected:
            continue
        if (d.start >= gstart - time_window and d.end <= gend + time_window and d.chapter_id == docs[group[0]].chapter_id):
            # Check similarity vs any selected id
            for j in list(selected):
                if retriever.sim(docs[j].id, d.id) >= sim_thr:
                    selected.add(i)
                    break
    
    return sorted(selected)


def extend_groups(
    docs: List[WindowDoc],
    groups: List[List[int]],
    retriever,
    config: ClipConfig,
    use_semantics: bool = True,
) -> List[List[int]]:
    """Extend groups using thread continuity and optional semantic similarity."""
    extended_groups: List[List[int]] = []
    
    for g in groups:
        # First extend by thread
        g2 = extend_group_by_thread(docs, g)
        
        # Then optionally extend by semantics
        if use_semantics and getattr(retriever, "have_index", False):
            g3 = extend_group_by_semantics(
                docs, g2, retriever, 
                time_window=config.semantic_time_window,
                sim_thr=config.semantic_similarity_threshold
            )
        else:
            g3 = g2
        
        extended_groups.append(g3)
    
    return extended_groups
