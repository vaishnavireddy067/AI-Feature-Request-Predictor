import faiss
import numpy as np

def build_faiss_index(embeddings):
    """
    Builds a FAISS index from a numpy array of embeddings.
    Embeddings are L2-normalized so inner product corresponds to cosine similarity.
    """
    dim = embeddings.shape[1]
    embeddings_norm = embeddings.copy().astype('float32')
    faiss.normalize_L2(embeddings_norm)
    
    # Inner product on normalized vectors = Cosine Similarity (1.0 = identical)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)
    return index

def find_duplicates(index, embeddings, similarity_threshold=0.82):
    """
    Identifies duplicate and near-duplicate feature requests using cosine similarity.
    similarity_threshold: between 0.0 and 1.0 (e.g. 0.82 for high semantic similarity).
    Returns a list of duplicate groups: [{'primary_idx': int, 'duplicate_indices': [int], 'similarity': float}]
    """
    n = len(embeddings)
    if n <= 1:
        return []
        
    k = min(10, n)
    embeddings_norm = embeddings.copy().astype('float32')
    faiss.normalize_L2(embeddings_norm)
    
    sims, indices = index.search(embeddings_norm, k=k)
    
    visited = set()
    duplicate_groups = []
    
    for i in range(n):
        if i in visited:
            continue
            
        group_members = [i]
        best_sim = 1.0
        
        for j in range(k):
            neighbor_idx = int(indices[i][j])
            sim = float(sims[i][j])
            
            # FAISS returns -1 for empty slots, or self
            if neighbor_idx < 0 or neighbor_idx >= n or neighbor_idx == i or neighbor_idx in visited:
                continue
                
            if sim >= similarity_threshold:
                group_members.append(neighbor_idx)
                visited.add(neighbor_idx)
                best_sim = min(best_sim, sim)
                
        if len(group_members) > 1:
            visited.add(i)
            # Deterministic primary index (first chronologically / lowest index)
            primary_idx = min(group_members)
            duplicate_indices = [idx for idx in group_members if idx != primary_idx]
            
            duplicate_groups.append({
                "primary_idx": primary_idx,
                "duplicate_indices": duplicate_indices,
                "all_indices": group_members,
                "count": len(group_members),
                "avg_similarity": round(best_sim, 2)
            })
            
    return duplicate_groups

def semantic_search(query_embedding, index, k=5):
    """
    Searches the FAISS index with a query embedding using cosine similarity.
    Returns cosine similarity scores and indices of top k results.
    """
    q_norm = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(q_norm)
    k_actual = min(k, index.ntotal)
    if k_actual <= 0:
        return [], []
    sims, indices = index.search(q_norm, k=k_actual)
    return sims[0], indices[0]
