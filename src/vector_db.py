import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any

class VectorDB:
    def __init__(self, dim=768, db_path="vector_db.index", meta_path="vector_db_meta.json"):
        self.dim = dim
        self.db_path = db_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)
        self.meta = []
        
        # Load existing index and metadata if available
        try:
            if os.path.exists(db_path):
                self.index = faiss.read_index(db_path)
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing vector DB: {e}")
            # Reset to empty state
            self.index = faiss.IndexFlatL2(dim)
            self.meta = []

    def add(self, embeddings: List[np.ndarray], meta_list: List[Dict[str, Any]]):
        """Add embeddings and metadata to the vector database"""
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.meta.extend(meta_list)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k=5):
        """Search for similar embeddings"""
        if self.index.ntotal == 0:
            return []
        
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.meta) and distances[0][i] < float('inf'):
                result = self.meta[idx].copy()
                result['similarity_score'] = float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity
                results.append(result)
        
        return results

    def save(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, self.db_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving vector DB: {e}")

    def get_stats(self):
        """Get database statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dim,
            "metadata_count": len(self.meta)
        }

    def clear(self):
        """Clear all data from the vector database"""
        self.index = faiss.IndexFlatL2(self.dim)
        self.meta = []
        self.save()