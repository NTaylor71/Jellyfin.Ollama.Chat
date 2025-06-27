import faiss
import numpy as np
import os
import uuid

INDEX_FILE = "faiss_index.idx"

class FaissIndex:
    def __init__(self, vector_size=4096):
        self.vector_size = vector_size
        self.index = faiss.IndexFlatL2(vector_size)
        self.id_to_metadata = {}

    def add_vectors(self, vectors: list[dict]):
        ids = []
        embeddings = []

        for item in vectors:
            vec = np.array(item["vector"], dtype="float32")
            embeddings.append(vec)
            vec_id = item.get("id") or str(uuid.uuid4())
            ids.append(vec_id)
            self.id_to_metadata[vec_id] = item.get("metadata", {})

        if embeddings:
            embeddings_np = np.vstack(embeddings)
            self.index.add(embeddings_np)

        return ids

    def search(self, query_vector: list[float], top_k: int = 5):
        if self.index.ntotal == 0:
            return []

        query = np.array([query_vector], dtype="float32")
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # No valid result
            metadata = list(self.id_to_metadata.values())[idx]
            results.append({
                "distance": float(dist),
                "metadata": metadata
            })

        return results

    def save(self, path=INDEX_FILE):
        faiss.write_index(self.index, path)

    def load(self, path=INDEX_FILE):
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            print(f"✅ Loaded FAISS index from {path}")
        else:
            print(f"⚠️ FAISS index file not found at {path} — starting empty.")
