import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class VideoSearchEngine:
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        print("Loading search engine...")
        self.model = SentenceTransformer(embed_model)
        self.index = None
        self.chapters = []
        self.dimension = 384
        print("Search engine ready.")
    
    def build_index(self, chapters: List[Dict]) -> None:
        if not chapters:
            raise ValueError("No chapters to index")
        
        self.chapters = chapters
        texts = [ch["text"] for ch in chapters]
        
        print(f"Indexing {len(chapters)} chapters...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build flat index (exact search)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        print(f"Index built with {self.index.ntotal} vectors.")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Embed and normalize the query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        actual_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(
            query_embedding.astype(np.float32),
            actual_k
        )
        
        # Build results
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chapter = self.chapters[idx].copy()
            chapter["similarity_score"] = round(float(score), 4)
            results.append(chapter)
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results


if __name__ == "__main__":
    from transcriber import VideoTranscriber
    from chapter_detector import ChapterDetector

    # Step 1: Transcribe
    transcriber = VideoTranscriber(model_size="small")
    segments = transcriber.transcribe("test_video1.mp4")

    # Step 2: Detect chapters
    detector = ChapterDetector()
    chapters = detector.detect_chapters(segments)

    # Step 3: Build search index
    engine = VideoSearchEngine()
    engine.build_index(chapters)

    # Step 4: Interactive search loop
    print("\n" + "="*50)
    print("SEARCH READY — type a query, or 'quit' to exit")
    print("="*50)

    while True:
        query = input("\n🔍 Search: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        results = engine.search(query, top_k=3)
        print(f"\nTop results for '{query}':")
        for i, r in enumerate(results, 1):
            score_pct = int(r['similarity_score'] * 100)
            print(f"  {i}. [{r['timestamp_display']}] {r['title']} — {score_pct}% match")
            print(f"     {r['text'][:100]}...")
