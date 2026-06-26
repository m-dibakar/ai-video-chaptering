import numpy as np
import ruptures as rpt
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class ChapterDetector:
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        print(f"Loading embedding model...")
        self.model = SentenceTransformer(embed_model)
        print("Embedding model loaded.")
    
    def _group_segments(self, segments: List[Dict], window_size: int = 5) -> List[Dict]:
        chunks = []
        for i in range(0, len(segments), window_size):
            window = segments[i:i + window_size]
            chunks.append({
                "start": window[0]["start"],
                "end": window[-1]["end"],
                "text": " ".join([s["text"] for s in window])
            })
        return chunks
    
    def _embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def _generate_title(self, text: str) -> str:
        stopwords = {
            "the","a","an","is","are","was","were","be","been","being","have",
            "has","had","do","does","did","will","would","could","should","may",
            "might","i","we","you","he","she","it","they","me","us","him","her",
            "them","my","our","your","his","its","their","this","that","these",
            "those","and","but","or","so","at","by","in","of","on","to","up",
            "as","if","into","about","after","before","with","from","also",
            "just","like","very","really","well","now","then","what","when",
            "where","who","how","all","get","got","going","know","think","said"
        }
        words = text.lower().split()
        word_freq = {}
        for word in words:
            clean = ''.join(c for c in word if c.isalpha())
            if clean and len(clean) > 3 and clean not in stopwords:
                word_freq[clean] = word_freq.get(clean, 0) + 1
        
        if not word_freq:
            return "Chapter Content"
        
        top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:4]
        return " ".join(w.title() for w in top_words)
    
    def _find_best_penalty(self, embeddings: np.ndarray) -> float:
        """
        Automatically find a penalty that gives a reasonable number of chapters.
        
        Target: roughly 1 chapter per 3-5 minutes of content.
        We try penalties from low to high and pick the first one
        that gives <= max_chapters.
        """
        n_chunks = len(embeddings)
        
        # Target between 3 and 12 chapters regardless of video length
        min_chapters = 3
        max_chapters = min(12, max(3, n_chunks // 3))
        
        algo = rpt.Pelt(model="rbf").fit(embeddings)
        
        penalties = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
        best_pen = 0.5  # fallback
        
        print(f"\nAuto-tuning penalty (target: {min_chapters}-{max_chapters} chapters):")
        for pen in penalties:
            bkps = algo.predict(pen=pen)
            n = len(bkps)
            print(f"  pen={pen} → {n} chapters")
            if min_chapters <= n <= max_chapters:
                best_pen = pen
                print(f"  ✓ Selected pen={pen} ({n} chapters)")
                break
        
        return best_pen
    
    def detect_chapters(self, segments: List[Dict], window_size: int = 5) -> List[Dict]:
        print(f"\nTotal segments: {len(segments)}")
        
        # Step 1: Group
        chunks = self._group_segments(segments, window_size=window_size)
        print(f"Grouped into {len(chunks)} chunks")
        
        # Step 2: Embed
        print("Generating embeddings...")
        embeddings = self._embed_chunks(chunks)
        
        # Step 3: Auto-tune penalty and detect breakpoints
        best_pen = self._find_best_penalty(embeddings)
        algo = rpt.Pelt(model="rbf").fit(embeddings)
        breakpoints = algo.predict(pen=best_pen)
        
        # Step 4: Build chapters
        chapter_starts = [0] + breakpoints[:-1]
        chapter_ends = breakpoints
        
        chapters = []
        for idx, (start_idx, end_idx) in enumerate(zip(chapter_starts, chapter_ends)):
            chapter_chunks = chunks[start_idx:end_idx]
            if not chapter_chunks:
                continue
            
            full_text = " ".join([c["text"] for c in chapter_chunks])
            title = self._generate_title(full_text)
            start_time = chapter_chunks[0]["start"]
            end_time = chapter_chunks[-1]["end"]
            m, s = int(start_time // 60), int(start_time % 60)
            
            chapters.append({
                "chapter_id": idx + 1,
                "start": start_time,
                "end": end_time,
                "title": title,
                "timestamp_display": f"{m:02d}:{s:02d}",
                "text": full_text
            })
        
        return chapters


if __name__ == "__main__":
    import sys
    from transcriber import VideoTranscriber
    
    video = sys.argv[1] if len(sys.argv) > 1 else "test_video1.mp4"
    
    transcriber = VideoTranscriber(model_size="small")
    segments = transcriber.transcribe(video)
    
    detector = ChapterDetector()
    chapters = detector.detect_chapters(segments)
    
    print(f"\n{'='*50}")
    print(f"CHAPTERS DETECTED: {len(chapters)}")
    print(f"{'='*50}")
    for ch in chapters:
        print(f"\nChapter {ch['chapter_id']}: [{ch['timestamp_display']}] {ch['title']}")
        print(f"  {ch['start']:.1f}s → {ch['end']:.1f}s")
        print(f"  Preview: {ch['text'][:120]}...")
