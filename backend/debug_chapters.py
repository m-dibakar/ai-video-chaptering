import numpy as np
import ruptures as rpt
from sentence_transformers import SentenceTransformer
from transcriber import VideoTranscriber

# Transcribe
transcriber = VideoTranscriber(model_size="small")
segments = transcriber.transcribe("test_video1.mp4")

# Group into chunks
chunks = []
window_size = 5
for i in range(0, len(segments), window_size):
    window = segments[i:i + window_size]
    chunks.append(" ".join([s["text"] for s in window]))

print(f"Total chunks: {len(chunks)}")

# Embed
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Try different penalty values and see how many chapters each gives
print("\nPenalty → Chapters detected:")
print("-" * 30)
algo = rpt.Pelt(model="rbf").fit(embeddings)
for pen in [0.5, 1, 2, 3, 5, 10, 15, 20]:
    bkps = algo.predict(pen=pen)
    print(f"  pen={pen:5} → {len(bkps)} chapters")
