# AI Video Chaptering & Semantic Search Engine

Upload any video and get AI-generated chapters with natural language search — built to demonstrate the same capabilities as Muvi's Alie AI engine.

## What it does
- **Auto-transcription** — Whisper (faster-whisper) converts speech to timestamped text
- **Chapter detection** — Sentence embeddings + change-point detection (ruptures) finds topic shifts
- **Semantic search** — FAISS vector index lets you search inside video with natural language
- **REST API** — FastAPI backend with async background processing
- **React UI** — Video player with clickable chapter sidebar and search bar

## Tech Stack
| Layer | Tools |
|-------|-------|
| Transcription | faster-whisper (Whisper small, CUDA) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (IndexFlatIP, cosine similarity) |
| Chapter Detection | ruptures (PELT algorithm, RBF kernel) |
| Backend | FastAPI + Uvicorn |
| Frontend | React |

## Run locally

### Backend
```bash
conda activate ai-video-chaptering
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

Open `http://localhost:3000` — upload any `.mp4` and click Analyze Video.

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status check |
| POST | `/upload` | Upload video, returns job_id |
| GET | `/jobs/{job_id}` | Poll processing status |
| GET | `/search?q=query` | Semantic search across chapters |

## How chapter detection works
Each chapter is a group of transcript segments converted to a 384-dim vector via MiniLM. The PELT algorithm detects change-points in the embedding sequence — positions where cosine distance between adjacent windows jumps, indicating a topic shift. Penalty is auto-tuned to produce 3-12 chapters per video.

## Publications
This project was built alongside IEEE research in deep learning and semantic segmentation.
- IEEE TENCON 2025 — Attentive Depth-Mapped Dice Loss
- IEEE Signal Processing Letters — Variance Stabilized Loss Function
