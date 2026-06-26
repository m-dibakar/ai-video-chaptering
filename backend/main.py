import os
import uuid
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from transcriber import VideoTranscriber
from chapter_detector import ChapterDetector
from search_engine import VideoSearchEngine

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="AI Video Chaptering API",
    description="Upload a video, get AI chapters, search inside it.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Lazy model loading (load once, reuse)
# ─────────────────────────────────────────────
_transcriber = None
_detector = None
_search_engine = None
_lock = threading.Lock()

def get_models():
    global _transcriber, _detector, _search_engine
    with _lock:
        if _transcriber is None:
            print("Loading models for the first time...")
            _transcriber = VideoTranscriber(model_size="small")
            _detector = ChapterDetector()
            _search_engine = VideoSearchEngine()
    return _transcriber, _detector, _search_engine

# ─────────────────────────────────────────────
# In-memory job store
# ─────────────────────────────────────────────
jobs = {}

# ─────────────────────────────────────────────
# Pydantic models (response shapes)
# ─────────────────────────────────────────────
class JobStatus(BaseModel):
    job_id: str
    status: str
    chapters: Optional[list] = None
    error: Optional[str] = None

# ─────────────────────────────────────────────
# Background processing
# ─────────────────────────────────────────────
def process_video(job_id: str, video_path: str):
    try:
        transcriber, detector, search_engine = get_models()

        jobs[job_id]["status"] = "transcribing"
        segments = transcriber.transcribe(video_path)

        jobs[job_id]["status"] = "detecting_chapters"
        chapters = detector.detect_chapters(segments)

        jobs[job_id]["status"] = "indexing"
        search_engine.build_index(chapters)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["chapters"] = chapters
        print(f"Job {job_id} done: {len(chapters)} chapters")

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"Job {job_id} failed: {e}")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=JobStatus)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    allowed = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format '{ext}'")

    job_id = str(uuid.uuid4())
    video_path = str(UPLOAD_DIR / f"{job_id}{ext}")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "queued", "chapters": None, "error": None}
    background_tasks.add_task(process_video, job_id, video_path)

    return JobStatus(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        chapters=job.get("chapters"),
        error=job.get("error")
    )


@app.get("/search")
def search(q: str, top_k: int = 3):
    _, _, search_engine = get_models()

    if search_engine.index is None:
        raise HTTPException(400, "No video processed yet. Upload a video first.")

    if not q.strip():
        raise HTTPException(400, "Query cannot be empty.")

    results = search_engine.search(q, top_k=min(top_k, 10))
    return {"query": q, "results": results, "count": len(results)}


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
