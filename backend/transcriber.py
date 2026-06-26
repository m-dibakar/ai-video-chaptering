from faster_whisper import WhisperModel
import os

class VideoTranscriber:
    def __init__(self, model_size="small"):
        """
        Load the Whisper model into memory.

        Why "cuda" with fallback to "cpu"?
        - If GPU is available, use it (much faster)
        - If not, fall back to CPU automatically
        """
        device = "cuda" if self._gpu_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        print(f"Loading Whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded.")

    def _gpu_available(self):
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def transcribe(self, video_path: str) -> list[dict]:
        """
        Transcribe a video file and return timestamped segments.

        Returns:
            List of dicts: [{"start": 0.0, "end": 3.2, "text": "..."}, ...]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"Transcribing: {video_path}")

        # faster-whisper returns a generator, not a list
        # We iterate over it to build our list
        segments_generator, info = self.model.transcribe(
            video_path,
            beam_size=5,           # Higher = more accurate but slower
            word_timestamps=False  # We only need segment-level timestamps
        )

        segments = []
        for segment in segments_generator:
            segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

        print(f"Transcription complete: {len(segments)} segments")
        print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

        return segments


# --- TEST THIS MODULE STANDALONE ---
# Run: python3 transcriber.py
if __name__ == "__main__":
    transcriber = VideoTranscriber(model_size="small")

    # Create a test: use any .mp4 file you have
    # For testing, download a short clip from YouTube using yt-dlp
    segments = transcriber.transcribe("test_video1.mp4")

    for seg in segments[:5]:  # Print first 5 segments
        print(f"[{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text']}")