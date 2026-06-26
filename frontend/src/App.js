import React, { useState, useRef, useEffect } from 'react';
import { uploadVideo, pollJobStatus, searchVideo } from './api';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [statusMsg, setStatusMsg] = useState('');
  const [chapters, setChapters] = useState([]);
  const [activeChapter, setActiveChapter] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setVideoFile(file);
    setVideoURL(URL.createObjectURL(file));
    setStatus('idle');
    setStatusMsg('');
    setChapters([]);
    setSearchResults([]);
    setActiveChapter(null);
  };

  const handleUpload = async () => {
    if (!videoFile) return;
    try {
      setStatus('uploading');
      setStatusMsg('Uploading video...');
      const { job_id } = await uploadVideo(videoFile);
      setJobId(job_id);
      setStatus('processing');
      setStatusMsg('Processing: transcribing audio...');
    } catch (err) {
      setStatus('error');
      setStatusMsg(`Upload failed: ${err.message}`);
    }
  };

  // Poll for job completion
  useEffect(() => {
    if (status !== 'processing' || !jobId) return;

    const interval = setInterval(async () => {
      try {
        const job = await pollJobStatus(jobId);
        if (job.status === 'transcribing') {
          setStatusMsg('Processing: transcribing audio...');
        } else if (job.status === 'detecting_chapters') {
          setStatusMsg('Processing: detecting chapter boundaries...');
        } else if (job.status === 'indexing') {
          setStatusMsg('Processing: building search index...');
        } else if (job.status === 'completed') {
          clearInterval(interval);
          setChapters(job.chapters);
          setStatus('done');
          setStatusMsg(`Done! ${job.chapters.length} chapters detected.`);
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setStatus('error');
          setStatusMsg(`Error: ${job.error}`);
        }
      } catch (err) {
        clearInterval(interval);
        setStatus('error');
        setStatusMsg(`Polling failed: ${err.message}`);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [status, jobId]);

  const jumpTo = (startTime, chapterId) => {
    if (videoRef.current) {
      videoRef.current.currentTime = startTime;
      videoRef.current.play();
      setActiveChapter(chapterId);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const data = await searchVideo(searchQuery);
      setSearchResults(data.results);
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setSearching(false);
    }
  };

  const isProcessing = status === 'uploading' || status === 'processing';

  return (
    <div className="app">
      <header>
        <h1>🎬 AI Video Chaptering</h1>
        <p>Upload a video to auto-generate chapters and search inside it</p>
      </header>

      <main>
        {/* Upload Bar */}
        <div className="upload-bar">
          <label className="file-label">
            📁 {videoFile ? videoFile.name : 'Choose a video file'}
            <input type="file" accept="video/*" onChange={handleFileSelect} hidden />
          </label>
          <button onClick={handleUpload} disabled={!videoFile || isProcessing}>
            {isProcessing ? '⏳ Processing...' : '🚀 Analyze Video'}
          </button>
          {statusMsg && (
            <span className={`status-msg ${status}`}>{statusMsg}</span>
          )}
        </div>

        {/* Main Content */}
        {videoURL && (
          <div className="content-grid">

            {/* Left: Video + Search */}
            <div className="left-panel">
              <video ref={videoRef} src={videoURL} controls className="video-player" />

              {status === 'done' && (
                <div className="search-box">
                  <div className="search-input-row">
                    <input
                      type="text"
                      placeholder='Search inside video... e.g. "panic monster"'
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    />
                    <button onClick={handleSearch} disabled={searching}>
                      {searching ? '...' : '🔍'}
                    </button>
                  </div>

                  {searchResults.length > 0 && (
                    <div className="search-results">
                      <p className="results-label">
                        Results for "{searchQuery}"
                      </p>
                      {searchResults.map((r, i) => (
                        <div
                          key={i}
                          className="result-item"
                          onClick={() => jumpTo(r.start, r.chapter_id)}
                        >
                          <span className="ts">{r.timestamp_display}</span>
                          <div className="result-info">
                            <span className="result-title">{r.title}</span>
                            <span className="result-preview">
                              {r.text.slice(0, 80)}...
                            </span>
                          </div>
                          <span className="score">
                            {Math.round(r.similarity_score * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Right: Chapter Sidebar */}
            {chapters.length > 0 && (
              <aside className="sidebar">
                <h2>📚 Chapters <span className="count">{chapters.length}</span></h2>
                <ul>
                  {chapters.map((ch) => (
                    <li
                      key={ch.chapter_id}
                      className={activeChapter === ch.chapter_id ? 'active' : ''}
                      onClick={() => jumpTo(ch.start, ch.chapter_id)}
                    >
                      <span className="ch-time">{ch.timestamp_display}</span>
                      <span className="ch-title">{ch.title}</span>
                    </li>
                  ))}
                </ul>
              </aside>
            )}

          </div>
        )}
      </main>
    </div>
  );
}

export default App;
