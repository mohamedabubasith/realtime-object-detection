import { useRef, useState } from 'react'
import { uploadFile, createSession } from '../api'

const TABS = [
  { id: 'upload', label: 'Upload Video', icon: '⬆' },
  { id: 'url', label: 'Stream / YouTube', icon: '▶' },
  { id: 'rtsp', label: 'RTSP Camera', icon: '🎥' },
  { id: 'webcam', label: 'Webcam', icon: '📷' },
]

export default function SourceSelector({ onStarted, busy }) {
  const [tab, setTab] = useState('upload')
  const [url, setUrl] = useState('')
  const [rtsp, setRtsp] = useState('')
  const [webcam, setWebcam] = useState('0')
  const [loop, setLoop] = useState(true)
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const [starting, setStarting] = useState(false)
  const fileInput = useRef(null)

  const disabled = busy || starting

  async function start() {
    setError(null)
    setStarting(true)
    try {
      let payload
      if (tab === 'upload') {
        if (!file) throw new Error('Choose a video file first')
        setProgress(0.001)
        const up = await uploadFile(file, setProgress)
        payload = {
          source_type: 'upload',
          source: up.file_id,
          label: up.filename,
          loop,
        }
      } else if (tab === 'url') {
        if (!url.trim()) throw new Error('Paste a stream / YouTube URL')
        payload = { source_type: 'url', source: url.trim() }
      } else if (tab === 'rtsp') {
        if (!rtsp.trim()) throw new Error('Enter an rtsp:// URL')
        payload = { source_type: 'rtsp', source: rtsp.trim() }
      } else {
        payload = { source_type: 'webcam', source: String(webcam) }
      }
      const session = await createSession(payload)
      onStarted(session)
    } catch (e) {
      setError(e.message)
    } finally {
      setStarting(false)
      setProgress(0)
    }
  }

  return (
    <div className="card source-selector">
      <h2>Start Detection</h2>
      <p className="card-sub">
        Choose a source — bounding boxes and live stats appear instantly.
      </p>
      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
            disabled={disabled}
          >
            <span className="tab-icon" aria-hidden="true">{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>

      <div className="tab-body">
        {tab === 'upload' && (
          <div className="field">
            <label>Video file</label>
            <div
              className={`dropzone ${file ? 'has-file' : ''}`}
              onClick={() => fileInput.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault()
                if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0])
              }}
            >
              <input
                ref={fileInput}
                type="file"
                accept="video/*"
                hidden
                onChange={(e) => setFile(e.target.files[0] || null)}
              />
              {file ? (
                <span>{file.name} · {(file.size / 1e6).toFixed(1)} MB</span>
              ) : (
                <span>Click or drop a video here (mp4, mov, mkv, …)</span>
              )}
            </div>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={loop}
                onChange={(e) => setLoop(e.target.checked)}
              />
              Loop video when it ends
            </label>
            {progress > 0 && (
              <div className="progress">
                <div className="progress-bar" style={{ width: `${progress * 100}%` }} />
              </div>
            )}
          </div>
        )}

        {tab === 'url' && (
          <div className="field">
            <label>YouTube / HLS (.m3u8) / direct video URL</label>
            <input
              type="text"
              placeholder="https://www.youtube.com/watch?v=…"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={disabled}
            />
            <p className="hint">
              Works with YouTube links, live streams, and .m3u8 / .mp4 URLs.
            </p>
          </div>
        )}

        {tab === 'rtsp' && (
          <div className="field">
            <label>RTSP camera URL</label>
            <input
              type="text"
              placeholder="rtsp://user:pass@192.168.1.50:554/stream"
              value={rtsp}
              onChange={(e) => setRtsp(e.target.value)}
              disabled={disabled}
            />
            <p className="hint">IP cameras / NVRs. TCP transport is used automatically.</p>
          </div>
        )}

        {tab === 'webcam' && (
          <div className="field">
            <label>Webcam device index</label>
            <input
              type="number"
              min="0"
              value={webcam}
              onChange={(e) => setWebcam(e.target.value)}
              disabled={disabled}
            />
            <p className="hint">
              Reads the camera on the machine running the backend (0 = default).
            </p>
          </div>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <button className="btn primary" onClick={start} disabled={disabled}>
        {starting ? 'Starting…' : 'Start Detection'}
      </button>
    </div>
  )
}
