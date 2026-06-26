import { useEffect, useRef, useState } from 'react'
import { videoUrl } from '../api'
import RoiOverlay from './RoiOverlay'

// Displays the annotated MJPEG stream in an <img>. Adds a cache-busting key so a
// fresh session reloads, and an overlay with the live object count.
export default function VideoView({ session, stats, live, showRoi = true }) {
  const [reloadKey, setReloadKey] = useState(0)
  const [loaded, setLoaded] = useState(false)
  const [errored, setErrored] = useState(false)
  const imgRef = useRef(null)

  useEffect(() => {
    setLoaded(false)
    setErrored(false)
    setReloadKey((k) => k + 1)
  }, [session?.session_id])

  // CRITICAL: an MJPEG <img> holds its HTTP connection open forever. If we don't
  // clear src when the session changes or the component unmounts, the browser
  // leaks connections/memory. Clear it explicitly on cleanup.
  useEffect(() => {
    const node = imgRef.current
    return () => {
      if (node) node.src = ''
    }
  }, [session?.session_id])

  if (!session) {
    return (
      <div className="card video-view placeholder">
        <div className="placeholder-inner">
          <div className="placeholder-icon" aria-hidden="true">🎯</div>
          <p className="placeholder-title">No active detection</p>
          <p className="muted">
            Pick a source on the right to start drawing live bounding boxes.
          </p>
        </div>
      </div>
    )
  }

  const count = stats?.current_count ?? 0
  const status = stats?.status || session.status
  const failed = status === 'error' || errored
  const errorText =
    stats?.error ||
    'Stream unavailable — the source may have ended or failed to open.'
  // Only overlay drawing tools when the stream is live and the session is
  // actively running (not finished / stopped / errored).
  const running = !['finished', 'stopped', 'error'].includes(status)
  const roiActive = showRoi && loaded && !errored && running

  return (
    <div className="card video-view">
      <div className="video-frame">
        {!loaded && !failed && (
          <div className="video-overlay center">
            <span className="spinner" aria-hidden="true" />
            {status === 'starting' ? 'Resolving source…' : 'Connecting to stream…'}
          </div>
        )}
        {failed && (
          <div className="video-overlay center error">
            <strong>Couldn’t start detection</strong>
            <span className="error-detail">{errorText}</span>
          </div>
        )}
        <img
          key={reloadKey}
          ref={imgRef}
          src={`${videoUrl(session.session_id)}?t=${reloadKey}`}
          alt="annotated detection stream"
          onLoad={() => setLoaded(true)}
          onError={() => setErrored(true)}
        />
        {roiActive && (
          <RoiOverlay
            sessionId={session.session_id}
            stats={stats}
            imgRef={imgRef}
          />
        )}
        {loaded && (
          <>
            <div className={`badge count ${count > 0 ? 'active' : ''}`}>
              <span aria-hidden="true">🎯</span> {count} object
              {count === 1 ? '' : 's'}
            </div>
            <div className={`badge status ${status}`}>
              <span className={`status-led ${live ? 'on' : ''}`} aria-hidden="true" />
              {status}
            </div>
          </>
        )}
      </div>
      <div className="video-caption" title={session.source_label}>
        {session.source_label}
      </div>
    </div>
  )
}
