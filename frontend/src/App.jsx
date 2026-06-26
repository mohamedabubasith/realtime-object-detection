import { useEffect, useState } from 'react'
import { getConfig, getHealth, stopSession, deleteSession } from './api'
import { useStatsSocket } from './hooks/useStatsSocket'
import SourceSelector from './components/SourceSelector'
import VideoView from './components/VideoView'
import StatsPanel from './components/StatsPanel'
import DetectionLog from './components/DetectionLog'
import Controls from './components/Controls'

export default function App() {
  const [config, setConfig] = useState(null)
  const [online, setOnline] = useState(null)
  const [session, setSession] = useState(null)
  const [showRoi, setShowRoi] = useState(true)
  const { stats, connected } = useStatsSocket(session?.session_id)

  useEffect(() => {
    let alive = true
    const check = async () => {
      try {
        await getHealth()
        const cfg = await getConfig()
        if (alive) {
          setConfig(cfg)
          setOnline(true)
        }
      } catch {
        if (alive) setOnline(false)
      }
    }
    check()
    const id = setInterval(check, 5000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  async function handleStop() {
    if (!session) return
    try {
      await stopSession(session.session_id)
    } catch {
      /* ignore */
    }
  }

  async function handleReset() {
    if (session) {
      try {
        await deleteSession(session.session_id)
      } catch {
        /* ignore */
      }
    }
    setSession(null)
  }

  const statusLabel =
    online === null
      ? 'Connecting…'
      : online
        ? 'Backend online'
        : 'Backend offline'
  const statusClass = online === null ? '' : online ? 'on' : 'off'

  return (
    <div className="app">
      <header className="app-bar">
        <div className="app-bar-inner">
          <div className="brand">
            <span className="logo" aria-hidden="true">🎯</span>
            <div className="brand-text">
              <h1>Object Detection</h1>
              <p className="subtitle">
                CPU-optimized real-time object detection — video, streams &amp;
                live cameras
              </p>
            </div>
          </div>
          <div className={`status-pill ${statusClass}`}>
            <span className="status-led" aria-hidden="true" />
            {statusLabel}
          </div>
        </div>
      </header>

      <div className="page">
        {online === false && (
          <div className="error-banner global">
            Cannot reach the backend. Start it with{' '}
            <code>cd backend &amp;&amp; ./run.sh</code> (default
            http://localhost:8000).
          </div>
        )}

        <main className="layout">
          <section className="main-col">
            <VideoView
              session={session}
              stats={stats}
              live={connected}
              showRoi={showRoi}
            />
            {session && <DetectionLog log={stats?.log} />}
          </section>

          <aside className="side-col">
            {!session ? (
              <SourceSelector onStarted={setSession} busy={!online} />
            ) : (
              <>
                <Controls
                  session={session}
                  stats={stats}
                  config={config}
                  onStop={handleStop}
                  onReset={handleReset}
                  showRoi={showRoi}
                  onToggleRoi={setShowRoi}
                />
                <StatsPanel stats={stats} />
              </>
            )}
          </aside>
        </main>
      </div>

      <footer className="app-footer">
        <span>
          Detection is best-effort and never 100% — tune the confidence
          threshold for your footage.
        </span>
        <span className="footer-model">
          Model: <code>{config?.model_path || '…'}</code>
        </span>
      </footer>
    </div>
  )
}
