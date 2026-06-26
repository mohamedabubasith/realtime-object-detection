import { useEffect, useState } from 'react'
import { updateSettings } from '../api'

// Preferred palette of common COCO classes, shown first (when the model exposes
// them). "car" is the default target.
const PREFERRED = [
  'car',
  'truck',
  'bus',
  'motorcycle',
  'bicycle',
  'person',
  'dog',
  'cat',
]

// Live controls: target classes (primary), confidence threshold, detection
// cadence, stop / new source.
export default function Controls({
  session,
  stats,
  config,
  onStop,
  onReset,
  showRoi,
  onToggleRoi,
}) {
  const [conf, setConf] = useState(stats?.conf_threshold ?? 0.35)
  const [everyN, setEveryN] = useState(config?.process_every_n ?? 2)
  const [classes, setClasses] = useState(stats?.target_classes ?? ['car'])

  // Sync slider with server value when a new session's first stats arrive.
  useEffect(() => {
    if (stats?.conf_threshold != null) setConf(stats.conf_threshold)
  }, [session?.session_id]) // eslint-disable-line react-hooks/exhaustive-deps

  const terminal = ['finished', 'stopped', 'error'].includes(stats?.status)

  async function pushConf(v) {
    setConf(v)
    try {
      await updateSettings(session.session_id, { conf_threshold: v })
    } catch {
      /* non-fatal */
    }
  }

  async function pushEveryN(v) {
    setEveryN(v)
    try {
      await updateSettings(session.session_id, { process_every_n: v })
    } catch {
      /* non-fatal */
    }
  }

  async function toggleClass(name) {
    const next = classes.includes(name)
      ? classes.filter((c) => c !== name)
      : [...classes, name]
    if (next.length === 0) return // keep at least one
    setClasses(next)
    try {
      await updateSettings(session.session_id, { target_classes: next })
    } catch {
      /* non-fatal */
    }
  }

  // Build the chip palette: preferred common classes the model supports, in
  // order, then anything else already selected (so saved targets always show).
  const available = config?.available_classes || ['car']
  const palette = [
    ...PREFERRED.filter((c) => available.includes(c)),
    ...classes.filter((c) => !PREFERRED.includes(c)),
  ]

  return (
    <div className="card controls">
      <h2>Controls</h2>

      <div className="control control-classes">
        <div className="control-head">
          <span>Object classes</span>
          <strong>{classes.length} selected</strong>
        </div>
        <div className="class-toggles">
          {palette.map((c) => (
            <button
              key={c}
              className={`toggle ${classes.includes(c) ? 'on' : ''}`}
              onClick={() => toggleClass(c)}
              disabled={terminal}
            >
              {c}
            </button>
          ))}
        </div>
        <p className="hint">Pick which of the 80 COCO classes to track.</p>
      </div>

      <div className="control">
        <div className="control-head">
          <span>Confidence threshold</span>
          <strong>{(conf * 100).toFixed(0)}%</strong>
        </div>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          value={conf}
          onChange={(e) => pushConf(parseFloat(e.target.value))}
          disabled={terminal}
        />
        <p className="hint">
          Higher = fewer false positives, may miss faint objects.
        </p>
      </div>

      <div className="control">
        <div className="control-head">
          <span>Detect every Nth frame</span>
          <strong>{everyN}</strong>
        </div>
        <input
          type="range"
          min="1"
          max="6"
          step="1"
          value={everyN}
          onChange={(e) => pushEveryN(parseInt(e.target.value))}
          disabled={terminal}
        />
        <p className="hint">Higher = lighter CPU, less frequent box updates.</p>
      </div>

      {onToggleRoi && (
        <div className="control control-roi">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={!!showRoi}
              onChange={(e) => onToggleRoi(e.target.checked)}
              disabled={terminal}
            />
            Show line / zone drawing tools
          </label>
          <p className="hint">
            Draw counting lines and zones directly on the video.
          </p>
        </div>
      )}

      <div className="control-actions">
        {!terminal ? (
          <button className="btn danger" onClick={onStop}>
            Stop Detection
          </button>
        ) : (
          <div className="terminal-note">Session {stats?.status}.</div>
        )}
        <button className="btn ghost" onClick={onReset}>
          New Source
        </button>
      </div>
    </div>
  )
}
