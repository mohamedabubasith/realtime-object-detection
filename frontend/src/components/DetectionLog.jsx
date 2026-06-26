function fmtTime(t) {
  if (!t) return ''
  const d = new Date(t * 1000)
  return d.toLocaleTimeString()
}

export default function DetectionLog({ log = [] }) {
  const entries = [...log].reverse()
  return (
    <div className="card detection-log">
      <div className="card-head">
        <h2>Activity Log</h2>
        {entries.length > 0 && (
          <span className="count-pill">{entries.length}</span>
        )}
      </div>
      <div className="log-list">
        {entries.length === 0 && (
          <div className="muted log-empty">No events yet.</div>
        )}
        {entries.map((e, i) => (
          <div className="log-row" key={`${e.t}-${i}`}>
            <span className="log-time">{fmtTime(e.t)}</span>
            <span className="log-msg">{e.message}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
