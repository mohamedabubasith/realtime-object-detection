function Stat({ label, value, sub }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

// Derive a per-class breakdown purely client-side from the live detections list
// (group by .cls, count). No backend change required.
function breakdown(detections = []) {
  const counts = new Map()
  for (const d of detections) {
    if (!d || d.cls == null) continue
    counts.set(d.cls, (counts.get(d.cls) || 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])
}

export default function StatsPanel({ stats }) {
  const s = stats || {}
  const pct = (v) => (v ? `${(v * 100).toFixed(0)}%` : '—')
  const perClass = breakdown(s.detections)
  const maxCount = perClass.length ? perClass[0][1] : 0

  const hasUnique = s.unique_total != null || s.unique_active != null
  const lineCounts = s.line_counts || []
  const zoneOcc = s.zone_occupancy || []

  return (
    <div className="card stats-panel">
      <h2>Live Stats</h2>

      {hasUnique && (
        <div className="unique-row">
          <div className="unique-cell">
            <div className="unique-value">{s.unique_total ?? 0}</div>
            <div className="unique-label">Unique total</div>
          </div>
          <div className="unique-cell">
            <div className="unique-value">{s.unique_active ?? 0}</div>
            <div className="unique-label">Active now</div>
          </div>
        </div>
      )}

      <div className="stat-grid">
        <Stat
          label="Detected now"
          value={s.current_count ?? 0}
          sub={s.smoothed_count != null ? `smoothed ${s.smoothed_count}` : null}
        />
        <Stat label="Peak" value={s.max_count ?? 0} />
        <Stat label="FPS" value={s.fps ?? 0} />
        <Stat
          label="Confidence"
          value={pct(s.last_confidence)}
          sub={s.avg_confidence ? `avg ${pct(s.avg_confidence)}` : null}
        />
        <Stat label="Frames" value={s.processed_frames ?? 0} />
        <Stat
          label="Elapsed"
          value={s.elapsed_seconds != null ? `${s.elapsed_seconds}s` : '—'}
        />
        <Stat
          label="Resolution"
          value={s.width ? `${s.width}×${s.height}` : '—'}
        />
        <Stat label="Total hits" value={s.total_detections ?? 0} />
      </div>

      <div className="breakdown">
        <div className="breakdown-head">
          <span>In this frame</span>
          <span className="muted">by class</span>
        </div>
        {perClass.length === 0 ? (
          <p className="muted breakdown-empty">No objects in the current frame.</p>
        ) : (
          <ul className="breakdown-list">
            {perClass.map(([cls, n]) => (
              <li key={cls} className="breakdown-row">
                <span className="breakdown-name">{cls}</span>
                <span className="breakdown-bar-track">
                  <span
                    className="breakdown-bar"
                    style={{ width: `${maxCount ? (n / maxCount) * 100 : 0}%` }}
                  />
                </span>
                <span className="breakdown-count">{n}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {lineCounts.length > 0 && (
        <div className="roi-stats">
          <div className="roi-stats-head">
            <span>Line crossings</span>
            <span className="muted">in / out</span>
          </div>
          <ul className="roi-stats-list">
            {lineCounts.map((lc) => (
              <li key={lc.id} className="roi-stats-row">
                <span className="roi-stats-name" title={lc.name}>
                  {lc.name}
                </span>
                <span className="roi-stats-vals">
                  <span className="roi-pill in">IN {lc.in ?? 0}</span>
                  <span className="roi-pill out">OUT {lc.out ?? 0}</span>
                  <span className="roi-stats-total">{lc.total ?? 0}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {zoneOcc.length > 0 && (
        <div className="roi-stats">
          <div className="roi-stats-head">
            <span>Zone occupancy</span>
            <span className="muted">now / peak · dwell</span>
          </div>
          <ul className="roi-stats-list">
            {zoneOcc.map((zo) => (
              <li key={zo.id} className="roi-stats-row">
                <span className="roi-stats-name" title={zo.name}>
                  {zo.name}
                </span>
                <span className="roi-stats-vals">
                  <span className="roi-pill occ">{zo.count ?? 0}</span>
                  <span className="roi-stats-sub">peak {zo.peak ?? 0}</span>
                  {zo.avg_dwell_s != null && (
                    <span className="roi-stats-sub">
                      {Number(zo.avg_dwell_s).toFixed(1)}s
                    </span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {s.target_classes?.length > 0 && (
        <div className="targets">
          <span className="targets-label">Targets</span>
          <div className="chips">
            {s.target_classes.map((c) => (
              <span key={c} className="chip">{c}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
