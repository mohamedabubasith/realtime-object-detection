import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { updateSettings } from '../api'

// Interactive line/zone drawing layer rendered over the annotated MJPEG <img>.
//
// Coordinates are stored NORMALIZED to the VIDEO CONTENT rectangle (the actual
// rendered video pixels, in [0,1]). The <img> uses object-fit: contain, so when
// the video aspect != the element aspect there are letterbox bars. We compute
// the real content rect from naturalWidth/naturalHeight vs clientWidth/client
// Height and map mouse <-> normalized coords through it, keeping shapes aligned
// with the backend's frame coordinates.

const DEBOUNCE_MS = 400

// ---- geometry helpers ----------------------------------------------------

// The rendered video content rect inside the <img> element (object-fit:contain).
// Returns {left, top, width, height} in element-local CSS pixels, or null if the
// image hasn't reported its natural size yet.
function contentRect(img) {
  if (!img) return null
  const ew = img.clientWidth
  const eh = img.clientHeight
  const nw = img.naturalWidth
  const nh = img.naturalHeight
  if (!ew || !eh) return null
  if (!nw || !nh) {
    // No natural size yet (stream not loaded): assume the image fills the box.
    return { left: 0, top: 0, width: ew, height: eh }
  }
  const elAspect = ew / eh
  const vidAspect = nw / nh
  let width, height
  if (vidAspect > elAspect) {
    // video is wider -> full width, letterbox top/bottom
    width = ew
    height = ew / vidAspect
  } else {
    // video is taller -> full height, pillarbox left/right
    height = eh
    width = eh * vidAspect
  }
  return {
    left: (ew - width) / 2,
    top: (eh - height) / 2,
    width,
    height,
  }
}

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v)

// signed side of point P relative to directed line A->B
function side(a, b, p) {
  return Math.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
}

// ---- component -----------------------------------------------------------

export default function RoiOverlay({ sessionId, stats, imgRef }) {
  const [lines, setLines] = useState([])
  const [zones, setZones] = useState([])
  const [tool, setTool] = useState('idle') // idle | line | zone | select
  const [draft, setDraft] = useState([]) // points (normalized) of in-progress shape
  const [hover, setHover] = useState(null) // normalized cursor while drawing
  const [selected, setSelected] = useState(null) // {kind:'line'|'zone', id}
  const [rect, setRect] = useState(null) // content rect in CSS px
  const [editing, setEditing] = useState(null) // {kind, id} currently renaming

  const wrapRef = useRef(null)
  const debounceRef = useRef(null)
  const dirtyRef = useRef(false) // true once the user has edited locally
  const counterRef = useRef({ l: 0, z: 0 })

  // -- recompute content rect on resize / image load ----------------------
  const recompute = useCallback(() => {
    setRect(contentRect(imgRef?.current))
  }, [imgRef])

  useLayoutEffect(() => {
    recompute()
    const wrap = wrapRef.current
    const img = imgRef?.current
    if (!wrap) return
    const ro = new ResizeObserver(recompute)
    ro.observe(wrap)
    if (img) ro.observe(img)
    window.addEventListener('resize', recompute)
    // The MJPEG <img> swaps frames continuously; naturalWidth becomes known on
    // first load. Poll briefly until we have a rect with non-zero size.
    let tries = 0
    const poll = setInterval(() => {
      const r = contentRect(imgRef?.current)
      if (r && r.width > 0) {
        setRect(r)
        if (++tries > 20) clearInterval(poll)
      }
      if (++tries > 40) clearInterval(poll)
    }, 150)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', recompute)
      clearInterval(poll)
    }
  }, [recompute, imgRef])

  // -- initialize from server echo (only before the user starts editing) --
  useEffect(() => {
    if (dirtyRef.current) return
    if (stats?.lines) setLines(stats.lines)
    if (stats?.zones) setZones(stats.zones)
    // keep id counters ahead of any restored ids so new ids never collide
    const maxNum = (arr, prefix) =>
      arr.reduce((m, s) => {
        const n = parseInt(String(s.id).replace(prefix, ''), 10)
        return Number.isFinite(n) ? Math.max(m, n) : m
      }, 0)
    counterRef.current = {
      l: Math.max(counterRef.current.l, maxNum(stats?.lines || [], 'l')),
      z: Math.max(counterRef.current.z, maxNum(stats?.zones || [], 'z')),
    }
  }, [stats?.lines, stats?.zones])

  // -- debounced push to backend ------------------------------------------
  const scheduleSave = useCallback(
    (nextLines, nextZones) => {
      dirtyRef.current = true
      if (debounceRef.current) clearTimeout(debounceRef.current)
      debounceRef.current = setTimeout(async () => {
        try {
          await updateSettings(sessionId, {
            lines: nextLines,
            zones: nextZones,
          })
        } catch {
          /* non-fatal */
        }
      }, DEBOUNCE_MS)
    },
    [sessionId]
  )

  useEffect(() => () => debounceRef.current && clearTimeout(debounceRef.current), [])

  const commit = useCallback(
    (nextLines, nextZones) => {
      setLines(nextLines)
      setZones(nextZones)
      scheduleSave(nextLines, nextZones)
    },
    [scheduleSave]
  )

  // -- coordinate mapping --------------------------------------------------
  // event -> normalized [0,1] relative to the video content rect
  const toNorm = useCallback(
    (e) => {
      const r = rect || contentRect(imgRef?.current)
      const wrap = wrapRef.current
      if (!r || !wrap) return null
      const box = wrap.getBoundingClientRect()
      const x = e.clientX - box.left - r.left
      const y = e.clientY - box.top - r.top
      return [clamp01(x / r.width), clamp01(y / r.height)]
    },
    [rect, imgRef]
  )

  // normalized [0,1] -> CSS px within the wrapper (for SVG rendering)
  const toPx = useCallback(
    (p) => {
      const r = rect
      if (!r) return [0, 0]
      return [r.left + p[0] * r.width, r.top + p[1] * r.height]
    },
    [rect]
  )

  // -- drawing interactions ------------------------------------------------
  const onClick = useCallback(
    (e) => {
      if (tool !== 'line' && tool !== 'zone') return
      const p = toNorm(e)
      if (!p) return
      if (tool === 'line') {
        const next = [...draft, p]
        if (next.length === 2) {
          counterRef.current.l += 1
          const id = `l${counterRef.current.l}`
          const newLine = { id, name: `Line ${counterRef.current.l}`, points: next }
          commit([...lines, newLine], zones)
          setDraft([])
          setTool('idle')
          setSelected({ kind: 'line', id })
        } else {
          setDraft(next)
        }
      } else {
        setDraft([...draft, p])
      }
    },
    [tool, draft, toNorm, commit, lines, zones]
  )

  const finishZone = useCallback(() => {
    if (tool !== 'zone' || draft.length < 3) return
    counterRef.current.z += 1
    const id = `z${counterRef.current.z}`
    const newZone = { id, name: `Zone ${counterRef.current.z}`, points: draft }
    commit(lines, [...zones, newZone])
    setDraft([])
    setTool('idle')
    setSelected({ kind: 'zone', id })
  }, [tool, draft, commit, lines, zones])

  const onDoubleClick = useCallback(
    (e) => {
      if (tool === 'zone') {
        e.preventDefault()
        finishZone()
      }
    },
    [tool, finishZone]
  )

  const onMove = useCallback(
    (e) => {
      if (tool === 'line' || tool === 'zone') setHover(toNorm(e))
    },
    [tool, toNorm]
  )

  // ESC cancels an in-progress draft
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setDraft([])
        if (tool === 'line' || tool === 'zone') setTool('idle')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [tool])

  // -- editing actions -----------------------------------------------------
  function selectTool(t) {
    setDraft([])
    setHover(null)
    setTool((cur) => (cur === t ? 'idle' : t))
    if (t !== 'select') setSelected(null)
  }

  function deleteSelected() {
    if (!selected) return
    if (selected.kind === 'line') {
      commit(lines.filter((l) => l.id !== selected.id), zones)
    } else {
      commit(lines, zones.filter((z) => z.id !== selected.id))
    }
    setSelected(null)
  }

  function clearAll() {
    commit([], [])
    setSelected(null)
    setDraft([])
    setTool('idle')
  }

  function rename(kind, id, name) {
    if (kind === 'line') {
      commit(lines.map((l) => (l.id === id ? { ...l, name } : l)), zones)
    } else {
      commit(lines, zones.map((z) => (z.id === id ? { ...z, name } : z)))
    }
  }

  function pickShape(kind, id) {
    if (tool === 'select') setSelected({ kind, id })
  }

  // -- tally lookups -------------------------------------------------------
  const lineCounts = useMemo(() => {
    const m = new Map()
    for (const lc of stats?.line_counts || []) m.set(lc.id, lc)
    return m
  }, [stats?.line_counts])

  const zoneOcc = useMemo(() => {
    const m = new Map()
    for (const zo of stats?.zone_occupancy || []) m.set(zo.id, zo)
    return m
  }, [stats?.zone_occupancy])

  // -- render --------------------------------------------------------------
  const drawing = tool === 'line' || tool === 'zone'
  const r = rect

  // build SVG point string for a polygon/polyline (in px)
  const ptsStr = (pts) => pts.map((p) => toPx(p).join(',')).join(' ')

  // draft preview points (committed draft + live hover for the moving segment)
  const draftPreview = hover && draft.length > 0 ? [...draft, hover] : draft

  // centroid (px) of a polygon for label placement
  const centroidPx = (pts) => {
    if (!pts.length) return [0, 0]
    let sx = 0
    let sy = 0
    for (const p of pts) {
      const q = toPx(p)
      sx += q[0]
      sy += q[1]
    }
    return [sx / pts.length, sy / pts.length]
  }

  return (
    <div className="roi-layer" ref={wrapRef}>
      {/* SVG render of saved shapes + draft. pointer-events only while drawing
          or selecting, so the MJPEG and badges stay interactive otherwise. */}
      {r && (
        <svg
          className={`roi-svg ${drawing || tool === 'select' ? 'interactive' : ''}`}
          onClick={onClick}
          onDoubleClick={onDoubleClick}
          onMouseMove={onMove}
        >
          {/* zones (drawn first, under lines) */}
          {zones.map((z) => {
            const isSel = selected?.kind === 'zone' && selected.id === z.id
            return (
              <polygon
                key={z.id}
                className={`roi-zone ${isSel ? 'selected' : ''}`}
                points={ptsStr(z.points)}
                onClick={(e) => {
                  e.stopPropagation()
                  pickShape('zone', z.id)
                }}
              />
            )
          })}

          {/* lines with a small arrow head indicating A->B direction */}
          {lines.map((l) => {
            const isSel = selected?.kind === 'line' && selected.id === l.id
            const a = toPx(l.points[0])
            const b = toPx(l.points[1])
            return (
              <g key={l.id}>
                <line
                  className={`roi-line ${isSel ? 'selected' : ''}`}
                  x1={a[0]}
                  y1={a[1]}
                  x2={b[0]}
                  y2={b[1]}
                  onClick={(e) => {
                    e.stopPropagation()
                    pickShape('line', l.id)
                  }}
                />
                <ArrowHead a={a} b={b} selected={isSel} />
                <circle className="roi-endpoint" cx={a[0]} cy={a[1]} r="4" />
                <circle className="roi-endpoint" cx={b[0]} cy={b[1]} r="4" />
              </g>
            )
          })}

          {/* in-progress draft */}
          {draftPreview.length > 0 && tool === 'line' && (
            <polyline className="roi-draft" points={ptsStr(draftPreview)} />
          )}
          {draftPreview.length > 0 && tool === 'zone' && (
            <polyline className="roi-draft" points={ptsStr(draftPreview)} />
          )}
          {draft.map((p, i) => {
            const q = toPx(p)
            return <circle key={i} className="roi-draft-dot" cx={q[0]} cy={q[1]} r="4" />
          })}
        </svg>
      )}

      {/* HTML labels with live tallies + rename, positioned over the shapes */}
      {r &&
        lines.map((l) => {
          const a = toPx(l.points[0])
          const b = toPx(l.points[1])
          const mx = (a[0] + b[0]) / 2
          const my = (a[1] + b[1]) / 2
          const lc = lineCounts.get(l.id)
          const isSel = selected?.kind === 'line' && selected.id === l.id
          return (
            <ShapeLabel
              key={l.id}
              x={mx}
              y={my}
              kind="line"
              shape={l}
              selected={isSel}
              editing={editing?.kind === 'line' && editing.id === l.id}
              onEdit={() => setEditing({ kind: 'line', id: l.id })}
              onEditDone={() => setEditing(null)}
              onRename={(name) => rename('line', l.id, name)}
              tally={
                lc ? (
                  <span className="roi-tally">
                    IN <b>{lc.in ?? 0}</b> / OUT <b>{lc.out ?? 0}</b>
                  </span>
                ) : null
              }
            />
          )
        })}

      {r &&
        zones.map((z) => {
          const c = centroidPx(z.points)
          const zo = zoneOcc.get(z.id)
          const isSel = selected?.kind === 'zone' && selected.id === z.id
          return (
            <ShapeLabel
              key={z.id}
              x={c[0]}
              y={c[1]}
              kind="zone"
              shape={z}
              selected={isSel}
              editing={editing?.kind === 'zone' && editing.id === z.id}
              onEdit={() => setEditing({ kind: 'zone', id: z.id })}
              onEditDone={() => setEditing(null)}
              onRename={(name) => rename('zone', z.id, name)}
              tally={
                zo ? (
                  <span className="roi-tally">
                    occ <b>{zo.count ?? 0}</b> (peak {zo.peak ?? 0})
                    {zo.avg_dwell_s != null && (
                      <> · {Number(zo.avg_dwell_s).toFixed(1)}s</>
                    )}
                  </span>
                ) : null
              }
            />
          )
        })}

      {/* toolbar */}
      <div className="roi-toolbar" onClick={(e) => e.stopPropagation()}>
        <button
          className={`roi-tool ${tool === 'line' ? 'on' : ''}`}
          onClick={() => selectTool('line')}
          title="Draw a counting line (click 2 points). A→B direction sets in/out."
        >
          Line
        </button>
        <button
          className={`roi-tool ${tool === 'zone' ? 'on' : ''}`}
          onClick={() => selectTool('zone')}
          title="Draw a zone polygon (click points, double-click or Finish to close)."
        >
          Zone
        </button>
        <button
          className={`roi-tool ${tool === 'select' ? 'on' : ''}`}
          onClick={() => selectTool('select')}
          title="Select a shape to delete or rename it."
        >
          Select
        </button>
        {tool === 'zone' && draft.length >= 3 && (
          <button className="roi-tool finish" onClick={finishZone}>
            Finish
          </button>
        )}
        {tool === 'select' && selected && (
          <button className="roi-tool danger" onClick={deleteSelected}>
            Delete
          </button>
        )}
        <button
          className="roi-tool"
          onClick={clearAll}
          disabled={lines.length === 0 && zones.length === 0}
          title="Remove all lines and zones."
        >
          Clear
        </button>
        {drawing && (
          <span className="roi-hint">
            {tool === 'line'
              ? draft.length === 0
                ? 'Click the start point'
                : 'Click the end point'
              : draft.length < 3
                ? `Click points (${draft.length})`
                : 'Double-click or Finish to close'}
          </span>
        )}
      </div>
    </div>
  )
}

// small arrowhead at the B end of a directed line
function ArrowHead({ a, b, selected }) {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const len = Math.hypot(dx, dy) || 1
  const ux = dx / len
  const uy = dy / len
  const size = 11
  // base point a little back from the tip
  const tipX = b[0]
  const tipY = b[1]
  const baseX = b[0] - ux * size
  const baseY = b[1] - uy * size
  // perpendicular
  const px = -uy
  const py = ux
  const w = size * 0.55
  const p1 = `${tipX},${tipY}`
  const p2 = `${baseX + px * w},${baseY + py * w}`
  const p3 = `${baseX - px * w},${baseY - py * w}`
  return (
    <polygon className={`roi-arrow ${selected ? 'selected' : ''}`} points={`${p1} ${p2} ${p3}`} />
  )
}

// HTML label overlay for a shape: name (editable) + live tally
function ShapeLabel({ x, y, kind, shape, selected, editing, onEdit, onEditDone, onRename, tally }) {
  const inputRef = useRef(null)
  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editing])

  return (
    <div
      className={`roi-label ${kind} ${selected ? 'selected' : ''}`}
      style={{ left: `${x}px`, top: `${y}px` }}
    >
      {editing ? (
        <input
          ref={inputRef}
          className="roi-name-input"
          defaultValue={shape.name}
          onBlur={(e) => {
            const v = e.target.value.trim()
            if (v && v !== shape.name) onRename(v)
            onEditDone()
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') e.target.blur()
            if (e.key === 'Escape') onEditDone()
          }}
        />
      ) : (
        <span
          className="roi-name"
          onDoubleClick={(e) => {
            e.stopPropagation()
            onEdit()
          }}
          title="Double-click to rename"
        >
          {shape.name}
        </span>
      )}
      {tally}
    </div>
  )
}
