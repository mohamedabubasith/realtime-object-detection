import { useEffect, useRef, useState } from 'react'
import { statsWsUrl } from '../api'

// Subscribe to a session's live stats over WebSocket, with auto-reconnect while
// the session is active. Returns { stats, connected }.
export function useStatsSocket(sessionId) {
  const [stats, setStats] = useState(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const retryRef = useRef(null)
  const attemptRef = useRef(0)
  const closedByUs = useRef(false)
  const terminalRef = useRef(false)

  useEffect(() => {
    if (!sessionId) {
      setStats(null)
      setConnected(false)
      return
    }
    closedByUs.current = false
    terminalRef.current = false
    attemptRef.current = 0

    const TERMINAL = ['error', 'finished', 'stopped']

    const connect = () => {
      const ws = new WebSocket(statsWsUrl(sessionId))
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        attemptRef.current = 0 // reset backoff on a good connection
      }
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data)
          // A real stats snapshot always has session_id (it may ALSO carry an
          // `error` message when the source failed — we must still show it).
          // The bare {error: "Session not found"} envelope has no session_id.
          if (data.session_id) {
            setStats(data)
            if (TERMINAL.includes(data.status)) terminalRef.current = true
          }
        } catch {
          /* ignore malformed */
        }
      }
      ws.onclose = () => {
        setConnected(false)
        // Don't reconnect once the session has reached a terminal state — the
        // server intentionally closed it; reconnecting just loops forever.
        if (
          !closedByUs.current &&
          !terminalRef.current &&
          attemptRef.current < 10
        ) {
          // exponential backoff with jitter, capped at ~30s
          const n = attemptRef.current++
          const delay = Math.min(30000, 1000 * 2 ** n) + Math.random() * 500
          retryRef.current = setTimeout(connect, delay)
        }
      }
      ws.onerror = () => ws.close()
    }

    connect()

    return () => {
      closedByUs.current = true
      if (retryRef.current) clearTimeout(retryRef.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [sessionId])

  return { stats, connected }
}
