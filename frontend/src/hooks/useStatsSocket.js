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

  useEffect(() => {
    if (!sessionId) {
      setStats(null)
      setConnected(false)
      return
    }
    closedByUs.current = false
    attemptRef.current = 0

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
          if (!data.error) setStats(data)
        } catch {
          /* ignore malformed */
        }
      }
      ws.onclose = () => {
        setConnected(false)
        if (!closedByUs.current && attemptRef.current < 10) {
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
