// Thin API client.
//
// API_BASE defaults to "" (same origin) so the app works when the backend serves
// the built frontend on a single port — e.g. the Docker / Hugging Face deploy,
// and the Vite dev server (which proxies /api -> backend, see vite.config.js).
// Override with VITE_API_BASE (e.g. http://localhost:8000) to point at a backend
// on a different origin during split-stack development.

const API_BASE = import.meta.env.VITE_API_BASE || ''

export const apiBase = API_BASE

function wsBase() {
  if (API_BASE) return API_BASE.replace(/^http/, 'ws')
  // same-origin: derive ws/wss from the current page
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}`
}

async function request(method, path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = await res.json()
      detail = data.detail || JSON.stringify(data)
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.status === 204 ? null : res.json()
}

export const getConfig = () => request('GET', '/api/config')
export const getHealth = () => request('GET', '/api/health')

export async function uploadFile(file, onProgress) {
  // Use XHR so we can report upload progress.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/api/upload`)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText))
      } else {
        let msg = xhr.statusText
        try {
          msg = JSON.parse(xhr.responseText).detail || msg
        } catch {
          /* ignore */
        }
        reject(new Error(msg))
      }
    }
    xhr.onerror = () => reject(new Error('Upload failed (network error)'))
    const fd = new FormData()
    fd.append('file', file)
    xhr.send(fd)
  })
}

export const createSession = (payload) =>
  request('POST', '/api/sessions', payload)
export const stopSession = (id) =>
  request('POST', `/api/sessions/${id}/stop`)
export const deleteSession = (id) =>
  request('DELETE', `/api/sessions/${id}`)
export const updateSettings = (id, payload) =>
  request('PATCH', `/api/sessions/${id}/settings`, payload)

export const videoUrl = (id) => `${API_BASE}/api/sessions/${id}/video`
export const statsWsUrl = (id) => `${wsBase()}/api/sessions/${id}/ws`
