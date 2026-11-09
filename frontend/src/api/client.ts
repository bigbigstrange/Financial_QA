const TOKEN_KEY = 'finance_qa_token'
let accessToken: string | null = localStorage.getItem(TOKEN_KEY)
localStorage.removeItem(TOKEN_KEY)
let refreshInFlight: Promise<string | null> | null = null

export function getToken(): string | null {
  return accessToken
}

export function setToken(token: string | null) {
  accessToken = token
  localStorage.removeItem(TOKEN_KEY)
}

function csrfCookie(): string | null {
  const pair = document.cookie.split('; ').find(value => value.startsWith('finance_csrf='))
  return pair ? decodeURIComponent(pair.slice('finance_csrf='.length)) : null
}

export function hasRefreshSession(): boolean {
  return !!csrfCookie()
}

async function refreshAccess(): Promise<string | null> {
  if (refreshInFlight) return refreshInFlight
  refreshInFlight = (async () => {
    const csrf = csrfCookie()
    if (!csrf) return null
    const res = await fetch('/api/auth/refresh', {
      method: 'POST', credentials: 'same-origin', headers: { 'X-CSRF-Token': csrf },
    })
    if (!res.ok) return null
    const data = await res.json() as { access_token: string }
    setToken(data.access_token)
    return data.access_token
  })().finally(() => { refreshInFlight = null })
  return refreshInFlight
}

export async function revokeSession(): Promise<void> {
  const csrf = csrfCookie()
  if (!csrf) return
  await fetch('/api/auth/logout', {
    method: 'POST', credentials: 'same-origin', headers: { 'X-CSRF-Token': csrf },
  })
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers || {})
  if (!headers.has('Content-Type') && init.body) {
    headers.set('Content-Type', 'application/json')
  }
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)

  let res = await fetch(path, { ...init, headers, credentials: 'same-origin' })
  if (res.status === 401 && !path.startsWith('/api/auth/login') &&
      !path.startsWith('/api/auth/refresh') && !path.startsWith('/api/auth/logout')) {
    const renewed = await refreshAccess()
    if (renewed) {
      headers.set('Authorization', `Bearer ${renewed}`)
      res = await fetch(path, { ...init, headers, credentials: 'same-origin' })
    }
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = await res.json()
      detail = data.detail || JSON.stringify(data)
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res
}

export async function apiBlob(path: string, init: RequestInit = {}): Promise<Blob> {
  return (await request(path, init)).blob()
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await request(path, init)
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

export async function apiStream(
  path: string,
  body: unknown,
  onEvent: (event: string, data: any, id?: number) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await request(path, { method: 'POST', body: JSON.stringify(body), signal })
  return readEventStream(res, onEvent)
}

export async function apiResumeStream(
  path: string,
  lastEventId: number,
  onEvent: (event: string, data: any, id?: number) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await request(path, {
    headers: { 'Last-Event-ID': String(lastEventId) }, signal,
  })
  return readEventStream(res, onEvent)
}

async function readEventStream(
  res: Response,
  onEvent: (event: string, data: any, id?: number) => void,
): Promise<void> {
  if (!res.body) throw new Error('服务端未提供事件流')
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let pending = ''
  try {
    for (;;) {
      const { value, done } = await reader.read()
      pending += decoder.decode(value, { stream: !done })
      pending = pending.replace(/\r\n/g, '\n')
      let boundary: number
      while ((boundary = pending.indexOf('\n\n')) !== -1) {
        const block = pending.slice(0, boundary).replace(/\r/g, '')
        pending = pending.slice(boundary + 2)
        const event = block.match(/^event: (.+)$/m)?.[1]
        const data = block.match(/^data: (.+)$/m)?.[1]
        const rawId = block.match(/^id: (\d+)$/m)?.[1]
        if (event && data) onEvent(event, JSON.parse(data), rawId ? Number(rawId) : undefined)
      }
      if (done) break
    }
  } finally {
    reader.releaseLock()
  }
}
