/** The single place a bearer token is attached, and the single place a 401 is handled. */

import type {
  Alternative,
  CalendarEvent,
  ChatMessage,
  Conversation,
  DayPatchResponse,
  EventType,
  FamilyMember,
  HealthStatus,
  Itinerary,
  Preference,
  StreamEvent,
  TransportMode,
  User,
} from '../types'

const TOKEN_KEY = 'rihla.token'

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly detail?: unknown,
  ) {
    super(message)
  }
}

let onUnauthorized: (() => void) | null = null

export function setUnauthorizedHandler(handler: () => void) {
  onUnauthorized = handler
}

export const token = {
  get: () => localStorage.getItem(TOKEN_KEY),
  set: (value: string) => localStorage.setItem(TOKEN_KEY, value),
  clear: () => localStorage.removeItem(TOKEN_KEY),
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const current = token.get()
  return {
    'Content-Type': 'application/json',
    ...(current ? { Authorization: `Bearer ${current}` } : {}),
    ...extra,
  }
}

async function toError(response: Response): Promise<ApiError> {
  let detail: unknown
  let message = `Request failed (${response.status})`
  try {
    const body = await response.json()
    detail = body?.detail ?? body
    if (typeof detail === 'string') {
      message = detail
    } else if (Array.isArray(detail)) {
      // FastAPI validation errors: surface the first one in plain language.
      const first = detail[0] as { loc?: unknown[]; msg?: string } | undefined
      const field = Array.isArray(first?.loc) ? String(first!.loc.at(-1)) : ''
      message = [field, first?.msg].filter(Boolean).join(': ') || message
    } else if (detail && typeof detail === 'object' && 'error' in detail) {
      message = String((detail as { error: string }).error)
    }
  } catch {
    /* a non-JSON error body is fine; the status message stands */
  }
  return new ApiError(response.status, message, detail)
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { ...init, headers: headers(init.headers as never) })

  if (response.status === 401) {
    token.clear()
    onUnauthorized?.()
    throw new ApiError(401, 'Your session has expired. Please sign in again.')
  }
  if (!response.ok) throw await toError(response)
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })

const patch = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'PATCH', body: JSON.stringify(body) })

export const api = {
  health: () => request<HealthStatus>('/health'),

  // --- auth
  register: (email: string, password: string, name: string) =>
    post<{ access_token: string; user: User }>('/auth/register', { email, password, name }),
  login: (email: string, password: string) =>
    post<{ access_token: string; user: User }>('/auth/login', { email, password }),
  me: () => request<User>('/me'),

  // --- events
  events: () => request<CalendarEvent[]>('/events'),
  createEvent: (body: { title: string; event_type: EventType; date: string; notes?: string }) =>
    post<CalendarEvent>('/events', body),
  deleteEvent: (id: number) => request<void>(`/events/${id}`, { method: 'DELETE' }),

  // --- family + preferences
  family: () => request<FamilyMember[]>('/family'),
  saveFamily: (members: FamilyMember[]) =>
    request<FamilyMember[]>('/family', { method: 'PUT', body: JSON.stringify({ members }) }),
  preferences: () => request<Preference[]>('/preferences'),
  addPreference: (body: {
    kind: 'like' | 'dislike'
    subject: string
    category?: string | null
    source?: 'stated' | 'slot_edit'
    strength?: number
  }) => post<Preference>('/preferences', body),
  deletePreference: (id: number) => request<void>(`/preferences/${id}`, { method: 'DELETE' }),

  // --- itineraries
  itinerary: (id: number) => request<Itinerary>(`/itineraries/${id}`),

  alternatives: (itineraryId: number, slotId: number) =>
    request<Alternative[]>(`/itineraries/${itineraryId}/slots/${slotId}/alternatives`),
  patchSlot: (
    itineraryId: number,
    slotId: number,
    body: { action: 'replace' | 'adjust' | 'remove'; place_id?: number; start_time?: string },
  ) => patch<DayPatchResponse>(`/itineraries/${itineraryId}/slots/${slotId}`, body),
  cheaperDay: (itineraryId: number, dayIndex: number) =>
    post<Itinerary>(`/itineraries/${itineraryId}/days/${dayIndex}/cheaper`),
  prayerBreaks: (itineraryId: number) =>
    post<Itinerary>(`/itineraries/${itineraryId}/prayer-breaks`),
  setTransport: (itineraryId: number, mode: TransportMode) =>
    post<Itinerary>(`/itineraries/${itineraryId}/transport`, { mode }),

  // --- conversations
  conversations: () => request<Conversation[]>('/conversations'),
  createConversation: (title = 'New plan', eventId?: number | null) =>
    post<Conversation>('/conversations', { title, event_id: eventId ?? null }),
  messages: (id: number) => request<ChatMessage[]>(`/conversations/${id}/messages`),
  markSeen: (id: number) => post<Conversation>(`/conversations/${id}/seen`),
}

/**
 * Stream POST /chat.
 *
 * EventSource cannot send an Authorization header, so the SSE body is read from a fetch
 * ReadableStream and framed by hand. Frames are split on the blank line that terminates an SSE
 * event; a partial frame is carried over to the next chunk.
 */
export async function streamChat(
  message: string,
  conversationId: number | null,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/chat', {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ message, conversation_id: conversationId }),
    signal,
  })

  if (response.status === 401) {
    token.clear()
    onUnauthorized?.()
    throw new ApiError(401, 'Your session has expired. Please sign in again.')
  }
  if (!response.ok) throw await toError(response)
  if (!response.body) throw new ApiError(500, 'The server sent an empty response.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data: ')) continue
        try {
          onEvent(JSON.parse(line.slice(6)) as StreamEvent)
        } catch {
          /* a malformed frame must not kill the stream */
        }
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
}
