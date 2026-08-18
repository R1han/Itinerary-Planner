/** Mirrors backend/app/schemas.py. Kept hand-written and small rather than generated — the API
 *  surface is fixed by the spec, and a codegen step would be more machinery than it saves. */

export interface User {
  id: number
  email: string
  name: string
  home_base_lat: number
  home_base_lng: number
  default_currency: string
  default_budget: number
}

export type EventType =
  | 'birthday'
  | 'anniversary'
  | 'family_visit'
  | 'graduation'
  | 'eid'
  | 'holiday'
  | 'other'

export interface CalendarEvent {
  id: number
  title: string
  event_type: EventType
  date: string
  notes: string | null
  planned: boolean
}

export interface FamilyMember {
  id?: number
  role: 'adult' | 'child'
  age: number
  name?: string | null
}

export interface Preference {
  id: number
  kind: 'like' | 'dislike'
  subject: string
  category: string | null
  source: 'stated' | 'slot_edit'
  strength: number
  created_at: string
}

export interface Place {
  id: number
  name: string
  emirate: string
  lat: number
  lng: number
  category: string
  price_adult: number
  price_child: number
  min_age: number
  open_time: string
  close_time: string
  avg_duration_min: number
  tags: string[]
  image_url: string | null
  category_icon: string | null
  description: string
}

export interface CostChip {
  label: string
  count: number
  amount: number
  tone: 'adult' | 'child' | 'free'
}

export interface CostBreakdown {
  adults: number[]
  children: number[]
  free_children: number
  free_under_age: number | null
  travel_in: number
  total: number
  chips: CostChip[]
}

export interface Slot {
  id: number
  day_index: number
  position: number
  place_id: number
  start_time: string
  end_time: string
  locked: boolean
  cost_breakdown: CostBreakdown
  place: Place
}

export interface TravelSegment {
  id: number
  day_index: number
  from_slot_id: number | null
  to_slot_id: number | null
  distance_km: number
  duration_min: number
  mode: string
  est_cost: number
  estimated: boolean
  geometry_json: [number, number][] | null
}

export interface Day {
  day_index: number
  date: string
  theme: string
  subtotal: number
  driving_total_min: number
  slots: Slot[]
  segments: TravelSegment[]
}

export interface Budget {
  total: number
  cap: number
  remaining: number
  currency: string
  over_budget: boolean
  per_day: number[]
  categories: { activities: number; food: number; travel: number }
}

export interface Suggestion {
  id: string
  label: string
  action: 'cheaper_day' | 'prayer_breaks'
  day_index: number | null
}

export interface Itinerary {
  id: number
  title: string
  event_id: number | null
  event_title: string | null
  start_date: string
  num_days: number
  currency: string
  status: string
  transport_mode: TransportMode
  /** What the party has to travel in — derived from family size, e.g. "6-seater". */
  vehicle: string
  days: Day[]
  budget: Budget
  suggestions: Suggestion[]
  warnings: string[]
}

export type TransportMode = 'taxi' | 'own_car'

export interface ItinerarySummary {
  id: number
  title: string
  event_id: number | null
  start_date: string
  num_days: number
  total_budget: number
  currency: string
  status: string
  updated_at: string
}

export interface DayPatchResponse {
  day: Day
  budget: Budget
  suggestions: Suggestion[]
  warnings: string[]
}

export interface Alternative {
  place: Place
  start_time: string
  end_time: string
  cost_breakdown: CostBreakdown
  score: number
}

export interface Conversation {
  id: number
  title: string
  itinerary_id: number | null
  event_id: number | null
  updated_at: string
  last_seen_at: string
  unread: boolean
}

export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

/** Typed SSE events from POST /chat. */
export type StreamEvent =
  | { type: 'conversation'; data: { conversation_id: number; title: string } }
  | { type: 'token'; data: string }
  | {
      type: 'tool'
      data: { id: string; name: string; label: string; detail: string | null }
    }
  | { type: 'tool_done'; data: { id: string; outcome: string; failed: boolean } }
  | { type: 'itinerary_updated'; data: { itinerary_id: number } }
  | { type: 'budget_updated'; data: Budget }
  | { type: 'intake_required'; data: { missing_fields: string[] } }
  | { type: 'error'; data: { message: string } }
  | { type: 'done'; data: { conversation_id: number; failed?: boolean } }

/** One row in the chat's activity trace: what the assistant is doing and with what inputs. */
export interface ToolActivity {
  id: string
  name: string
  label: string
  detail: string | null
  outcome: string | null
  failed: boolean
}

export interface HealthStatus {
  status: string
  openai: boolean
  openrouteservice: boolean
}
