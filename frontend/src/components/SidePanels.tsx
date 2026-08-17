/** Events CRUD, family/preferences, and the form-based plan intake.
 *
 *  None of these screens exist in the design mock — the mock covers the workspace only. They are
 *  built from the same tokens rather than a second visual language. The plan form is a first-class
 *  way to generate a trip without conversing, not a fallback.
 */

import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { useStore } from '../state/store'
import type { CalendarEvent, EventType, FamilyMember, Preference } from '../types'
import { Calendar, Close, People, Plus, Trash } from './icons'

const EVENT_TYPES: EventType[] = [
  'birthday',
  'anniversary',
  'family_visit',
  'graduation',
  'eid',
  'holiday',
  'other',
]

const prettyType = (value: string) => value.replace(/_/g, ' ')
const today = () => new Date().toISOString().slice(0, 10)

function PanelShell({
  title,
  icon,
  children,
}: {
  title: string
  icon: React.ReactNode
  children: React.ReactNode
}) {
  const setPanel = useStore((s) => s.setPanel)
  return (
    <div className="panel">
      <div className="panel__header">
        {icon}
        <h2>{title}</h2>
        <span className="spacer" />
        <button className="icon-button" onClick={() => setPanel(null)} aria-label="Close panel">
          <Close />
        </button>
      </div>
      <div className="panel__body">{children}</div>
    </div>
  )
}

export function EventsPanel() {
  const setPanel = useStore((s) => s.setPanel)
  const setError = useStore((s) => s.setError)
  const [events, setEvents] = useState<CalendarEvent[]>([])
  const [form, setForm] = useState({
    title: '',
    event_type: 'birthday' as EventType,
    date: today(),
    notes: '',
  })
  const [busy, setBusy] = useState(false)

  const refresh = () => api.events().then(setEvents).catch(() => undefined)
  useEffect(() => {
    void refresh()
  }, [])

  const create = async () => {
    if (!form.title.trim()) return
    setBusy(true)
    try {
      await api.createEvent({ ...form, title: form.title.trim(), notes: form.notes || undefined })
      setForm({ title: '', event_type: 'birthday', date: today(), notes: '' })
      await refresh()
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Could not add that event.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <PanelShell title="Events" icon={<Calendar />}>
      <div className="field">
        <label htmlFor="event-title">New event</label>
        <input
          id="event-title"
          value={form.title}
          placeholder="e.g. Aisha's 7th birthday"
          onChange={(event) => setForm({ ...form, title: event.target.value })}
        />
      </div>
      <div className="field__row">
        <div className="field">
          <label htmlFor="event-type">Type</label>
          <select
            id="event-type"
            value={form.event_type}
            onChange={(event) => setForm({ ...form, event_type: event.target.value as EventType })}
          >
            {EVENT_TYPES.map((type) => (
              <option key={type} value={type}>
                {prettyType(type)}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="event-date">Date</label>
          <input
            id="event-date"
            type="date"
            min={today()}
            value={form.date}
            onChange={(event) => setForm({ ...form, date: event.target.value })}
          />
        </div>
      </div>
      <div className="field">
        <label htmlFor="event-notes">Notes</label>
        <input
          id="event-notes"
          value={form.notes}
          placeholder="loves animals, afraid of loud rides"
          onChange={(event) => setForm({ ...form, notes: event.target.value })}
        />
      </div>
      <button className="btn btn--primary" disabled={busy || !form.title.trim()} onClick={create}>
        <Plus size={13} color="#F7F1E7" /> Add event
      </button>

      {events.map((event) => (
        <div className="list-row" key={event.id}>
          <div className="list-row__body">
            <div className="list-row__title">{event.title}</div>
            <div className="list-row__meta">
              {event.date} · {prettyType(event.event_type)}
              {event.notes ? ` · ${event.notes}` : ''}
            </div>
          </div>
          <span className={`pill${event.planned ? '' : ' pill--muted'}`}>
            {event.planned ? 'Planned' : 'Not planned'}
          </span>
          <button
            className="slot-action slot-action--danger"
            aria-label={`Delete ${event.title}`}
            onClick={async () => {
              await api.deleteEvent(event.id)
              await refresh()
            }}
          >
            <Trash />
          </button>
        </div>
      ))}

      {events.length === 0 && (
        <p className="list-row__meta">No events yet. Add one and Rihla can plan around it.</p>
      )}

      <button className="btn btn--ghost" onClick={() => setPanel('plan')}>
        Plan a trip from a form instead
      </button>
    </PanelShell>
  )
}

export function FamilyPanel({ onSaved }: { onSaved: () => void }) {
  const setError = useStore((s) => s.setError)
  const [members, setMembers] = useState<FamilyMember[]>([])
  const [preferences, setPreferences] = useState<Preference[]>([])
  const [subject, setSubject] = useState('')
  const [kind, setKind] = useState<'like' | 'dislike'>('like')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void api.family().then(setMembers).catch(() => undefined)
    void api.preferences().then(setPreferences).catch(() => undefined)
  }, [])

  const save = async () => {
    setBusy(true)
    try {
      const saved = await api.saveFamily(
        members.map(({ role, age, name }) => ({ role, age, name: name ?? null })),
      )
      setMembers(saved)
      onSaved()
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Could not save the family.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <PanelShell title="Family & preferences" icon={<People />}>
      {members.map((member, index) => (
        <div className="field__row" key={index}>
          <div className="field">
            <label>Role</label>
            <select
              value={member.role}
              onChange={(event) => {
                const next = [...members]
                const role = event.target.value as 'adult' | 'child'
                next[index] = { ...member, role, age: role === 'adult' ? 34 : 8 }
                setMembers(next)
              }}
            >
              <option value="adult">Adult</option>
              <option value="child">Child</option>
            </select>
          </div>
          <div className="field">
            <label>Age</label>
            <input
              type="number"
              min={0}
              max={99}
              value={member.age}
              onChange={(event) => {
                const next = [...members]
                next[index] = { ...member, age: Number(event.target.value) }
                setMembers(next)
              }}
            />
          </div>
          <div className="field">
            <label>Name</label>
            <input
              value={member.name ?? ''}
              onChange={(event) => {
                const next = [...members]
                next[index] = { ...member, name: event.target.value }
                setMembers(next)
              }}
            />
          </div>
          <button
            className="slot-action slot-action--danger"
            style={{ alignSelf: 'end', marginBottom: 4 }}
            aria-label="Remove family member"
            onClick={() => setMembers(members.filter((_, i) => i !== index))}
          >
            <Trash />
          </button>
        </div>
      ))}

      <div className="field__row">
        <button
          className="btn btn--ghost"
          onClick={() => setMembers([...members, { role: 'adult', age: 34, name: '' }])}
        >
          Add adult
        </button>
        <button
          className="btn btn--ghost"
          onClick={() => setMembers([...members, { role: 'child', age: 8, name: '' }])}
        >
          Add child
        </button>
      </div>

      <button className="btn btn--primary" disabled={busy || !members.length} onClick={save}>
        Save family
      </button>

      <div className="field">
        <label>Likes and dislikes</label>
        <div className="field__row">
          <select value={kind} onChange={(event) => setKind(event.target.value as 'like' | 'dislike')}>
            <option value="like">Like</option>
            <option value="dislike">Dislike</option>
          </select>
          <input
            value={subject}
            placeholder="e.g. animals and zoos"
            onChange={(event) => setSubject(event.target.value)}
          />
          <button
            className="btn btn--ghost"
            disabled={!subject.trim()}
            onClick={async () => {
              await api.addPreference({ kind, subject: subject.trim() })
              setSubject('')
              setPreferences(await api.preferences())
            }}
          >
            Add
          </button>
        </div>
      </div>

      {preferences.map((preference) => (
        <div className="list-row" key={preference.id}>
          <div className="list-row__body">
            <div className="list-row__title">{preference.subject}</div>
            <div className="list-row__meta">
              {preference.kind} · from {preference.source === 'slot_edit' ? 'a slot edit' : 'you'}
            </div>
          </div>
          <button
            className="slot-action slot-action--danger"
            aria-label={`Forget ${preference.subject}`}
            onClick={async () => {
              await api.deletePreference(preference.id)
              setPreferences(await api.preferences())
            }}
          >
            <Trash />
          </button>
        </div>
      ))}
    </PanelShell>
  )
}

export function PlanPanel() {
  const user = useStore((s) => s.user)
  const setItinerary = useStore((s) => s.setItinerary)
  const setPanel = useStore((s) => s.setPanel)
  const refreshConversations = useStore((s) => s.refreshConversations)
  const conversationId = useStore((s) => s.conversationId)

  const [events, setEvents] = useState<CalendarEvent[]>([])
  const [eventId, setEventId] = useState<string>('')
  const [startDate, setStartDate] = useState(today())
  const [days, setDays] = useState(3)
  const [budget, setBudget] = useState(user?.default_budget ?? 3500)
  const [prayerBreaks, setPrayerBreaks] = useState(false)
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)
  const [missing, setMissing] = useState<string[]>([])

  useEffect(() => {
    void api
      .upcomingEvents()
      .then((rows) => {
        setEvents(rows)
        const unplanned = rows.find((row) => !row.planned)
        if (unplanned) {
          setEventId(String(unplanned.id))
          setStartDate(unplanned.date)
        }
      })
      .catch(() => undefined)
  }, [])

  const generate = async () => {
    if (!user) return
    setBusy(true)
    setProblem(null)
    setMissing([])
    try {
      const itinerary = await api.generate({
        event_id: eventId ? Number(eventId) : null,
        start_date: startDate,
        num_days: days,
        total_budget: budget,
        start_lat: user.home_base_lat,
        start_lng: user.home_base_lng,
        prayer_breaks: prayerBreaks,
      })
      setItinerary(itinerary)
      setPanel(null)

      // Bind the active thread to the plan so the rail shows the event's initial, as in the
      // design, rather than a generic "New plan".
      const thread =
        conversationId ?? (await api.createConversation(itinerary.title, itinerary.event_id)).id
      await api.updateConversation(thread, {
        title: itinerary.event_title ?? itinerary.title,
        itinerary_id: itinerary.id,
        event_id: itinerary.event_id,
      })
      await refreshConversations()
    } catch (error) {
      const detail = (error as { detail?: unknown }).detail
      if (detail && typeof detail === 'object' && 'missing_fields' in detail) {
        setMissing((detail as { missing_fields: string[] }).missing_fields)
        setProblem('Some details are still missing before I can plan this.')
      } else {
        setProblem(error instanceof Error ? error.message : 'Could not generate that plan.')
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <PanelShell title="Plan a trip" icon={<Calendar />}>
      <div className="field">
        <label htmlFor="plan-event">Event</label>
        <select
          id="plan-event"
          value={eventId}
          onChange={(event) => {
            setEventId(event.target.value)
            const match = events.find((row) => String(row.id) === event.target.value)
            if (match) setStartDate(match.date)
          }}
        >
          <option value="">No particular event</option>
          {events.map((event) => (
            <option key={event.id} value={event.id}>
              {event.title} — {event.date}
            </option>
          ))}
        </select>
      </div>

      <div className="field__row">
        <div className="field">
          <label htmlFor="plan-start">Start date</label>
          <input
            id="plan-start"
            type="date"
            min={today()}
            value={startDate}
            onChange={(event) => setStartDate(event.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="plan-days">Days</label>
          <input
            id="plan-days"
            type="number"
            min={1}
            max={5}
            value={days}
            onChange={(event) => setDays(Number(event.target.value))}
          />
        </div>
      </div>

      <div className="field">
        <label htmlFor="plan-budget">Total budget (AED)</label>
        <input
          id="plan-budget"
          type="number"
          min={1}
          step={50}
          value={budget}
          onChange={(event) => setBudget(Number(event.target.value))}
        />
      </div>

      <label className="field__row" style={{ alignItems: 'center', gap: 8 }}>
        <input
          type="checkbox"
          style={{ flex: 'none', width: 16, height: 16 }}
          checked={prayerBreaks}
          onChange={(event) => setPrayerBreaks(event.target.checked)}
        />
        <span style={{ fontSize: 13 }}>Leave gaps for prayer times</span>
      </label>

      {problem && <div className="error-text">{problem}</div>}
      {missing.length > 0 && (
        <div className="notice-text">
          Still needed: {missing.map((field) => field.replace(/_/g, ' ')).join(', ')}. Set these in
          Family &amp; preferences.
        </div>
      )}

      <button className="btn btn--primary" disabled={busy} onClick={generate}>
        {busy ? 'Planning…' : 'Generate itinerary'}
      </button>
      <p className="list-row__meta">
        Max 5 days, UAE only. The planner enforces travel time, opening hours, age limits and your
        budget cap.
      </p>
    </PanelShell>
  )
}
