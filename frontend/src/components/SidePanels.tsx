/** Events CRUD, family/preferences, and the form-based plan intake.
 *
 *  Neither screen exists in the design mock — the mock covers the workspace only — so both are
 *  built from the same tokens rather than a second visual language. Planning itself happens in
 *  chat; these panels only manage the events and family it plans around.
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

