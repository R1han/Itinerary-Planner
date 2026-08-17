import { useEffect, useRef, useState } from 'react'
import { useStore } from '../state/store'
import type { FamilyMember } from '../types'
import { Gear, Sparkle } from './icons'

function formatRange(startDate: string, days: number): string {
  const start = new Date(`${startDate}T00:00:00`)
  const end = new Date(start)
  end.setDate(end.getDate() + Math.max(0, days - 1))

  const month = (date: Date) => date.toLocaleDateString('en-GB', { month: 'short' })
  return start.getMonth() === end.getMonth()
    ? `${month(start)} ${start.getDate()}–${end.getDate()}`
    : `${month(start)} ${start.getDate()} – ${month(end)} ${end.getDate()}`
}

function partySummary(family: FamilyMember[]): string {
  const adults = family.filter((m) => m.role === 'adult').length
  const children = family.length - adults
  const parts = [`${adults} adult${adults === 1 ? '' : 's'}`]
  if (children) parts.push(`${children} kid${children === 1 ? '' : 's'}`)
  return parts.join(' · ')
}

interface Props {
  family: FamilyMember[]
}

export function TopBar({ family }: Props) {
  const itinerary = useStore((s) => s.itinerary)
  const user = useStore((s) => s.user)
  const signOut = useStore((s) => s.signOut)
  const setPanel = useStore((s) => s.setPanel)
  const llmAvailable = useStore((s) => s.llmAvailable)
  const mapsAvailable = useStore((s) => s.mapsAvailable)

  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand__mark">
          <Sparkle />
        </div>
        <span className="brand__name">Rihla</span>
      </div>

      <div className="divider-v" />

      {itinerary ? (
        <div className="event-chip" title={itinerary.title}>
          <span className="event-chip__dot" />
          <span className="event-chip__label">Planning:</span>
          <span className="event-chip__title">{itinerary.event_title ?? itinerary.title}</span>
          <span className="event-chip__dates">
            · {formatRange(itinerary.start_date, itinerary.num_days)}
          </span>
        </div>
      ) : (
        <div className="event-chip">
          <span className="event-chip__dot" />
          <span className="event-chip__label">No plan yet</span>
        </div>
      )}

      <div className="spacer" />

      <div className="topbar__meta">
        <span>{itinerary ? 'Saved' : 'Ready'}</span>
        {family.length > 0 && (
          <>
            <span className="sep">·</span>
            <span>{partySummary(family)}</span>
          </>
        )}
      </div>

      <div className="menu" ref={menuRef}>
        <button
          className="icon-button"
          onClick={() => setOpen((value) => !value)}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label="Settings and account"
        >
          <Gear />
        </button>

        {open && (
          <div className="menu__panel" role="menu">
            <div className="menu__note">Signed in as {user?.name}</div>
            <button
              className="menu__item"
              role="menuitem"
              onClick={() => {
                setPanel('family')
                setOpen(false)
              }}
            >
              Family &amp; preferences
            </button>
            <button
              className="menu__item"
              role="menuitem"
              onClick={() => {
                setPanel('events')
                setOpen(false)
              }}
            >
              Events
            </button>
            {(!llmAvailable || !mapsAvailable) && (
              <div className="menu__note">
                {!llmAvailable && <div>Assistant offline — form planning still works.</div>}
                {!mapsAvailable && <div>Maps API offline — travel times are estimates.</div>}
              </div>
            )}
            <button className="menu__item" role="menuitem" onClick={signOut}>
              Sign out
            </button>
          </div>
        )}
      </div>
    </header>
  )
}
