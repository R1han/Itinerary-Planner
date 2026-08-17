import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { useStore } from '../state/store'
import type { Alternative, Day, Slot, TravelSegment } from '../types'
import { Car, Clock, Star, Swap, Trash } from './icons'
import { Thumb, categoryLabel } from './Thumb'

/** Anchor a popover to an element in viewport coordinates.
 *
 *  The strip body scrolls, so an absolutely-positioned panel inside it gets clipped at the
 *  container edge. Fixed positioning lets the editor and the preference toast overflow the strip,
 *  which is what the design shows, and it flips above the card when there is no room below.
 */
function useAnchored(anchor: React.RefObject<HTMLElement>, open: boolean) {
  const panel = useRef<HTMLDivElement>(null)
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null)

  const measure = useCallback(() => {
    const anchorEl = anchor.current
    const panelEl = panel.current
    if (!anchorEl || !panelEl) return

    const box = anchorEl.getBoundingClientRect()
    // Measure the panel rather than assuming a height: the editor grows from three menu rows to
    // a list of alternatives, and a guessed height flips it off the top of the screen.
    const height = panelEl.offsetHeight
    const width = panelEl.offsetWidth || 330
    const margin = 12
    const below = box.bottom + 6

    const top =
      below + height <= window.innerHeight - margin
        ? below
        : Math.max(margin, Math.min(box.top - height - 6, window.innerHeight - height - margin))

    setPosition({
      top,
      left: Math.max(margin, Math.min(box.right - width, window.innerWidth - width - margin)),
    })
  }, [anchor])

  useLayoutEffect(() => {
    if (!open) {
      setPosition(null)
      return
    }
    measure()
    const observer = new ResizeObserver(measure)
    if (panel.current) observer.observe(panel.current)
    window.addEventListener('resize', measure)
    window.addEventListener('scroll', measure, true)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
      window.removeEventListener('scroll', measure, true)
    }
  }, [open, measure])

  return { panel, position }
}

const DINING_CATEGORIES = new Set(['casual_dining', 'fine_dining'])

/** Dismiss a popover when the pointer goes down anywhere outside it.
 *
 *  The anchor is excluded as well as the panel: the card's own action buttons toggle the editor,
 *  and closing on their mousedown would fight the toggle on the click that follows.
 */
function useDismissOnOutsideClick(
  refs: React.RefObject<HTMLElement | null>[],
  onDismiss: () => void,
) {
  const latest = useRef(onDismiss)
  latest.current = onDismiss

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      const target = event.target as Node | null
      if (!target) return
      if (refs.some((ref) => ref.current?.contains(target))) return
      latest.current()
    }
    document.addEventListener('mousedown', onPointerDown)
    return () => document.removeEventListener('mousedown', onPointerDown)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, refs)
}

function weekday(dateIso: string): string {
  return new Date(`${dateIso}T00:00:00`).toLocaleDateString('en-GB', { weekday: 'short' })
}

/** "🚗 35 min · AED 40", or "~50 min estimated · AED 55" when the leg is a haversine estimate. */
function TravelConnector({ segment, currency }: { segment: TravelSegment; currency: string }) {
  return (
    <div className="connector">
      <span className="connector__tick" />
      <Car color="rgba(31,42,42,.5)" />
      <span className="connector__time">
        {segment.estimated ? '~' : ''}
        {segment.duration_min} min
      </span>
      {segment.estimated && <span className="connector__estimated">estimated</span>}
      <span className="connector__sep">·</span>
      <span className="connector__cost">
        {currency} {Math.round(segment.est_cost)}
      </span>
      <span className="connector__rule" />
    </div>
  )
}

interface EditorProps {
  slot: Slot
  itineraryId: number
  currency: string
  onClose: () => void
  anchor: React.RefObject<HTMLElement>
}

function SlotEditor({ slot, itineraryId, currency, onClose, anchor }: EditorProps) {
  const { panel, position } = useAnchored(anchor, true)
  useDismissOnOutsideClick([panel, anchor], onClose)
  const applyDayPatch = useStore((s) => s.applyDayPatch)
  const setPendingPreference = useStore((s) => s.setPendingPreference)
  const setError = useStore((s) => s.setError)

  const [mode, setMode] = useState<'menu' | 'replace' | 'adjust'>('menu')
  const [alternatives, setAlternatives] = useState<Alternative[] | null>(null)
  const [time, setTime] = useState(slot.start_time)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (mode !== 'replace' || alternatives) return
    let cancelled = false
    api
      .alternatives(itineraryId, slot.id)
      .then((options) => !cancelled && setAlternatives(options))
      .catch(() => !cancelled && setAlternatives([]))
    return () => {
      cancelled = true
    }
  }, [mode, alternatives, itineraryId, slot.id])

  const apply = async (
    body: { action: 'replace' | 'adjust' | 'remove'; place_id?: number; start_time?: string },
    askAboutPreference: boolean,
  ) => {
    setBusy(true)
    try {
      const result = await api.patchSlot(itineraryId, slot.id, body)
      applyDayPatch(result.day, result.budget, result.suggestions)
      // Never offer to avoid a meal category: the planner has to schedule lunch and dinner, so
      // "avoid dining in future plans" is advice it cannot act on.
      if (askAboutPreference && !DINING_CATEGORIES.has(slot.place.category)) {
        setPendingPreference({
          slotId: slot.id,
          placeName: slot.place.name,
          category: slot.place.category,
        })
      }
      onClose()
    } catch (error) {
      setError(error instanceof Error ? error.message : 'That change did not work.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className="editor"
      role="dialog"
      ref={panel}
      aria-label={`Edit ${slot.place.name}`}
      style={{
        top: position?.top ?? 0,
        left: position?.left ?? 0,
        // Hidden for the single frame between mount and measurement, so it never flashes at 0,0.
        visibility: position ? 'visible' : 'hidden',
      }}
    >
      {mode === 'menu' && (
        <>
          <div className="editor__title">Edit this stop</div>
          <button className="editor__option" onClick={() => setMode('replace')}>
            <Swap size={16} />
            <div className="editor__option-body">
              <div className="editor__option-name">Replace</div>
              <div className="editor__option-meta">Three options that fit this exact window</div>
            </div>
          </button>
          <button className="editor__option" onClick={() => setMode('adjust')}>
            <Clock size={16} />
            <div className="editor__option-body">
              <div className="editor__option-name">Adjust time</div>
              <div className="editor__option-meta">Currently {slot.start_time}</div>
            </div>
          </button>
          <button
            className="editor__option"
            disabled={busy}
            onClick={() => void apply({ action: 'remove' }, true)}
          >
            <Trash size={16} />
            <div className="editor__option-body">
              <div className="editor__option-name" style={{ color: 'var(--coral-dark)' }}>
                Remove
              </div>
              <div className="editor__option-meta">The rest of the day is re-checked</div>
            </div>
          </button>
        </>
      )}

      {mode === 'replace' && (
        <>
          <div className="editor__title">Replace with</div>
          {alternatives === null && <div className="editor__option-meta">Finding options…</div>}
          {alternatives?.length === 0 && (
            <div className="editor__option-meta">
              Nothing else fits this window and the remaining budget.
            </div>
          )}
          {alternatives?.map((option) => (
            <button
              key={option.place.id}
              className="editor__option"
              disabled={busy}
              onClick={() => void apply({ action: 'replace', place_id: option.place.id }, true)}
            >
              <Thumb
                src={option.place.image_url}
                category={option.place.category}
                alt={option.place.name}
                size={40}
                radius={10}
              />
              <div className="editor__option-body">
                <div className="editor__option-name">{option.place.name}</div>
                <div className="editor__option-meta">
                  {option.start_time}–{option.end_time} · {currency}{' '}
                  {Math.round(option.cost_breakdown.total)}
                </div>
              </div>
            </button>
          ))}
          <div className="editor__row">
            <button className="btn btn--ghost" onClick={() => setMode('menu')}>
              Back
            </button>
          </div>
        </>
      )}

      {mode === 'adjust' && (
        <>
          <div className="editor__title">Start time</div>
          <div className="editor__row">
            <input
              type="time"
              value={time}
              onChange={(event) => setTime(event.target.value)}
              aria-label="New start time"
            />
            <button
              className="btn btn--primary"
              disabled={busy}
              onClick={() => void apply({ action: 'adjust', start_time: time }, false)}
            >
              Apply
            </button>
          </div>
          <div className="editor__option-meta">
            Later stops shift to keep travel time and opening hours valid.
          </div>
          <div className="editor__row">
            <button className="btn btn--ghost" onClick={() => setMode('menu')}>
              Back
            </button>
          </div>
        </>
      )}
    </div>
  )
}

/** "Avoid this type in future?" — only written after the user confirms (spec §9). */
function PreferenceToast({ anchor }: { anchor: React.RefObject<HTMLElement> }) {
  const { panel, position } = useAnchored(anchor, true)
  const pending = useStore((s) => s.pendingPreference)
  const setPending = useStore((s) => s.setPendingPreference)
  const setError = useStore((s) => s.setError)
  // Dismissing without answering records nothing — the same outcome as "No", which is the safe
  // default for a question about a lasting preference.
  useDismissOnOutsideClick([panel], () => setPending(null))

  if (!pending) return null

  const confirm = async () => {
    try {
      await api.addPreference({
        kind: 'dislike',
        subject: `${categoryLabel(pending.category).toLowerCase()} like ${pending.placeName}`,
        category: pending.category,
        source: 'slot_edit',
        strength: 0.7,
      })
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Could not save that preference.')
    } finally {
      setPending(null)
    }
  }

  return (
    <div
      className="pref-toast"
      role="status"
      ref={panel}
      style={{
        top: position?.top ?? 0,
        left: position?.left ?? 0,
        visibility: position ? 'visible' : 'hidden',
      }}
    >
      <div className="pref-toast__row">
        <Star />
        <div>
          You swapped out {pending.placeName}. Avoid{' '}
          {categoryLabel(pending.category).toLowerCase()} in future plans?
        </div>
      </div>
      <div className="pref-toast__actions">
        <button className="pref-toast__yes" onClick={() => void confirm()}>
          Yes
        </button>
        <button className="pref-toast__no" onClick={() => setPending(null)}>
          No
        </button>
      </div>
    </div>
  )
}

function SlotCard({ slot, index, currency }: { slot: Slot; index: number; currency: string }) {
  const hoveredSlotId = useStore((s) => s.hoveredSlotId)
  const selectedSlotId = useStore((s) => s.selectedSlotId)
  const editingSlotId = useStore((s) => s.editingSlotId)
  const setHoveredSlot = useStore((s) => s.setHoveredSlot)
  const setSelectedSlot = useStore((s) => s.setSelectedSlot)
  const setEditingSlot = useStore((s) => s.setEditingSlot)
  const itinerary = useStore((s) => s.itinerary)
  const pendingPreference = useStore((s) => s.pendingPreference)

  const active = slot.id === hoveredSlotId || slot.id === selectedSlotId
  const editing = slot.id === editingSlotId
  const askingPreference = pendingPreference?.slotId === slot.id

  const shellRef = useRef<HTMLDivElement>(null)

  return (
    <div className="slot-shell" ref={shellRef}>
      <div
        className={`slotcard${active ? ' slotcard--active' : ''}`}
        onMouseEnter={() => setHoveredSlot(slot.id)}
        onMouseLeave={() => setHoveredSlot(null)}
        onClick={() => setSelectedSlot(slot.id)}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            setSelectedSlot(slot.id)
          }
        }}
        aria-label={`${slot.place.name}, ${slot.start_time} to ${slot.end_time}`}
      >
        <div className="slotcard__thumb-wrap">
          <Thumb
            src={slot.place.image_url}
            category={slot.place.category}
            alt={slot.place.name}
            size={74}
          />
          <span className="slotcard__badge">{index + 1}</span>
        </div>

        <div className="slotcard__body">
          <div className="slotcard__head">
            <span className="slotcard__name">{slot.place.name}</span>
            <span className="tag">{categoryLabel(slot.place.category)}</span>
            <span className="spacer" />
            <span className="slotcard__time">
              {slot.start_time}–{slot.end_time}
            </span>
            {active && (
              <div className="slotcard__actions">
                <button
                  className="slot-action"
                  title="Replace"
                  aria-label={`Replace ${slot.place.name}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    setEditingSlot(editing ? null : slot.id)
                  }}
                >
                  <Swap />
                </button>
                <button
                  className="slot-action"
                  title="Adjust time"
                  aria-label={`Adjust the time of ${slot.place.name}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    setEditingSlot(editing ? null : slot.id)
                  }}
                >
                  <Clock />
                </button>
                <button
                  className="slot-action slot-action--danger"
                  title="Remove"
                  aria-label={`Remove ${slot.place.name}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    setEditingSlot(editing ? null : slot.id)
                  }}
                >
                  <Trash />
                </button>
              </div>
            )}
          </div>

          <div className="cost-chips">
            {slot.cost_breakdown.chips.map((chip) => (
              <span key={chip.label} className={`cost-chip cost-chip--${chip.tone}`}>
                {chip.label}
              </span>
            ))}
          </div>
        </div>
      </div>

      {editing && itinerary && (
        <SlotEditor
          slot={slot}
          itineraryId={itinerary.id}
          currency={currency}
          onClose={() => setEditingSlot(null)}
          anchor={shellRef}
        />
      )}

      {askingPreference && <PreferenceToast anchor={shellRef} />}
    </div>
  )
}

function DayTabs({ days }: { days: Day[] }) {
  const selectedDay = useStore((s) => s.selectedDay)
  const setSelectedDay = useStore((s) => s.setSelectedDay)

  return (
    <div className="daytabs" role="tablist" aria-label="Days">
      {days.map((day) => (
        <button
          key={day.day_index}
          role="tab"
          aria-selected={day.day_index === selectedDay}
          className={`daytab${day.day_index === selectedDay ? ' daytab--active' : ''}`}
          onClick={() => setSelectedDay(day.day_index)}
        >
          Day {day.day_index + 1} · {weekday(day.date)}
        </button>
      ))}
    </div>
  )
}

export function ItineraryStrip() {
  const itinerary = useStore((s) => s.itinerary)
  const selectedDay = useStore((s) => s.selectedDay)

  if (!itinerary) return null
  const day = itinerary.days[selectedDay]

  const segmentInto = new Map(
    (day?.segments ?? []).map((segment) => [segment.to_slot_id, segment]),
  )

  return (
    <section className="strip" aria-label="Itinerary">
      <div className="strip__header">
        <DayTabs days={itinerary.days} />
        <span className="spacer" />
        <span className="strip__subtotal-label">Day {selectedDay + 1} subtotal</span>
        <span className="strip__subtotal">
          {itinerary.currency} {Math.round(day?.subtotal ?? 0).toLocaleString()}
        </span>
      </div>

      <div className="strip__body">
        {!day || day.slots.length === 0 ? (
          <div className="empty">
            <div className="empty__inner">
              <h2>Nothing scheduled</h2>
              <p>
                This day came out empty — usually the budget or the distance from your start point.
                Try &ldquo;Cheaper Day&rdquo;, or raise the cap and regenerate.
              </p>
            </div>
          </div>
        ) : (
          day.slots.map((slot, index) => {
            const inbound = segmentInto.get(slot.id)
            return (
              <div key={slot.id}>
                {index > 0 && inbound && (
                  <TravelConnector segment={inbound} currency={itinerary.currency} />
                )}
                <SlotCard slot={slot} index={index} currency={itinerary.currency} />
              </div>
            )
          })
        )}
      </div>
    </section>
  )
}
