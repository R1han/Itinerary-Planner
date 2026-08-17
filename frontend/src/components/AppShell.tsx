import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { useStore } from '../state/store'
import type { FamilyMember } from '../types'
import { BudgetPanel } from './BudgetPanel'
import { ChatPanel } from './ChatPanel'
import { ItineraryStrip } from './ItineraryStrip'
import { MapView } from './MapView'
import { ThreadRail } from './ThreadRail'
import { TopBar } from './TopBar'
import { EventsPanel, FamilyPanel } from './SidePanels'

export function AppShell() {
  const itinerary = useStore((s) => s.itinerary)
  const loading = useStore((s) => s.loadingItinerary)
  const panel = useStore((s) => s.panel)
  const setPanel = useStore((s) => s.setPanel)
  const error = useStore((s) => s.error)
  const setError = useStore((s) => s.setError)
  const sheetOpen = useStore((s) => s.sheetOpen)
  const setEditingSlot = useStore((s) => s.setEditingSlot)

  const [family, setFamily] = useState<FamilyMember[]>([])
  const loadFamily = () => api.family().then(setFamily).catch(() => undefined)

  useEffect(() => {
    void loadFamily()
  }, [])

  // Escape closes whichever transient surface is open, innermost first.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      const state = useStore.getState()
      if (state.editingSlotId !== null) setEditingSlot(null)
      else if (state.pendingPreference) state.setPendingPreference(null)
      else if (state.panel) setPanel(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [setEditingSlot, setPanel])

  return (
    <div className="shell">
      <TopBar family={family} />

      <div className="workspace">
        <aside className={`leftpane${sheetOpen ? ' leftpane--open' : ''}`}>
          <ThreadRail />
          <ChatPanel />
        </aside>

        <main className="rightpane" style={{ position: 'relative' }}>
          <MapView />

          {loading ? (
            <div className="empty">
              <div className="empty__inner">
                <p>Loading your plan…</p>
              </div>
            </div>
          ) : itinerary ? (
            <ItineraryStrip />
          ) : (
            <div className="empty">
              <div className="empty__inner">
                <h2>No plan yet</h2>
                <p>
                  Ask Rihla &ldquo;what events are upcoming?&rdquo; and it will offer to plan one.
                </p>
                <button className="btn btn--primary" onClick={() => setPanel('events')}>
                  Manage events
                </button>
              </div>
            </div>
          )}

          <BudgetPanel />

          {panel === 'events' && <EventsPanel />}
          {panel === 'family' && <FamilyPanel onSaved={loadFamily} />}
        </main>
      </div>

      {error && (
        <div
          role="alert"
          className="error-text"
          style={{ position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)', zIndex: 90 }}
          onClick={() => setError(null)}
        >
          {error} — dismiss
        </div>
      )}
    </div>
  )
}
