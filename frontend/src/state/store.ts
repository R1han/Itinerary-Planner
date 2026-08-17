/** Zustand store. Holds the session, the active plan, and the selection state the strip and the
 *  map both read — hovering a card highlights its pin and vice versa (spec §9). */

import { create } from 'zustand'
import { api, token } from '../api/client'
import type {
  Budget,
  ChatMessage,
  Conversation,
  Day,
  Itinerary,
  Suggestion,
  User,
} from '../types'

export interface PendingPreference {
  slotId: number
  placeName: string
  category: string
}

interface State {
  // session
  user: User | null
  booting: boolean
  llmAvailable: boolean
  mapsAvailable: boolean

  // plan
  itinerary: Itinerary | null
  loadingItinerary: boolean

  // conversation
  conversations: Conversation[]
  conversationId: number | null
  messages: ChatMessage[]
  streaming: boolean
  streamedText: string
  notice: string | null

  // selection shared by strip and map
  selectedDay: number
  hoveredSlotId: number | null
  selectedSlotId: number | null
  editingSlotId: number | null
  pendingPreference: PendingPreference | null

  // ui
  sheetOpen: boolean
  panel: 'events' | 'family' | 'plan' | null
  error: string | null
}

interface Actions {
  boot: () => Promise<void>
  signIn: (accessToken: string, user: User) => Promise<void>
  signOut: () => void

  loadItinerary: (id: number) => Promise<void>
  setItinerary: (itinerary: Itinerary) => void
  applyDayPatch: (day: Day, budget: Budget, suggestions: Suggestion[]) => void
  applyBudget: (budget: Budget) => void

  refreshConversations: () => Promise<void>
  selectConversation: (id: number) => Promise<void>
  startConversation: () => Promise<void>
  appendMessage: (message: ChatMessage) => void
  setStreaming: (streaming: boolean) => void
  setStreamedText: (updater: (previous: string) => string) => void
  setNotice: (notice: string | null) => void

  setSelectedDay: (day: number) => void
  setHoveredSlot: (id: number | null) => void
  setSelectedSlot: (id: number | null) => void
  setEditingSlot: (id: number | null) => void
  setPendingPreference: (pending: PendingPreference | null) => void

  setSheetOpen: (open: boolean) => void
  setPanel: (panel: State['panel']) => void
  setError: (error: string | null) => void
}

export const useStore = create<State & Actions>((set, get) => ({
  user: null,
  booting: true,
  llmAvailable: true,
  mapsAvailable: true,

  itinerary: null,
  loadingItinerary: false,

  conversations: [],
  conversationId: null,
  messages: [],
  streaming: false,
  streamedText: '',
  notice: null,

  selectedDay: 0,
  hoveredSlotId: null,
  selectedSlotId: null,
  editingSlotId: null,
  pendingPreference: null,

  sheetOpen: false,
  panel: null,
  error: null,

  async boot() {
    // Report which integrations are live so the UI can explain its own degradations rather than
    // silently behaving differently.
    api
      .health()
      .then((health) =>
        set({ llmAvailable: health.openai, mapsAvailable: health.openrouteservice }),
      )
      .catch(() => undefined)

    if (!token.get()) {
      set({ booting: false })
      return
    }
    try {
      const user = await api.me()
      set({ user })
      await get().refreshConversations()

      const plans = await api.itineraries()
      if (plans.length) await get().loadItinerary(plans[0].id)
    } catch {
      token.clear()
      set({ user: null })
    } finally {
      set({ booting: false })
    }
  },

  async signIn(accessToken, user) {
    token.set(accessToken)
    set({ user, error: null })
    await get().refreshConversations()
    const plans = await api.itineraries().catch(() => [])
    if (plans.length) await get().loadItinerary(plans[0].id)
  },

  signOut() {
    token.clear()
    set({
      user: null,
      itinerary: null,
      conversations: [],
      conversationId: null,
      messages: [],
      selectedDay: 0,
      hoveredSlotId: null,
      selectedSlotId: null,
      editingSlotId: null,
      pendingPreference: null,
      panel: null,
      error: null,
    })
  },

  async loadItinerary(id) {
    set({ loadingItinerary: true })
    try {
      const itinerary = await api.itinerary(id)
      set({ itinerary, selectedDay: 0, selectedSlotId: null, editingSlotId: null })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Could not load that plan.' })
    } finally {
      set({ loadingItinerary: false })
    }
  },

  setItinerary(itinerary) {
    const { selectedDay } = get()
    set({
      itinerary,
      selectedDay: Math.min(selectedDay, Math.max(0, itinerary.days.length - 1)),
      editingSlotId: null,
    })
  },

  applyDayPatch(day, budget, suggestions) {
    const current = get().itinerary
    if (!current) return
    set({
      itinerary: {
        ...current,
        days: current.days.map((existing) =>
          existing.day_index === day.day_index ? day : existing,
        ),
        budget,
        suggestions,
      },
      editingSlotId: null,
    })
  },

  applyBudget(budget) {
    const current = get().itinerary
    if (current) set({ itinerary: { ...current, budget } })
  },

  async refreshConversations() {
    try {
      const conversations = await api.conversations()
      set({ conversations })
      if (!get().conversationId && conversations.length) {
        await get().selectConversation(conversations[0].id)
      }
    } catch {
      /* the rail is not worth failing the boot over */
    }
  },

  async selectConversation(id) {
    set({ conversationId: id, streamedText: '', notice: null })
    try {
      const [messages] = await Promise.all([api.messages(id), api.markSeen(id)])
      set({
        messages,
        conversations: get().conversations.map((c) =>
          c.id === id ? { ...c, unread: false } : c,
        ),
      })
      const conversation = get().conversations.find((c) => c.id === id)
      if (conversation?.itinerary_id && conversation.itinerary_id !== get().itinerary?.id) {
        await get().loadItinerary(conversation.itinerary_id)
      }
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Could not open that thread.' })
    }
  },

  async startConversation() {
    try {
      const conversation = await api.createConversation()
      set({
        conversations: [conversation, ...get().conversations],
        conversationId: conversation.id,
        messages: [],
        streamedText: '',
        itinerary: null,
        panel: null,
      })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Could not start a new plan.' })
    }
  },

  appendMessage(message) {
    set({ messages: [...get().messages, message] })
  },

  setStreaming(streaming) {
    set({ streaming })
  },

  setStreamedText(updater) {
    set({ streamedText: updater(get().streamedText) })
  },

  setNotice(notice) {
    set({ notice })
  },

  setSelectedDay(selectedDay) {
    set({ selectedDay, selectedSlotId: null, editingSlotId: null })
  },
  setHoveredSlot(hoveredSlotId) {
    set({ hoveredSlotId })
  },
  setSelectedSlot(selectedSlotId) {
    set({ selectedSlotId })
  },
  setEditingSlot(editingSlotId) {
    set({ editingSlotId, selectedSlotId: editingSlotId ?? get().selectedSlotId })
  },
  setPendingPreference(pendingPreference) {
    set({ pendingPreference })
  },

  setSheetOpen(sheetOpen) {
    set({ sheetOpen })
  },
  setPanel(panel) {
    set({ panel })
  },
  setError(error) {
    set({ error })
  },
}))

/** The day currently shown in the strip and filtered on the map. */
export function useActiveDay(): Day | null {
  return useStore((state) => state.itinerary?.days[state.selectedDay] ?? null)
}
